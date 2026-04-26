//! AudioSeal-compatible ONNX watermark detector.

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::{AfpError, AudioBuffer, Result};

/// Tunable parameters for [`WatermarkDetector`].
///
/// `model_path` must point at an ONNX file whose first input accepts
/// `[1, 1, T] f32` audio waveforms at `sample_rate`, and which exposes
/// at least two outputs in this order:
///
/// 1. **detection scores** — per-sample (or per-frame) probabilities
///    in `[0, 1]`. Used to compute mean confidence.
/// 2. **message logits** — `f32` logits for the embedded message bits;
///    bits are recovered as `logit ≥ 0`.
#[derive(Clone, Debug)]
pub struct WatermarkConfig {
    /// Filesystem path to the ONNX model.
    pub model_path: String,
    /// Number of message bits the model encodes (≤ 32). Default 16.
    pub message_bits: u8,
    /// Detection threshold on the mean detection score; above this the
    /// audio is considered watermarked. Default 0.5.
    pub threshold: f32,
    /// Sample rate the model expects, in Hz. Default 16 000 (AudioSeal).
    pub sample_rate: u32,
}

impl WatermarkConfig {
    /// Build a config with the given model path and AudioSeal defaults
    /// (`message_bits = 16`, `threshold = 0.5`, `sample_rate = 16_000`).
    #[must_use]
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            message_bits: 16,
            threshold: 0.5,
            sample_rate: 16_000,
        }
    }
}

/// One detection result from [`WatermarkDetector::detect`].
#[derive(Clone, Debug)]
pub struct WatermarkResult {
    /// `true` if the mean detection score exceeds `WatermarkConfig::threshold`.
    pub detected: bool,
    /// Mean detection score over the input, in `[0, 1]`.
    pub confidence: f32,
    /// Decoded message bits packed LSB-first into a `u32`. The low
    /// `message_bits` are populated; bits at or above `message_bits` are
    /// zero. Zero when the model returned fewer logits than requested.
    pub message: u32,
    /// Raw per-output detection scores, exactly as the model emitted
    /// them (no resampling). Length depends on the specific model.
    pub localization: Vec<f32>,
}

/// AudioSeal-style watermark detector.
///
/// The loaded ONNX model is held in `InferenceModel` form with no fixed
/// input shape; each [`detect`] call concretises the input length, runs
/// the optimisation pipeline, and emits a result. That's not the cheapest
/// possible path — for a hot-loop use case where buffers are always the
/// same length, prefer batching multiple calls under a single
/// [`WatermarkDetector`] instance, where `tract` will cache the optimised
/// plan after the first call.
///
/// [`detect`]: WatermarkDetector::detect
pub struct WatermarkDetector {
    cfg: WatermarkConfig,
    model: InferenceModel,
}

impl WatermarkDetector {
    /// Validate `cfg` and load the ONNX file at `cfg.model_path`.
    pub fn new(cfg: WatermarkConfig) -> Result<Self> {
        if cfg.message_bits > 32 {
            return Err(AfpError::Config(format!(
                "message_bits ({}) > 32",
                cfg.message_bits,
            )));
        }
        if !(0.0..=1.0).contains(&cfg.threshold) {
            return Err(AfpError::Config(format!(
                "threshold {} not in [0, 1]",
                cfg.threshold,
            )));
        }
        if cfg.sample_rate == 0 {
            return Err(AfpError::Config("sample_rate must be > 0".to_string()));
        }
        if cfg.model_path.is_empty() {
            return Err(AfpError::ModelNotFound(String::new()));
        }

        let path = Path::new(&cfg.model_path);
        if !path.exists() {
            return Err(AfpError::ModelNotFound(cfg.model_path.clone()));
        }

        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(|e| AfpError::ModelLoad(format!("load: {e}")))?;

        Ok(Self { cfg, model })
    }

    /// Borrow the configuration this detector was built with.
    #[must_use]
    pub fn config(&self) -> &WatermarkConfig {
        &self.cfg
    }

    /// Run the watermark detector on `audio`.
    ///
    /// Errors are returned (rather than panicking) for: a sample-rate
    /// mismatch with the model, an empty input buffer, or any failure
    /// inside `tract` while concretising / running the model.
    pub fn detect(&mut self, audio: AudioBuffer<'_>) -> Result<WatermarkResult> {
        if audio.rate.hz() != self.cfg.sample_rate {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        let n = audio.samples.len();
        if n == 0 {
            return Err(AfpError::AudioTooShort { needed: 1, got: 0 });
        }

        // Build [1, 1, T] f32 input tensor without going through ndarray.
        let input_tensor = Tensor::from_shape(&[1, 1, n], audio.samples)
            .map_err(|e| AfpError::Inference(format!("input shape: {e}")))?;

        // Concretise input shape and prepare a runnable plan.
        let runnable = self
            .model
            .clone()
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, n)),
            )
            .map_err(|e| AfpError::Inference(format!("input fact: {e}")))?
            .into_typed()
            .map_err(|e| AfpError::Inference(format!("type: {e}")))?
            .into_runnable()
            .map_err(|e| AfpError::Inference(format!("runnable: {e}")))?;

        let outputs = runnable
            .run(tvec!(input_tensor.into()))
            .map_err(|e| AfpError::Inference(format!("run: {e}")))?;

        if outputs.len() < 2 {
            return Err(AfpError::Inference(format!(
                "expected ≥ 2 outputs (detection, message), got {}",
                outputs.len(),
            )));
        }

        // Output 0: detection scores → localization + mean confidence.
        let detection = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("detection view: {e}")))?;
        let localization: Vec<f32> = detection.iter().copied().collect();
        let confidence = if localization.is_empty() {
            0.0
        } else {
            localization.iter().sum::<f32>() / localization.len() as f32
        };
        let detected = confidence > self.cfg.threshold;

        // Output 1: message bit logits → packed u32 (LSB-first).
        let message_view = outputs[1]
            .to_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("message view: {e}")))?;
        let bits = self.cfg.message_bits.min(32) as usize;
        let mut message: u32 = 0;
        if message_view.len() >= bits {
            for (i, &logit) in message_view.iter().take(bits).enumerate() {
                if logit >= 0.0 {
                    message |= 1u32 << i;
                }
            }
        }

        Ok(WatermarkResult {
            detected,
            confidence,
            message,
            localization,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unique_path(stem: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "audiofp-watermark-test-{}-{}-{n}.bin",
            std::process::id(),
            stem,
        ))
    }

    #[test]
    fn empty_model_path_returns_model_not_found() {
        let res = WatermarkDetector::new(WatermarkConfig::new(""));
        match res {
            Err(AfpError::ModelNotFound(_)) => {}
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
            Err(e) => panic!("expected ModelNotFound, got {e:?}"),
        }
    }

    #[test]
    fn missing_model_returns_model_not_found() {
        let res = WatermarkDetector::new(WatermarkConfig::new(
            "/nonexistent/path/to/audioseal.onnx",
        ));
        match res {
            Err(AfpError::ModelNotFound(_)) => {}
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
            Err(e) => panic!("expected ModelNotFound, got {e:?}"),
        }
    }

    #[test]
    fn message_bits_above_32_is_rejected() {
        let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
        cfg.message_bits = 33;
        match WatermarkDetector::new(cfg) {
            Err(AfpError::Config(_)) => {}
            Ok(_) => panic!("expected Config error, got Ok"),
            Err(e) => panic!("expected Config error, got {e:?}"),
        }
    }

    #[test]
    fn threshold_outside_unit_interval_is_rejected() {
        for bad in [-0.5_f32, 1.1, -1.0] {
            let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
            cfg.threshold = bad;
            match WatermarkDetector::new(cfg) {
                Err(AfpError::Config(_)) => {}
                Ok(_) => panic!("expected Config for threshold={bad}, got Ok"),
                Err(e) => panic!("expected Config for threshold={bad}, got {e:?}"),
            }
        }
    }

    #[test]
    fn zero_sample_rate_is_rejected() {
        let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
        cfg.sample_rate = 0;
        match WatermarkDetector::new(cfg) {
            Err(AfpError::Config(_)) => {}
            Ok(_) => panic!("expected Config error, got Ok"),
            Err(e) => panic!("expected Config error, got {e:?}"),
        }
    }

    #[test]
    fn corrupt_onnx_returns_model_load_error() {
        let path = unique_path("corrupt");
        // Write 64 bytes of garbage that definitely is not a valid ONNX
        // protobuf.
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let garbage = [0xAA_u8; 64];
            f.write_all(&garbage).unwrap();
        }
        let res =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()));
        std::fs::remove_file(&path).ok();
        match res {
            Err(AfpError::ModelLoad(_)) => {}
            Ok(_) => panic!("expected ModelLoad, got Ok"),
            Err(e) => panic!("expected ModelLoad, got {e:?}"),
        }
    }

    #[test]
    fn config_constructor_uses_audioseal_defaults() {
        let cfg = WatermarkConfig::new("model.onnx");
        assert_eq!(cfg.message_bits, 16);
        assert_eq!(cfg.threshold, 0.5);
        assert_eq!(cfg.sample_rate, 16_000);
    }
}
