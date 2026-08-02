//! AudioSeal-compatible ONNX watermark detector.

use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::error::{map_model_load_err, map_model_open_io};
use crate::{AfpError, AudioBuffer, Result};

/// Type alias for the compiled runnable plan produced by
/// `TypedModel::into_runnable()`. Cached to avoid rebuilding the
/// execution plan on every `detect()` call.
type Runnable = Arc<TypedSimplePlan>;

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
    /// Maximum input sample count accepted by [`detect`]. `None` disables
    /// the check (default). When set, inputs exceeding this cap are
    /// rejected with [`AfpError::InputTooLarge`] before any inference.
    ///
    /// [`detect`]: WatermarkDetector::detect
    pub max_input_samples: Option<usize>,
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
            max_input_samples: None,
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
    /// Raw detection scores from the model's first ONNX output, **flattened**
    /// with no resampling or time-axis remapping by `audiofp`.
    ///
    /// # Contract
    ///
    /// - **Values:** model-emitted `f32` scores (AudioSeal-style detectors
    ///   typically emit probabilities in `[0, 1]`).
    /// - **Length:** exactly the number of elements Tract yields when
    ///   flattening output `[0]`. This is **not** guaranteed equal to
    ///   `audio.samples.len()`.
    /// - **Time base:** model-dependent. Some AudioSeal exports emit one
    ///   score per input sample at [`WatermarkConfig::sample_rate`]; others
    ///   emit coarser per-frame / pooled maps. Treat hop and alignment as
    ///   part of the **model card**, not as a stable `audiofp` API promise.
    /// - **Aggregation:** [`Self::confidence`] is the arithmetic mean of
    ///   these scores (or `0.0` if empty); [`Self::detected`] compares that
    ///   mean to [`WatermarkConfig::threshold`].
    /// - **Stability:** tensor shape is **not** semver-guaranteed across
    ///   model versions — only that this field forwards whatever output
    ///   `[0]` contains.
    ///
    /// For “where is the watermark?”, threshold or plot this vector against
    /// the model's documented time base. Do not assume index `i` maps to
    /// sample `i` unless your specific ONNX export says so.
    pub localization: Vec<f32>,
}

/// AudioSeal-style watermark detector.
///
/// The loaded ONNX model is held in `InferenceModel` form with no fixed
/// input shape. The first [`detect`] call concretises the input length
/// and caches a typed model; subsequent calls of the **same input
/// length** reuse that typed plan. If the input length changes, the
/// detector transparently rebuilds the typed plan for the new length —
/// no cryptic Tract shape error reaches the caller. For best
/// performance, prefer batching at a fixed length.
///
/// [`detect`]: WatermarkDetector::detect
pub struct WatermarkDetector {
    cfg: WatermarkConfig,
    model: InferenceModel,
    /// Cached runnable plan paired with the input length it was built
    /// for. On a length mismatch the cache is rebuilt; equal-length
    /// repeat calls reuse the existing plan without cloning.
    ///
    /// [`detect`]: WatermarkDetector::detect
    cached: Option<(usize, Runnable)>,
}

impl WatermarkDetector {
    /// Validate `cfg` and load the ONNX file at `cfg.model_path`.
    ///
    /// The model is loaded in `InferenceModel` form with no fixed input
    /// shape. The first [`detect`] call concretises the model for that
    /// input length and caches the typed plan; later calls reuse it.
    ///
    /// [`detect`]: WatermarkDetector::detect
    ///
    /// # Errors
    ///
    /// - [`AfpError::Config`] — `message_bits > 32`, `threshold` outside
    ///   `[0, 1]`, or `sample_rate == 0`.
    /// - [`AfpError::ModelNotFound`] — `model_path` is empty or points at
    ///   a file that doesn't exist.
    /// - [`AfpError::ModelLoad`] — the file exists but Tract couldn't
    ///   parse it as an ONNX protobuf.
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
        if let Err(e) = std::fs::File::open(path) {
            return Err(map_model_open_io(&cfg.model_path, e));
        }
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(map_model_load_err)?;

        Ok(Self {
            cfg,
            model,
            cached: None,
        })
    }

    /// Borrow the configuration this detector was built with.
    #[must_use]
    pub fn config(&self) -> &WatermarkConfig {
        &self.cfg
    }

    /// Run the watermark detector on `audio`.
    ///
    /// Builds a `[1, 1, T] f32` input tensor from the buffer's samples,
    /// concretises the loaded model for that input length, runs
    /// inference, and decodes the model's two outputs into a
    /// [`WatermarkResult`].
    ///
    /// # Errors
    ///
    /// - [`AfpError::UnsupportedSampleRate`] — `audio.rate` differs from
    ///   `cfg.sample_rate`.
    /// - [`AfpError::AudioTooShort`] — empty input buffer.
    /// - [`AfpError::Inference`] — Tract failed at any of: shape inference,
    ///   typing, building the runnable plan, running inference, or extracting
    ///   the output tensors. The variant payload identifies which step.
    pub fn detect(&mut self, audio: AudioBuffer<'_>) -> Result<WatermarkResult> {
        crate::pcm::reject_non_finite(audio.samples)?;
        if audio.rate.hz() != self.cfg.sample_rate {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        let n = audio.samples.len();
        if self.cfg.max_input_samples.is_some_and(|limit| n > limit) {
            return Err(AfpError::InputTooLarge {
                limit: self.cfg.max_input_samples.unwrap(),
                provided: n,
            });
        }
        if n == 0 {
            return Err(AfpError::AudioTooShort { needed: 1, got: 0 });
        }

        // Build [1, 1, T] f32 input tensor without going through ndarray.
        let input_tensor = Tensor::from_shape(&[1, 1, n], audio.samples)
            .map_err(|e| AfpError::Inference(format!("input shape: {e}")))?;

        // Concretise input shape and prepare a runnable plan.
        // Reuse the cached runnable when the input length matches;
        // otherwise rebuild for the new length.
        let needs_rebuild = match &self.cached {
            Some((cached_n, _)) => *cached_n != n,
            None => true,
        };

        if needs_rebuild {
            let typed = self
                .model
                .clone()
                .with_input_fact(
                    0,
                    InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1, n)),
                )
                .map_err(|e| AfpError::Inference(format!("input fact: {e}")))?
                .into_typed()
                .map_err(|e| AfpError::Inference(format!("type: {e}")))?;
            let runnable = typed
                .into_runnable()
                .map_err(|e| AfpError::Inference(format!("runnable: {e}")))?;
            self.cached = Some((n, runnable));
        }

        let runnable = &self.cached.as_ref().unwrap().1;

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
            .to_plain_array_view::<f32>()
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
            .to_plain_array_view::<f32>()
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
        let res =
            WatermarkDetector::new(WatermarkConfig::new("/nonexistent/path/to/audioseal.onnx"));
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
        let res = WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()));
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
