//! AudioSeal-compatible ONNX watermark detector.

use crate::SampleRate;
use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::error::{map_model_load_err, map_model_open_io};
use crate::{AfpError, Result};

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
    ///   `samples.len()`.
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

/// Number of concretised input-length plans kept alive simultaneously.
///
/// # Memory tradeoff
///
/// Concretising does `self.model.clone()`, and tract's clone is a **deep**
/// copy (see the same note in `neural::embedder`), so each cached plan owns
/// its own copy of the model weights. This cap therefore multiplies the
/// detector's resident model memory by up to `MAX_CACHED_PLANS`. Four is a
/// deliberate compromise: it covers the common "a few fixed clip lengths"
/// workload, which is exactly the case that used to thrash a single slot,
/// without letting a caller with many distinct lengths grow memory without
/// bound. Raising it trades memory for rebuild latency; lowering it to 1
/// restores the old behaviour exactly.
const MAX_CACHED_PLANS: usize = 4;

/// AudioSeal-style watermark detector.
///
/// The loaded ONNX model is held in `InferenceModel` form with no fixed
/// input shape, and plans are cached per input length (up to four, least
/// recently used evicted first). A repeated input length reuses its plan; a
/// new length is concretised on demand — no cryptic Tract shape error
/// reaches the caller. For best performance, batch at a fixed length.
///
/// Each cached plan holds its own deep copy of the model weights, so
/// resident memory scales with the number of distinct input lengths in
/// flight (bounded by the cache size).
///
/// [`detect`]: WatermarkDetector::detect
pub struct WatermarkDetector {
    cfg: WatermarkConfig,
    model: InferenceModel,
    /// Concretised plans paired with the input length each was built for,
    /// in **LRU order: least-recently-used first, most-recently-used last**.
    /// A hit is moved to the back; a miss is pushed and, at
    /// [`MAX_CACHED_PLANS`], evicts the front.
    ///
    /// [`detect`]: WatermarkDetector::detect
    plans: Vec<(usize, Runnable)>,
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
            plans: Vec::new(),
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
    /// - [`AfpError::UnsupportedSampleRate`] — `rate` differs from
    ///   `cfg.sample_rate`.
    /// - [`AfpError::AudioTooShort`] — empty input buffer.
    /// - [`AfpError::Inference`] — Tract failed at any of: shape inference,
    ///   typing, building the runnable plan, running inference, or extracting
    ///   the output tensors. The variant payload identifies which step.
    pub fn detect(&mut self, samples: &[f32], rate: SampleRate) -> Result<WatermarkResult> {
        if rate.hz() != self.cfg.sample_rate {
            return Err(AfpError::UnsupportedSampleRate(rate.hz()));
        }
        let n = samples.len();
        if let Some(limit) = self.cfg.max_input_samples
            && n > limit
        {
            return Err(AfpError::InputTooLarge { limit, provided: n });
        }
        if n == 0 {
            return Err(AfpError::AudioTooShort { needed: 1, got: 0 });
        }
        crate::pcm::reject_non_finite(samples)?;

        // Build [1, 1, T] f32 input tensor without going through ndarray.
        let input_tensor = Tensor::from_shape(&[1, 1, n], samples)
            .map_err(|e| AfpError::Inference(format!("input shape: {e}")))?;

        // LRU plan lookup: a hit is served from cache and promoted to the
        // back; a miss concretises a new plan and evicts the LRU entry at
        // the cap. A single slot (the previous behaviour) thrashed whenever
        // two lengths alternated, rebuilding the ~1.3 ms tract plan on every
        // call (audit §4.2 #11).
        match self.plans.iter().position(|(len, _)| *len == n) {
            Some(i) => {
                let hit = self.plans.remove(i);
                self.plans.push(hit);
            }
            None => {
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
                    .into_optimized()
                    .map_err(|e| AfpError::Inference(format!("optimize: {e}")))?
                    .into_runnable()
                    .map_err(|e| AfpError::Inference(format!("runnable: {e}")))?;
                if self.plans.len() >= MAX_CACHED_PLANS {
                    // Front == least recently used.
                    self.plans.remove(0);
                }
                self.plans.push((n, runnable));
            }
        }

        let runnable = &self
            .plans
            .last()
            .expect("a plan was served or inserted above")
            .1;

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

    /// Like [`detect`](Self::detect), but validates detection scores before
    /// returning them: non-finite scores become [`AfpError::Inference`],
    /// and finite scores are clamped to `[0, 1]` for both
    /// [`WatermarkResult::confidence`] and [`WatermarkResult::localization`].
    /// `detected` is recomputed from the clamped confidence.
    ///
    /// Message decoding retains the raw model contract; this method does not
    /// validate message logits. The legacy detection method is unchanged.
    pub fn detect_validated(
        &mut self,
        samples: &[f32],
        rate: SampleRate,
    ) -> Result<WatermarkResult> {
        let mut result = self.detect(samples, rate)?;
        if !result.confidence.is_finite() {
            return Err(AfpError::Inference(format!(
                "non-finite confidence ({})",
                result.confidence
            )));
        }
        result.confidence = result.confidence.clamp(0.0, 1.0);
        for (i, v) in result.localization.iter_mut().enumerate() {
            if !v.is_finite() {
                return Err(AfpError::Inference(format!(
                    "non-finite localization score at index {i}"
                )));
            }
            *v = v.clamp(0.0, 1.0);
        }
        result.detected = result.confidence > self.cfg.threshold;
        Ok(result)
    }

    /// Alias for [`detect_validated`](Self::detect_validated).
    pub fn detect_checked(&mut self, samples: &[f32], rate: SampleRate) -> Result<WatermarkResult> {
        self.detect_validated(samples, rate)
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

    /// The positive path: load a real ONNX file, build the plan, run, and
    /// decode both outputs.
    ///
    /// This closes the §4.3 #13 gap — every pre-existing test above asserts a
    /// construction or validation *error*, so `detect()`'s body (plan
    /// building, the ≥2-output check, the confidence mean, the LSB-first
    /// message decode) had no coverage at all. The fixture is an identity
    /// model, so every expected value below is exact rather than a range.
    #[test]
    fn detect_runs_identity_model_and_decodes_both_outputs() {
        let path = crate::watermark::test_fixture::write_identity_onnx("positive");
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load identity onnx");

        // Identity → output 0 == input, so confidence is the input's mean and
        // output 1 == input, so bit i of the message is `sample[i] >= 0.0`.
        let samples: Vec<f32> = vec![0.8; 4096];
        let rate = SampleRate::new(16_000).expect("rate");
        let r = detector.detect(&samples, rate).expect("detect");

        assert!(r.detected, "mean 0.8 is above the 0.5 default threshold");
        assert!(
            (r.confidence - 0.8).abs() < 1e-4,
            "confidence should be the input mean, got {}",
            r.confidence
        );
        assert_eq!(r.localization.len(), samples.len());
        // All logits are +0.8 ≥ 0 → every bit set.
        assert_eq!(r.message, 0xFFFF, "all-positive logits set all 16 bits");

        crate::watermark::test_fixture::cleanup(&path);
    }

    /// A negative-input run must decode a zero message and not detect.
    #[test]
    fn detect_negative_input_decodes_zero_message() {
        let path = crate::watermark::test_fixture::write_identity_onnx("negative");
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load identity onnx");

        let samples: Vec<f32> = vec![-0.9; 4096];
        let rate = SampleRate::new(16_000).expect("rate");
        let r = detector.detect(&samples, rate).expect("detect");

        assert!(!r.detected, "mean -0.9 is below the 0.5 default threshold");
        assert_eq!(r.message, 0, "all-negative logits clear every bit");

        crate::watermark::test_fixture::cleanup(&path);
    }

    /// §4.2 #11 — the plan cache must serve repeated lengths and survive
    /// interleaving.
    ///
    /// The cache is single-slot, so alternating lengths rebuild the tract plan
    /// every call. Whatever the eviction policy, the *results* must not depend
    /// on how many distinct lengths were seen first, and same-length repeats
    /// must not degrade. Correctness here is what makes the LRU change safe.
    #[test]
    fn detect_is_correct_across_interleaved_input_lengths() {
        let path = crate::watermark::test_fixture::write_identity_onnx("lengths");
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load identity onnx");
        let rate = SampleRate::new(16_000).expect("rate");

        // Three distinct lengths, interleaved twice. Every pass over a given
        // length must agree with the first pass, whether the plan came from
        // the cache or a rebuild.
        let lengths = [2048usize, 4096, 1024];
        let mut first: Vec<(f32, u32, usize)> = Vec::new();
        for round in 0..2 {
            for &n in &lengths {
                let samples: Vec<f32> = vec![0.75; n];
                let r = detector.detect(&samples, rate).expect("detect");
                if round == 0 {
                    first.push((r.confidence, r.message, r.localization.len()));
                } else {
                    let (conf, msg, loc) = first[lengths.iter().position(|&x| x == n).unwrap()];
                    assert!(
                        (r.confidence - conf).abs() < 1e-6,
                        "confidence drifted for n={n} after interleaving"
                    );
                    assert_eq!(r.message, msg, "message drifted for n={n}");
                    assert_eq!(r.localization.len(), loc, "localization drifted for n={n}");
                }
            }
        }

        crate::watermark::test_fixture::cleanup(&path);
    }

    #[test]
    fn detect_validated_clamps_out_of_range_confidence() {
        let path = crate::watermark::test_fixture::write_identity_onnx("validated");
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load identity onnx");
        let samples: Vec<f32> = vec![-0.9; 4096];
        let rate = SampleRate::new(16_000).expect("rate");
        let raw = detector.detect(&samples, rate).expect("detect");
        assert!(raw.confidence < 0.0);

        let validated = detector
            .detect_validated(&samples, rate)
            .expect("detect_validated");
        assert!((0.0..=1.0).contains(&validated.confidence));
        assert_eq!(validated.confidence, 0.0);
        assert!(!validated.detected);
        assert_eq!(validated.localization.len(), samples.len());
        assert!(validated.localization.iter().all(|&v| v == 0.0));

        crate::watermark::test_fixture::cleanup(&path);
    }

    #[test]
    fn detect_checked_accepts_in_range_toy_onnx_output() {
        let n = 1024usize;
        let path =
            crate::watermark::test_fixture::write_constant_detection_onnx("checked-good", n, 0.75);
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load constant-det onnx");
        let samples: Vec<f32> = vec![0.1; n];
        let rate = SampleRate::new(16_000).expect("rate");
        let r = detector
            .detect_checked(&samples, rate)
            .expect("detect_checked");
        assert!(r.detected);
        assert!((r.confidence - 0.75).abs() < 1e-5);
        assert_eq!(r.localization.len(), n);
        assert!(r.localization.iter().all(|&v| (v - 0.75).abs() < 1e-5));
        crate::watermark::test_fixture::cleanup(&path);
    }

    #[test]
    fn detect_checked_rejects_nonfinite_toy_onnx_output() {
        let n = 512usize;
        let path = crate::watermark::test_fixture::write_constant_detection_onnx(
            "checked-nan",
            n,
            f32::NAN,
        );
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load nan-det onnx");
        let samples: Vec<f32> = vec![0.2; n];
        let rate = SampleRate::new(16_000).expect("rate");
        match detector.detect_checked(&samples, rate) {
            Err(AfpError::Inference(msg)) => {
                assert!(msg.contains("non-finite"), "unexpected: {msg}");
            }
            Ok(r) => panic!("expected Inference for NaN model output, got {r:?}"),
            Err(e) => panic!("expected Inference, got {e:?}"),
        }
        crate::watermark::test_fixture::cleanup(&path);
    }

    #[test]
    fn detect_checked_clamps_above_one_toy_onnx_output() {
        let n = 256usize;
        let path =
            crate::watermark::test_fixture::write_constant_detection_onnx("checked-high", n, 1.5);
        let mut detector =
            WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
                .expect("load high-det onnx");
        let samples: Vec<f32> = vec![0.3; n];
        let rate = SampleRate::new(16_000).expect("rate");
        let r = detector
            .detect_checked(&samples, rate)
            .expect("detect_checked");
        assert_eq!(r.confidence, 1.0);
        assert!(r.detected);
        assert!(r.localization.iter().all(|&v| v == 1.0));
        crate::watermark::test_fixture::cleanup(&path);
    }
}
