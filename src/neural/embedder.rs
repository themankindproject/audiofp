//! Generic ONNX log-mel embedder (offline / whole-buffer).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::dsp::mel::{MelFilterBank, MelScale};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, AudioBuffer, Fingerprinter, Result, TimestampMs};

use super::frontend::LogMelFrontend;

/// Tunable parameters for [`NeuralEmbedder`] / [`super::StreamingNeuralEmbedder`].
///
/// `model_path` must point at an ONNX model whose first input accepts a
/// `[1, n_mels, n_frames] f32` log-mel spectrogram and whose first output
/// is a flat `f32` embedding vector. `n_frames` is fully determined by
/// `(window_samples − n_fft) / hop + 1` where
/// `window_samples = round(window_secs · sample_rate)`.
#[derive(Clone, Debug)]
pub struct NeuralEmbedderConfig {
    /// Filesystem path to the ONNX model.
    pub model_path: String,
    /// Sample rate the model expects, in Hz. Default 16 000.
    pub sample_rate: u32,
    /// FFT length (must be a power of two). Default 1024.
    pub n_fft: usize,
    /// STFT hop, in samples. Default 320 (20 ms at 16 kHz).
    pub hop: usize,
    /// Number of mel bands. Default 128.
    pub n_mels: usize,
    /// Lowest frequency (Hz) covered by the mel filterbank. Default 0.
    pub fmin: f32,
    /// Highest frequency (Hz) covered by the mel filterbank. Default
    /// `sample_rate / 2`.
    pub fmax: f32,
    /// Mel scale convention. Default [`MelScale::Slaney`] (librosa default).
    pub mel_scale: MelScale,
    /// Window kind for the STFT. Default [`WindowKind::Hann`].
    pub window_kind: WindowKind,
    /// Analysis-window length in seconds. Default 1.0.
    pub window_secs: f32,
    /// Hop between successive analysis windows in seconds. Default 1.0
    /// (non-overlapping). Set lower for denser embeddings.
    pub hop_secs: f32,
    /// L2-normalise emitted embeddings. Default `true` — appropriate
    /// when downstream similarity is cosine.
    pub l2_normalize: bool,
}

impl NeuralEmbedderConfig {
    /// Build a config with the given model path and reasonable defaults
    /// (16 kHz, n_fft=1024, hop=320, 128 mels, 1 s non-overlapping
    /// windows, Slaney mel, Hann window, L2-normalised output).
    #[must_use]
    pub fn new(model_path: impl Into<String>) -> Self {
        let sr = 16_000u32;
        Self {
            model_path: model_path.into(),
            sample_rate: sr,
            n_fft: 1024,
            hop: 320,
            n_mels: 128,
            fmin: 0.0,
            fmax: sr as f32 / 2.0,
            mel_scale: MelScale::Slaney,
            window_kind: WindowKind::Hann,
            window_secs: 1.0,
            hop_secs: 1.0,
            l2_normalize: true,
        }
    }
}

/// One embedding emitted by [`NeuralEmbedder`].
#[derive(Clone, Debug)]
pub struct NeuralEmbedding {
    /// The (possibly L2-normalised) embedding vector.
    pub vector: Vec<f32>,
    /// Start of the analysis window this embedding was computed from.
    pub t_start: TimestampMs,
}

/// All embeddings produced by [`NeuralEmbedder`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct NeuralFingerprint {
    /// One entry per analysis window, in input order.
    pub embeddings: Vec<NeuralEmbedding>,
    /// Length of each embedding vector. Determined by the model at
    /// construction time.
    pub embedding_dim: usize,
    /// `1.0 / hop_secs` — convenience for downstream consumers.
    pub frames_per_sec: f32,
}

/// Tract's typed runnable model. Expensive to build; we build it once
/// in [`NeuralEmbedder::new`] and reuse it for every call.
pub(crate) type Runnable =
    SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Heavy state shared by [`NeuralEmbedder`] and
/// [`super::StreamingNeuralEmbedder`]. Both compose this rather than
/// inherit, so neither re-implements the front-end.
pub(crate) struct EmbedderCore {
    pub(crate) cfg: NeuralEmbedderConfig,
    pub(crate) frontend: LogMelFrontend,
    pub(crate) runnable: Runnable,

    /// Total samples in one analysis window (`round(window_secs · sr)`).
    pub(crate) window_samples: usize,
    /// Total samples between successive windows (`round(hop_secs · sr)`).
    pub(crate) hop_samples: usize,
    /// STFT frame count for one analysis window.
    pub(crate) n_frames: usize,
    /// Embedding dimension reported by the model on a probe call.
    pub(crate) embedding_dim: usize,
}

impl EmbedderCore {
    /// Compute one embedding from exactly `window_samples` samples of
    /// PCM at the configured sample rate, writing into a caller-managed
    /// `Vec`. The vector is `clear()`ed first; on success it has
    /// length `embedding_dim`. Reuses the existing allocation when
    /// capacity is sufficient.
    ///
    /// # Panics
    ///
    /// Panics if `window.len() != self.window_samples`.
    pub(crate) fn embed_window_into(&mut self, window: &[f32], out: &mut Vec<f32>) -> Result<()> {
        assert_eq!(
            window.len(),
            self.window_samples,
            "embed_window requires exactly window_samples"
        );

        let n_mels = self.frontend.n_mels();
        let n_frames = self.n_frames;

        // Allocate the model input tensor once and write log-mel
        // straight into its `[1, n_mels, n_frames]` row-major buffer
        // with strided writes — no intermediate `Vec` and no transpose.
        //
        // SAFETY: `Tensor::uninitialized` returns a tensor whose backing
        // buffer is uninitialised; we must overwrite every element before
        // `run()` reads it. The for_each_frame callback fires exactly
        // `n_frames` times, and for each `f` we write every
        // `m in 0..n_mels` index — covering all `n_mels * n_frames`
        // positions in the tensor.
        let mut tensor = unsafe {
            Tensor::uninitialized::<f32>(&[1, n_mels, n_frames])
                .map_err(|e| AfpError::Inference(format!("input alloc: {e}")))?
        };

        {
            let dst = tensor
                .as_slice_mut::<f32>()
                .map_err(|e| AfpError::Inference(format!("input slice: {e}")))?;
            self.frontend.for_each_frame(window, |f, mel_row| {
                // Strided write: position (m, f) in the `[n_mels, n_frames]`
                // matrix lives at `m * n_frames + f`.
                for m in 0..n_mels {
                    dst[m * n_frames + f] = mel_row[m];
                }
            });
        }

        let outputs = self
            .runnable
            .run(tvec!(tensor.into()))
            .map_err(|e| AfpError::Inference(format!("run: {e}")))?;
        if outputs.is_empty() {
            return Err(AfpError::Inference("model produced no outputs".to_string()));
        }

        let view = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("output view: {e}")))?;
        if view.len() != self.embedding_dim {
            return Err(AfpError::Inference(format!(
                "expected embedding of {} dims, got {}",
                self.embedding_dim,
                view.len(),
            )));
        }

        out.clear();
        out.reserve(self.embedding_dim);
        out.extend(view.iter().copied());

        if self.cfg.l2_normalize {
            let sumsq: f32 = out.iter().map(|x| x * x).sum();
            let norm = sumsq.sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for v in out.iter_mut() {
                    *v *= inv;
                }
            }
        }

        Ok(())
    }
}

/// Generic ONNX log-mel audio embedder (offline / whole-buffer).
///
/// See the [module docs](super) for the model contract and an example.
pub struct NeuralEmbedder {
    pub(crate) core: EmbedderCore,
}

impl NeuralEmbedder {
    /// Validate `cfg`, load + optimise the ONNX model, and run a probe
    /// inference to determine the embedding dimension.
    ///
    /// All expensive work (typing, optimisation, runnable plan
    /// construction) happens here, **once**. Subsequent calls to
    /// [`extract`] only run the front-end and the inference itself.
    ///
    /// [`extract`]: NeuralEmbedder::extract
    ///
    /// # Errors
    ///
    /// - [`AfpError::Config`] — invalid sample rate, FFT length, hop,
    ///   mel band count, frequency range, window/hop seconds, or the
    ///   derived window length is shorter than `n_fft`.
    /// - [`AfpError::ModelNotFound`] — `model_path` is empty or points
    ///   at a file that doesn't exist.
    /// - [`AfpError::ModelLoad`] — the file exists but Tract couldn't
    ///   parse it as ONNX.
    /// - [`AfpError::Inference`] — the model couldn't accept the
    ///   contracted input shape, or the probe inference produced no /
    ///   empty outputs.
    pub fn new(cfg: NeuralEmbedderConfig) -> Result<Self> {
        // --- Config validation ---------------------------------------
        if cfg.sample_rate == 0 {
            return Err(AfpError::Config("sample_rate must be > 0".to_string()));
        }
        if cfg.n_fft < 2 || !cfg.n_fft.is_power_of_two() {
            return Err(AfpError::Config(format!(
                "n_fft must be a power of two >= 2 (got {})",
                cfg.n_fft,
            )));
        }
        if cfg.hop == 0 || cfg.hop > cfg.n_fft {
            return Err(AfpError::Config(format!(
                "hop must satisfy 0 < hop <= n_fft (hop={}, n_fft={})",
                cfg.hop, cfg.n_fft,
            )));
        }
        if cfg.n_mels == 0 {
            return Err(AfpError::Config("n_mels must be > 0".to_string()));
        }
        let nyquist = cfg.sample_rate as f32 / 2.0;
        if !(cfg.fmin >= 0.0 && cfg.fmax > cfg.fmin && cfg.fmax <= nyquist) {
            return Err(AfpError::Config(format!(
                "require 0 <= fmin < fmax <= sr/2 (fmin={}, fmax={}, sr={})",
                cfg.fmin, cfg.fmax, cfg.sample_rate,
            )));
        }
        if !(cfg.window_secs > 0.0 && cfg.window_secs.is_finite()) {
            return Err(AfpError::Config(format!(
                "window_secs must be a positive finite number (got {})",
                cfg.window_secs,
            )));
        }
        if !(cfg.hop_secs > 0.0 && cfg.hop_secs.is_finite()) {
            return Err(AfpError::Config(format!(
                "hop_secs must be a positive finite number (got {})",
                cfg.hop_secs,
            )));
        }

        let window_samples = (cfg.window_secs * cfg.sample_rate as f32).round() as usize;
        let hop_samples = (cfg.hop_secs * cfg.sample_rate as f32).round() as usize;
        if window_samples < cfg.n_fft {
            return Err(AfpError::Config(format!(
                "window_samples ({}) must be >= n_fft ({})",
                window_samples, cfg.n_fft,
            )));
        }
        if hop_samples == 0 {
            return Err(AfpError::Config(
                "hop_samples must be > 0 (hop_secs * sample_rate too small)".to_string(),
            ));
        }
        // The streaming buffer drains `hop_samples` per emitted embedding
        // out of a buffer that is only guaranteed to contain
        // `window_samples` — sparse-hop sampling (hop > window) would
        // skip uncollected input and panic the drain. Reject up front.
        if hop_samples > window_samples {
            return Err(AfpError::Config(format!(
                "hop_samples ({hop_samples}) must be <= window_samples ({window_samples}); \
                 hop_secs ({}) > window_secs ({})",
                cfg.hop_secs, cfg.window_secs,
            )));
        }
        let n_frames = (window_samples - cfg.n_fft) / cfg.hop + 1;

        // --- Model loading -------------------------------------------
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

        // Concretise input shape, type, optimise, and build the runnable
        // plan — once. This is the work the watermark detector
        // (incorrectly) does per call; doing it once is the single
        // largest perf win available here.
        let runnable: Runnable = model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), tvec!(1, cfg.n_mels, n_frames)),
            )
            .map_err(|e| AfpError::Inference(format!("input fact: {e}")))?
            .into_typed()
            .map_err(|e| AfpError::Inference(format!("type: {e}")))?
            .into_optimized()
            .map_err(|e| AfpError::Inference(format!("optimize: {e}")))?
            .into_runnable()
            .map_err(|e| AfpError::Inference(format!("runnable: {e}")))?;

        // --- Front-end pre-planning ----------------------------------
        let stft_cfg = StftConfig {
            n_fft: cfg.n_fft,
            hop: cfg.hop,
            window: cfg.window_kind,
            // We always use non-centred framing for predictable n_frames
            // and zero-allocation framing in the hot loop.
            center: false,
        };
        let stft = ShortTimeFFT::new(stft_cfg);
        let mel = MelFilterBank::new(
            cfg.n_mels,
            cfg.n_fft,
            cfg.sample_rate,
            cfg.fmin,
            cfg.fmax,
            cfg.mel_scale,
        );

        // --- Probe inference to discover embedding_dim ----------------
        let probe = Tensor::from_shape(
            &[1, cfg.n_mels, n_frames],
            &vec![0.0_f32; cfg.n_mels * n_frames],
        )
        .map_err(|e| AfpError::Inference(format!("probe alloc: {e}")))?;
        let probe_out = runnable
            .run(tvec!(probe.into()))
            .map_err(|e| AfpError::Inference(format!("probe run: {e}")))?;
        if probe_out.is_empty() {
            return Err(AfpError::Inference(
                "model produced no outputs on probe".to_string(),
            ));
        }
        let probe_view = probe_out[0]
            .to_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("probe view: {e}")))?;
        let embedding_dim = probe_view.len();
        if embedding_dim == 0 {
            return Err(AfpError::Inference(
                "model produced empty embedding on probe".to_string(),
            ));
        }

        let frontend = LogMelFrontend::new(stft, mel, window_samples);

        Ok(Self {
            core: EmbedderCore {
                cfg,
                frontend,
                runnable,
                window_samples,
                hop_samples,
                n_frames,
                embedding_dim,
            },
        })
    }

    /// Embedding dimension reported by the model.
    #[must_use]
    pub fn embedding_dim(&self) -> usize {
        self.core.embedding_dim
    }

    /// Number of samples in one analysis window.
    #[must_use]
    pub fn window_samples(&self) -> usize {
        self.core.window_samples
    }

    /// Number of samples between successive analysis windows.
    #[must_use]
    pub fn hop_samples(&self) -> usize {
        self.core.hop_samples
    }
}

impl Fingerprinter for NeuralEmbedder {
    type Output = NeuralFingerprint;
    type Config = NeuralEmbedderConfig;

    fn name(&self) -> &'static str {
        "neural-onnx-v0"
    }

    fn config(&self) -> &Self::Config {
        &self.core.cfg
    }

    fn required_sample_rate(&self) -> u32 {
        self.core.cfg.sample_rate
    }

    fn min_samples(&self) -> usize {
        self.core.window_samples
    }

    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
        if audio.rate.hz() != self.core.cfg.sample_rate {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        if audio.samples.len() < self.core.window_samples {
            return Err(AfpError::AudioTooShort {
                needed: self.core.window_samples,
                got: audio.samples.len(),
            });
        }

        let sr = audio.rate.hz() as u64;
        let window_samples = self.core.window_samples;
        let hop_samples = self.core.hop_samples;
        let embedding_dim = self.core.embedding_dim;

        // Preallocate the output container — we know exactly how many
        // windows fit in the buffer.
        let n_windows = (audio.samples.len() - window_samples) / hop_samples + 1;
        let mut embeddings = Vec::with_capacity(n_windows);

        let mut start = 0usize;
        while start + window_samples <= audio.samples.len() {
            let window = &audio.samples[start..start + window_samples];
            // Pre-size the per-embedding Vec to embedding_dim so it doesn't
            // grow during the L2-norm + extend in embed_window_into.
            let mut vector = Vec::with_capacity(embedding_dim);
            self.core.embed_window_into(window, &mut vector)?;
            let t_start = TimestampMs((start as u64) * 1000 / sr);
            embeddings.push(NeuralEmbedding { vector, t_start });
            start += hop_samples;
        }

        Ok(NeuralFingerprint {
            embeddings,
            embedding_dim,
            frames_per_sec: 1.0 / self.core.cfg.hop_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn unique_path(stem: &str) -> std::path::PathBuf {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("audiofp_neural_{stem}_{pid}_{nanos}.onnx"))
    }

    fn assert_config_err<F: FnOnce(&str)>(cfg: NeuralEmbedderConfig, check: F) {
        match NeuralEmbedder::new(cfg) {
            Err(AfpError::Config(msg)) => check(&msg),
            Err(e) => panic!("expected Config error, got {e:?}"),
            Ok(_) => panic!("expected Config error, got Ok"),
        }
    }

    #[test]
    fn empty_model_path_returns_model_not_found() {
        match NeuralEmbedder::new(NeuralEmbedderConfig::new("")) {
            Err(AfpError::ModelNotFound(p)) => assert!(p.is_empty()),
            Err(e) => panic!("expected ModelNotFound(\"\"), got {e:?}"),
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
        }
    }

    #[test]
    fn missing_model_returns_model_not_found() {
        let path = unique_path("missing");
        assert!(!path.exists());
        let cfg = NeuralEmbedderConfig::new(path.to_string_lossy().to_string());
        match NeuralEmbedder::new(cfg) {
            Err(AfpError::ModelNotFound(p)) => assert_eq!(p, path.to_string_lossy()),
            Err(e) => panic!("expected ModelNotFound, got {e:?}"),
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
        }
    }

    #[test]
    fn corrupt_onnx_returns_model_load_error() {
        let path = unique_path("corrupt");
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(b"not a valid onnx protobuf")
                .expect("write temp file");
        }
        let cfg = NeuralEmbedderConfig::new(path.to_string_lossy().to_string());
        let res = NeuralEmbedder::new(cfg);
        let _ = std::fs::remove_file(&path);
        match res {
            Err(AfpError::ModelLoad(_)) => {}
            Err(e) => panic!("expected ModelLoad, got {e:?}"),
            Ok(_) => panic!("expected ModelLoad, got Ok"),
        }
    }

    #[test]
    fn zero_sample_rate_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.sample_rate = 0;
        assert_config_err(cfg, |_| {});
    }

    #[test]
    fn non_power_of_two_n_fft_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.n_fft = 1000;
        assert_config_err(cfg, |msg| assert!(msg.contains("n_fft")));
    }

    #[test]
    fn hop_larger_than_n_fft_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.hop = 4096; // > default n_fft 1024
        assert_config_err(cfg, |msg| assert!(msg.contains("hop")));
    }

    #[test]
    fn fmax_above_nyquist_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.fmax = cfg.sample_rate as f32; // > sr/2
        assert_config_err(cfg, |msg| {
            assert!(msg.contains("fmax") || msg.contains("fmin"))
        });
    }

    #[test]
    fn window_shorter_than_n_fft_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        // 1024 samples / 16000 Hz = 0.064 s; ask for 0.01 s windows.
        cfg.window_secs = 0.01;
        assert_config_err(cfg, |msg| {
            assert!(msg.contains("window_samples") && msg.contains("n_fft"))
        });
    }

    #[test]
    fn hop_larger_than_window_is_rejected() {
        // Sparse-hop sampling would underflow the streaming carry —
        // reject at construction so neither offline nor streaming can
        // be invoked with this config.
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.window_secs = 0.5;
        cfg.hop_secs = 1.0;
        assert_config_err(cfg, |msg| {
            assert!(
                msg.contains("hop_samples") && msg.contains("window_samples"),
                "expected hop>window message, got: {msg}",
            )
        });
    }

    #[test]
    fn config_constructor_uses_documented_defaults() {
        let cfg = NeuralEmbedderConfig::new("model.onnx");
        assert_eq!(cfg.sample_rate, 16_000);
        assert_eq!(cfg.n_fft, 1024);
        assert_eq!(cfg.hop, 320);
        assert_eq!(cfg.n_mels, 128);
        assert_eq!(cfg.fmin, 0.0);
        assert_eq!(cfg.fmax, 8_000.0);
        assert_eq!(cfg.mel_scale, MelScale::Slaney);
        assert_eq!(cfg.window_kind, WindowKind::Hann);
        assert_eq!(cfg.window_secs, 1.0);
        assert_eq!(cfg.hop_secs, 1.0);
        assert!(cfg.l2_normalize);
    }
}
