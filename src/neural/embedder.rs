//! Generic ONNX log-mel embedder (offline / whole-buffer).
//!
//! This module uses limited `unsafe` for zero-copy Tract tensor
//! initialization — the only path in the crate that requires it.
#![allow(unsafe_code)]

use crate::SampleRate;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use std::path::Path;

use tract_onnx::prelude::*;

use crate::dsp::mel::{MelFilterBank, MelScale};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::error::{map_model_load_err, map_model_open_io};
use crate::{AfpError, Fingerprinter, Result, TimestampMs};

use super::frontend::LogMelFrontend;

/// In-place L2 normalisation: scales `v` so its Euclidean norm is 1.
/// Leaves the vector unchanged if its norm is below `1e-12` (effectively
/// zero).
#[inline]
fn l2_normalize_inplace(v: &mut [f32]) {
    let sumsq: f32 = v.iter().map(|x| x * x).sum();
    let norm = sumsq.sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

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
    /// Maximum input sample count accepted by [`extract`]. `None`
    /// disables the check. Default: `None` (unlimited — neural
    /// models may handle long audio). Set to limit memory.
    ///
    /// [`extract`]: NeuralEmbedder::extract
    pub max_input_samples: Option<usize>,
    /// Maximum samples accepted in a single streaming `push`. `None`
    /// disables (default). Excess samples are dropped (push is infallible).
    pub max_push_samples: Option<usize>,
    /// Number of analysis windows to batch into a single model call
    /// during offline [`extract`](NeuralEmbedder::extract). Default 1
    /// (one inference call per window — original behaviour).
    ///
    /// Higher values amortise the fixed per-call overhead of the ONNX
    /// runtime across multiple windows. The streaming path always uses
    /// single-window inference regardless of this setting.
    ///
    /// Must be ≥ 1.
    pub batch_size: usize,
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
            max_input_samples: None,
            max_push_samples: None,
            batch_size: 1,
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
pub(crate) type Runnable = Arc<TypedSimplePlan>;

/// Heavy state shared by [`NeuralEmbedder`] and
/// [`super::StreamingNeuralEmbedder`]. Both compose this rather than
/// inherit, so neither re-implements the front-end.
pub(crate) struct EmbedderCore {
    pub(crate) cfg: NeuralEmbedderConfig,
    pub(crate) frontend: LogMelFrontend,
    pub(crate) runnable: Runnable,

    /// Batched runnable for offline inference when `batch_size > 1`.
    /// Accepts input shape `[batch_size, n_mels, n_frames]`. `None`
    /// when `batch_size == 1` (streaming also always uses `runnable`).
    pub(crate) batch_runnable: Option<Runnable>,

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

        // Allocate the model input tensor and write log-mel straight
        // into its `[1, n_mels, n_frames]` row-major buffer with strided
        // writes — no intermediate `Vec` and no transpose.
        //
        // NOTE: tract 0.22.1's `Tensor::clone()` is a deep copy
        // (`self.deep_clone()`), not a refcount bump, so caching the
        // tensor and cloning per call would not save the allocation.
        // A genuine zero-alloc path requires a tract API change
        // (`from_raw_vec` or similar); tracked in issue #7.
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
            // SAFETY: we just allocated `tensor` as f32 with the correct shape above.
            let dst = unsafe { tensor.as_slice_mut_unchecked::<f32>() };
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
            .to_plain_array_view::<f32>()
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
            l2_normalize_inplace(out);
        }

        Ok(())
    }

    /// Compute embeddings for a batch of windows in a single model call.
    ///
    /// `windows` is a slice of PCM slices, each of length
    /// `self.window_samples`. Results are appended to `out` in order.
    /// Requires `self.batch_runnable` to be `Some` and `windows.len()`
    /// to equal the batch size that runnable was built for.
    ///
    /// # Panics
    ///
    /// Panics if any window has the wrong length, if `batch_runnable` is
    /// `None`, or if `windows.len()` != configured batch size.
    pub(crate) fn embed_batch_into(
        &mut self,
        windows: &[&[f32]],
        out: &mut Vec<NeuralEmbedding>,
        timestamps: &[TimestampMs],
    ) -> Result<()> {
        let batch = windows.len();
        assert!(batch > 0, "embed_batch_into requires at least one window");
        let batch_runnable = self
            .batch_runnable
            .as_ref()
            .expect("embed_batch_into requires batch_runnable");

        let n_mels = self.frontend.n_mels();
        let n_frames = self.n_frames;
        let embedding_dim = self.embedding_dim;
        let window_stride = n_mels * n_frames; // elements per batch item

        // Allocate input tensor [batch, n_mels, n_frames].
        // SAFETY: we will fill every element before run() reads it.
        let mut tensor = unsafe {
            Tensor::uninitialized::<f32>(&[batch, n_mels, n_frames])
                .map_err(|e| AfpError::Inference(format!("batch input alloc: {e}")))?
        };

        {
            let dst = unsafe { tensor.as_slice_mut_unchecked::<f32>() };
            for (b, window) in windows.iter().enumerate() {
                assert_eq!(
                    window.len(),
                    self.window_samples,
                    "embed_batch_into: window {b} has wrong length"
                );
                let base = b * window_stride;
                self.frontend.for_each_frame(window, |f, mel_row| {
                    for m in 0..n_mels {
                        dst[base + m * n_frames + f] = mel_row[m];
                    }
                });
            }
        }

        let outputs = batch_runnable
            .run(tvec!(tensor.into()))
            .map_err(|e| AfpError::Inference(format!("batch run: {e}")))?;
        if outputs.is_empty() {
            return Err(AfpError::Inference(
                "batch model produced no outputs".to_string(),
            ));
        }

        let view = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("batch output view: {e}")))?;
        let expected_len = batch * embedding_dim;
        if view.len() != expected_len {
            return Err(AfpError::Inference(format!(
                "expected batch output of {} elements ({}×{}), got {}",
                expected_len,
                batch,
                embedding_dim,
                view.len(),
            )));
        }

        let mut scratch = Vec::with_capacity(embedding_dim);
        for (b, ts) in timestamps.iter().enumerate().take(batch) {
            let slice = &view
                .as_slice()
                .expect("output tensor must be contiguous f32")
                [b * embedding_dim..(b + 1) * embedding_dim];
            scratch.clear();
            scratch.extend_from_slice(slice);

            if self.cfg.l2_normalize {
                l2_normalize_inplace(&mut scratch);
            }

            out.push(NeuralEmbedding {
                vector: scratch.clone(),
                t_start: *ts,
            });
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
        if cfg.batch_size == 0 {
            return Err(AfpError::Config("batch_size must be >= 1".to_string()));
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
        // Open first so missing paths become `ModelNotFound` without an
        // `exists()` race; Tract re-opens by path for the actual parse.
        if let Err(e) = std::fs::File::open(path) {
            return Err(map_model_open_io(&cfg.model_path, e));
        }
        let model = tract_onnx::onnx()
            .model_for_path(path)
            .map_err(map_model_load_err)?;

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
            .to_plain_array_view::<f32>()
            .map_err(|e| AfpError::Inference(format!("probe view: {e}")))?;
        let embedding_dim = probe_view.len();
        if embedding_dim == 0 {
            return Err(AfpError::Inference(
                "model produced empty embedding on probe".to_string(),
            ));
        }

        // --- Build batch runnable (when batch_size > 1) ---------------
        let batch_runnable = if cfg.batch_size > 1 {
            let batch_model = tract_onnx::onnx()
                .model_for_path(path)
                .map_err(map_model_load_err)?;
            let plan: Runnable = batch_model
                .with_input_fact(
                    0,
                    InferenceFact::dt_shape(
                        f32::datum_type(),
                        tvec!(cfg.batch_size, cfg.n_mels, n_frames),
                    ),
                )
                .map_err(|e| AfpError::Inference(format!("batch input fact: {e}")))?
                .into_typed()
                .map_err(|e| AfpError::Inference(format!("batch type: {e}")))?
                .into_optimized()
                .map_err(|e| AfpError::Inference(format!("batch optimize: {e}")))?
                .into_runnable()
                .map_err(|e| AfpError::Inference(format!("batch runnable: {e}")))?;
            Some(plan)
        } else {
            None
        };

        let frontend = LogMelFrontend::new(stft, mel, window_samples);

        Ok(Self {
            core: EmbedderCore {
                cfg,
                frontend,
                runnable,
                batch_runnable,
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

    fn required_sample_rate(&self) -> SampleRate {
        // The neural config already validates `sample_rate != 0` at
        // construction, so the newtype conversion is infallible.
        SampleRate::new(self.core.cfg.sample_rate).expect("neural sample_rate is non-zero")
    }

    fn min_samples(&self) -> usize {
        self.core.window_samples
    }

    fn extract(&mut self, samples: &[f32], rate: SampleRate) -> Result<Self::Output> {
        crate::pcm::reject_non_finite(samples)?;
        if let Some(limit) = self.core.cfg.max_input_samples
            && samples.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: samples.len(),
            });
        }
        if rate.hz() != self.core.cfg.sample_rate {
            return Err(AfpError::UnsupportedSampleRate(rate.hz()));
        }
        if samples.len() < self.core.window_samples {
            return Err(AfpError::AudioTooShort {
                needed: self.core.window_samples,
                got: samples.len(),
            });
        }

        let sr = rate.hz() as u64;
        let window_samples = self.core.window_samples;
        let hop_samples = self.core.hop_samples;
        let embedding_dim = self.core.embedding_dim;
        let batch_size = self.core.cfg.batch_size;

        // Preallocate the output container — we know exactly how many
        // windows fit in the buffer.
        let n_windows = (samples.len() - window_samples) / hop_samples + 1;
        let mut embeddings = Vec::with_capacity(n_windows);

        if batch_size > 1 && self.core.batch_runnable.is_some() {
            // --- Batched inference path --------------------------------
            let mut start = 0usize;
            while start + window_samples <= samples.len() {
                let mut batch_windows: Vec<&[f32]> = Vec::with_capacity(batch_size);
                let mut batch_timestamps: Vec<TimestampMs> = Vec::with_capacity(batch_size);
                let mut s = start;
                while batch_windows.len() < batch_size && s + window_samples <= samples.len() {
                    batch_windows.push(&samples[s..s + window_samples]);
                    batch_timestamps.push(TimestampMs((s as u64) * 1000 / sr));
                    s += hop_samples;
                }

                if batch_windows.len() == batch_size {
                    // Full batch — single model call.
                    self.core.embed_batch_into(
                        &batch_windows,
                        &mut embeddings,
                        &batch_timestamps,
                    )?;
                } else {
                    // Partial final batch — fall back to single-window inference.
                    for (window, ts) in batch_windows.iter().zip(batch_timestamps.iter()) {
                        let mut vector = Vec::with_capacity(embedding_dim);
                        self.core.embed_window_into(window, &mut vector)?;
                        embeddings.push(NeuralEmbedding {
                            vector,
                            t_start: *ts,
                        });
                    }
                }

                start = s;
            }
        } else {
            // --- Single-window inference path (original behaviour) ------
            // One reused scratch vector: `embed_window_into` writes into
            // it, then we clone into the owned embedding. Avoids a fresh
            // `Vec::with_capacity` per window (N windows → 1 alloc vs N).
            let mut vector = Vec::with_capacity(embedding_dim);
            let mut start = 0usize;
            while start + window_samples <= samples.len() {
                let window = &samples[start..start + window_samples];
                vector.clear();
                self.core.embed_window_into(window, &mut vector)?;
                let t_start = TimestampMs((start as u64) * 1000 / sr);
                embeddings.push(NeuralEmbedding {
                    vector: vector.clone(),
                    t_start,
                });
                start += hop_samples;
            }
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
        assert_eq!(cfg.batch_size, 1);
    }

    // Public API contract pin.

    #[test]
    fn public_api_name_matches_documented_value() {
        // Use the in-process passthrough fixture so we don't need a
        // real ONNX file (no fixture ships with the crate).
        let cfg = NeuralEmbedderConfig::new("test-fixture");
        let fp = crate::neural::test_support::passthrough_embedder(cfg).unwrap();
        assert_eq!(fp.name(), "neural-onnx-v0");
        assert_eq!(fp.required_sample_rate(), crate::SampleRate::HZ_16000);
        // 1.0 s window at 16 kHz = 16 000 samples.
        assert_eq!(fp.min_samples(), 16_000);
    }

    // Happy-path coverage of the offline `extract` method.
    // Uses the in-process passthrough fixture to exercise the full pipeline
    // (front-end → strided tensor write → runnable → L2 normalisation)
    // without needing a real ONNX model file.

    #[test]
    fn offline_extract_produces_normalised_embeddings_of_expected_shape() {
        use crate::SampleRate;
        use crate::neural::test_support::{passthrough_embedder, small_cfg, synth_audio};

        let cfg = small_cfg();
        let mut fp = passthrough_embedder(cfg.clone()).expect("passthrough embedder");

        // 2 s of synthetic audio — 8 windows of cfg.window_secs (= 0.25 s)
        // each. Each window emits one embedding of length `n_mels * n_frames`.
        let samples = synth_audio(42, 2 * cfg.sample_rate as usize, cfg.sample_rate);

        let fp_out = fp
            .extract(&samples, SampleRate::new(cfg.sample_rate).unwrap())
            .expect("extract");
        let expected_dim = fp.embedding_dim();
        let window_samples = fp.window_samples();
        let hop_samples = fp.hop_samples();
        let n_windows = (samples.len().saturating_sub(window_samples)) / hop_samples + 1;

        assert_eq!(fp_out.embedding_dim, expected_dim);
        assert_eq!(fp_out.embeddings.len(), n_windows);
        for (i, e) in fp_out.embeddings.iter().enumerate() {
            assert_eq!(e.vector.len(), expected_dim, "window {i}");
            // L2-normalised when l2_normalize is on (small_cfg's default).
            let sumsq: f32 = e.vector.iter().map(|x| x * x).sum();
            assert!(
                (sumsq - 1.0).abs() < 1e-3,
                "window {i} not L2-normalised: sumsq={sumsq}"
            );
            // t_start is the window's start in ms, growing by hop.
            let expected_t = (i * hop_samples * 1000) as u64 / cfg.sample_rate as u64;
            assert_eq!(e.t_start.0, expected_t, "window {i} t_start");
        }
    }

    // Batched inference tests.

    #[test]
    fn zero_batch_size_is_rejected() {
        let mut cfg = NeuralEmbedderConfig::new("any.onnx");
        cfg.batch_size = 0;
        assert_config_err(cfg, |msg| assert!(msg.contains("batch_size")));
    }

    #[test]
    fn batched_extract_matches_single_window_extract() {
        use crate::SampleRate;
        use crate::neural::test_support::{passthrough_embedder, small_cfg, synth_audio};

        // Baseline: single-window (batch_size=1).
        let cfg_single = small_cfg();
        let mut fp_single = passthrough_embedder(cfg_single.clone()).expect("single embedder");

        // 2.5 s audio → 10 windows (at 0.25 s each).
        // batch_size=4 gives 2 full batches + 2 remainder.
        let n_samples = (2.5 * cfg_single.sample_rate as f32) as usize;
        let samples = synth_audio(7, n_samples, cfg_single.sample_rate);
        let out_single = fp_single
            .extract(&samples, SampleRate::new(cfg_single.sample_rate).unwrap())
            .expect("extract single");

        // Batched: batch_size=4.
        let mut cfg_batch = small_cfg();
        cfg_batch.batch_size = 4;
        let mut fp_batch = passthrough_embedder(cfg_batch).expect("batch embedder");
        let out_batch = fp_batch
            .extract(&samples, SampleRate::new(cfg_single.sample_rate).unwrap())
            .expect("extract batch");

        // Same number of embeddings.
        assert_eq!(
            out_batch.embeddings.len(),
            out_single.embeddings.len(),
            "embedding count mismatch: batch={} vs single={}",
            out_batch.embeddings.len(),
            out_single.embeddings.len(),
        );

        // Each embedding must be bit-exact (same front-end, same model).
        for (i, (a, b)) in out_single
            .embeddings
            .iter()
            .zip(out_batch.embeddings.iter())
            .enumerate()
        {
            assert_eq!(a.t_start, b.t_start, "window {i} timestamp mismatch");
            assert_eq!(a.vector.len(), b.vector.len(), "window {i} dim mismatch");
            for (j, (va, vb)) in a.vector.iter().zip(b.vector.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < 1e-6,
                    "window {i} dim {j} differs: single={va} batch={vb}",
                );
            }
        }
    }

    #[test]
    fn batched_extract_works_with_exact_multiple_of_batch_size() {
        use crate::SampleRate;
        use crate::neural::test_support::{passthrough_embedder, small_cfg, synth_audio};

        // Exactly 4 windows with batch_size=4 → one full batch, no remainder.
        let mut cfg = small_cfg();
        cfg.batch_size = 4;
        let mut fp = passthrough_embedder(cfg.clone()).expect("batch embedder");

        // 1.0 s = 4 windows of 0.25 s each (non-overlapping).
        let samples = synth_audio(99, cfg.sample_rate as usize, cfg.sample_rate);
        let out = fp
            .extract(&samples, SampleRate::new(cfg.sample_rate).unwrap())
            .expect("extract");
        assert_eq!(out.embeddings.len(), 4);
        for e in &out.embeddings {
            assert_eq!(e.vector.len(), out.embedding_dim);
        }
    }

    #[test]
    fn batched_extract_works_with_fewer_windows_than_batch_size() {
        use crate::SampleRate;
        use crate::neural::test_support::{passthrough_embedder, small_cfg, synth_audio};

        // batch_size=8 but only 2 windows worth of audio → all
        // handled by single-window fallback (partial batch).
        let mut cfg = small_cfg();
        cfg.batch_size = 8;
        let mut fp = passthrough_embedder(cfg.clone()).expect("batch embedder");

        // 0.5 s = 2 windows of 0.25 s.
        let n_samples = (cfg.sample_rate as f32 * 0.5) as usize;
        let samples = synth_audio(13, n_samples, cfg.sample_rate);
        let out = fp
            .extract(&samples, SampleRate::new(cfg.sample_rate).unwrap())
            .expect("extract");
        assert_eq!(out.embeddings.len(), 2);
    }
}
