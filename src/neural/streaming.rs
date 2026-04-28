//! Streaming variant of [`super::NeuralEmbedder`].

use alloc::vec::Vec;

use crate::{Result, StreamingFingerprinter, TimestampMs};

use super::embedder::{EmbedderCore, NeuralEmbedderConfig};

/// Generic ONNX log-mel audio embedder, streaming variant.
///
/// Buffers PCM until at least one full analysis window is available,
/// emits one embedding per [`NeuralEmbedderConfig::hop_secs`], and
/// drops consumed samples to keep memory bounded — buffer length is
/// always strictly less than `window_samples + chunk_size`.
///
/// Output is bit-exactly equivalent to
/// [`NeuralEmbedder::extract`](super::NeuralEmbedder) for the same
/// total input, regardless of how the input is chunked across calls.
/// Both paths share the same `embed_window_into` routine, all scratch
/// is reset per call, and non-centred framing means no STFT state
/// crosses window boundaries. Verified end-to-end by the
/// `streaming_matches_offline_on_full_buffer`,
/// `streaming_chunk_size_invariant`, and
/// `overlapping_window_streaming_matches_offline` tests using an
/// in-process passthrough tract fixture.
///
/// # Example
///
/// ```no_run
/// use audiofp::neural::{NeuralEmbedderConfig, StreamingNeuralEmbedder};
/// use audiofp::StreamingFingerprinter;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut s = StreamingNeuralEmbedder::new(NeuralEmbedderConfig::new("my_model.onnx"))?;
/// // Feed 16 kHz PCM in arbitrary-sized chunks.
/// let chunk = vec![0.0_f32; 4_096];
/// let frames = s.push(&chunk);
/// for (t, vector) in frames {
///     println!("t={} ms, dim={}", t.0, vector.len());
/// }
/// # Ok(()) }
/// ```
pub struct StreamingNeuralEmbedder {
    core: EmbedderCore,
    /// Ring buffer of unconsumed input samples. Bounded by
    /// `window_samples - 1 + max push size`.
    sample_carry: Vec<f32>,
    /// Total samples consumed (i.e. drained from `sample_carry`) since
    /// construction. Drives output timestamps so they match the offline
    /// embedder.
    samples_consumed: u64,
}

impl StreamingNeuralEmbedder {
    /// Build a streaming embedder with the given config.
    ///
    /// Performs the same work as [`super::NeuralEmbedder::new`] —
    /// validation, ONNX load, optimisation, runnable plan construction,
    /// and a probe inference — and shares the same error contract.
    pub fn new(cfg: NeuralEmbedderConfig) -> Result<Self> {
        let inner = super::embedder::NeuralEmbedder::new(cfg)?;
        Ok(Self {
            core: inner.core,
            sample_carry: Vec::new(),
            samples_consumed: 0,
        })
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &NeuralEmbedderConfig {
        &self.core.cfg
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

    /// Same contract as [`StreamingFingerprinter::push`] but propagates
    /// inference errors instead of panicking. Prefer this entry point
    /// when you need to surface model failures.
    pub fn try_push(&mut self, samples: &[f32]) -> Result<Vec<(TimestampMs, Vec<f32>)>> {
        let mut out = Vec::new();
        self.try_push_with(samples, |t, v| {
            out.push((t, v.to_vec()));
        })?;
        Ok(out)
    }

    /// Zero-allocation streaming variant: feed PCM and invoke
    /// `callback(t_start, &embedding)` for each embedding that becomes
    /// available. The embedding slice borrows the embedder's internal
    /// buffer and is overwritten on the next emit — copy out before
    /// the next iteration if you need to keep it.
    ///
    /// Allocates exactly one `Vec<f32>` for the embedding scratch on
    /// the first emit per call, reused across every emit in the same
    /// `try_push_with`. Returns the number of embeddings emitted.
    ///
    /// **On error**: if inference fails partway through a multi-window
    /// push, embeddings already passed to the callback have been
    /// committed (their samples are drained from the carry and
    /// `samples_consumed` advanced), but timestamps for any subsequent
    /// `push` will not realign to a hypothetical "fresh start". Call
    /// [`reset`] before reusing the embedder if you need a clean
    /// timeline after an error.
    ///
    /// [`reset`]: StreamingNeuralEmbedder::reset
    pub fn try_push_with<F: FnMut(TimestampMs, &[f32])>(
        &mut self,
        samples: &[f32],
        mut callback: F,
    ) -> Result<usize> {
        self.sample_carry.extend_from_slice(samples);
        let window_samples = self.core.window_samples;
        let hop_samples = self.core.hop_samples;
        let sr = self.core.cfg.sample_rate as u64;
        let mut buf = Vec::with_capacity(self.core.embedding_dim);
        let mut emitted = 0usize;

        while self.sample_carry.len() >= window_samples {
            // Disjoint borrow: `sample_carry` (immutable) and `core`
            // (mutable) are different fields of `self`, so this is sound
            // under Rust 2024's borrow rules.
            {
                let window = &self.sample_carry[..window_samples];
                self.core.embed_window_into(window, &mut buf)?;
            }
            let t_start = TimestampMs(self.samples_consumed * 1000 / sr);
            callback(t_start, &buf);
            emitted += 1;

            // Drop the hop_samples we've now committed to; the remainder
            // becomes the prefix of the next window. This keeps the
            // carry strictly bounded — never larger than
            // `window_samples + max_push - hop_samples`.
            self.sample_carry.drain(..hop_samples);
            self.samples_consumed += hop_samples as u64;
        }

        Ok(emitted)
    }

    /// Discard any unconsumed samples without emitting embeddings.
    ///
    /// Useful between independent input streams sharing one embedder
    /// instance — call before feeding the next stream so leftover
    /// samples from the previous one don't bleed into the first window.
    pub fn reset(&mut self) {
        self.sample_carry.clear();
        self.samples_consumed = 0;
    }

    /// Test-only: number of unconsumed samples currently buffered.
    /// Exposed so the carry-bounded invariant can be checked directly.
    #[cfg(test)]
    pub(crate) fn carry_len(&self) -> usize {
        self.sample_carry.len()
    }

    /// Test-only: total samples drained since construction (or last reset).
    #[cfg(test)]
    pub(crate) fn samples_consumed(&self) -> u64 {
        self.samples_consumed
    }

    /// Test-only constructor: wrap a pre-built `EmbedderCore`. Used by
    /// the in-process passthrough fixture in `test_support`.
    #[cfg(test)]
    pub(crate) fn __from_core_for_test(core: EmbedderCore) -> Self {
        Self {
            core,
            sample_carry: Vec::new(),
            samples_consumed: 0,
        }
    }
}

impl StreamingFingerprinter for StreamingNeuralEmbedder {
    type Frame = Vec<f32>;

    /// Feed PCM samples and return any embeddings that became available.
    ///
    /// # Panics
    ///
    /// Panics if ONNX inference fails. Use [`try_push`] if you need to
    /// recover from inference errors.
    ///
    /// [`try_push`]: StreamingNeuralEmbedder::try_push
    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)> {
        self.try_push(samples)
            .unwrap_or_else(|e| panic!("neural inference failed during push: {e}"))
    }

    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)> {
        // Non-centred framing means partial windows can't produce
        // embeddings — drop them.
        self.sample_carry.clear();
        Vec::new()
    }

    fn latency_ms(&self) -> u32 {
        // Worst-case latency between a sample entering `push` and the
        // embedding covering it being returned: the time to fill one
        // analysis window past that sample, capped at u32.
        let ms = (self.core.window_samples as u64 * 1000) / self.core.cfg.sample_rate.max(1) as u64;
        ms.min(u32::MAX as u64) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AfpError;
    use crate::Fingerprinter;
    use crate::neural::test_support::{passthrough_streaming, small_cfg, synth_audio};

    // ---------- Error path coverage (no model needed) ----------

    #[test]
    fn missing_model_propagates_through_streaming_constructor() {
        let cfg = NeuralEmbedderConfig::new("/definitely/does/not/exist.onnx");
        match StreamingNeuralEmbedder::new(cfg) {
            Err(AfpError::ModelNotFound(_)) => {}
            Err(e) => panic!("expected ModelNotFound, got {e:?}"),
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
        }
    }

    #[test]
    fn empty_path_propagates_through_streaming_constructor() {
        let cfg = NeuralEmbedderConfig::new("");
        match StreamingNeuralEmbedder::new(cfg) {
            Err(AfpError::ModelNotFound(p)) => assert!(p.is_empty()),
            Err(e) => panic!("expected ModelNotFound, got {e:?}"),
            Ok(_) => panic!("expected ModelNotFound, got Ok"),
        }
    }

    // ---------- Behaviour tests via passthrough fixture ----------

    fn fixture() -> StreamingNeuralEmbedder {
        passthrough_streaming(small_cfg()).expect("fixture builds")
    }

    #[test]
    fn embedding_dim_matches_passthrough_shape() {
        let s = fixture();
        // small_cfg: n_mels=8, n_frames=(4000-256)/128+1=30
        assert_eq!(s.embedding_dim(), 8 * 30);
    }

    #[test]
    fn latency_ms_matches_window_duration() {
        let s = fixture();
        // window_secs = 0.25 → 250 ms.
        assert_eq!(s.latency_ms(), 250);
    }

    #[test]
    fn empty_push_emits_nothing_and_does_not_buffer() {
        let mut s = fixture();
        let out = s.push(&[]);
        assert!(out.is_empty());
        assert_eq!(s.carry_len(), 0);
    }

    #[test]
    fn sub_window_push_only_buffers() {
        let mut s = fixture();
        let half = s.window_samples() / 2;
        let chunk = vec![0.0_f32; half];
        let out = s.push(&chunk);
        assert!(out.is_empty(), "no embedding before window full");
        assert_eq!(s.carry_len(), half);
        assert_eq!(s.samples_consumed(), 0);
    }

    #[test]
    fn one_full_window_emits_one_embedding() {
        let mut s = fixture();
        let chunk = synth_audio(1, s.window_samples(), 16_000);
        let out = s.push(&chunk);
        assert_eq!(out.len(), 1);
        let (t, vec) = &out[0];
        assert_eq!(t.0, 0);
        assert_eq!(vec.len(), s.embedding_dim());
        assert_eq!(s.carry_len(), 0); // hop == window in small_cfg → fully drained
    }

    #[test]
    fn timestamps_advance_by_hop_secs() {
        let mut s = fixture();
        // Push 4 windows worth of audio at once.
        let n = 4 * s.window_samples();
        let chunk = synth_audio(2, n, 16_000);
        let out = s.push(&chunk);
        assert_eq!(out.len(), 4);
        for (i, (t, _)) in out.iter().enumerate() {
            // small_cfg hop_secs = 0.25 → 250 ms apart.
            assert_eq!(t.0, (i as u64) * 250);
        }
    }

    #[test]
    fn carry_is_bounded_under_arbitrarily_large_pushes() {
        let mut s = fixture();
        // Push a lot of audio in one shot. Carry must end strictly less
        // than window_samples after the loop drains complete windows.
        let n = 100 * s.window_samples() + 17;
        let chunk = synth_audio(3, n, 16_000);
        let out = s.push(&chunk);
        assert_eq!(out.len(), 100);
        assert!(
            s.carry_len() < s.window_samples(),
            "carry {} >= window {}",
            s.carry_len(),
            s.window_samples(),
        );
        // The 17 leftover samples remain in the carry.
        assert_eq!(s.carry_len(), 17);
    }

    #[test]
    fn flush_clears_carry_and_emits_nothing() {
        let mut s = fixture();
        let chunk = vec![0.0_f32; s.window_samples() / 2];
        s.push(&chunk);
        assert!(s.carry_len() > 0);
        let f = s.flush();
        assert!(f.is_empty());
        assert_eq!(s.carry_len(), 0);
    }

    #[test]
    fn reset_clears_carry_and_consumed_count() {
        let mut s = fixture();
        let chunk = synth_audio(4, 3 * s.window_samples(), 16_000);
        let _ = s.push(&chunk);
        assert!(s.samples_consumed() > 0);
        s.reset();
        assert_eq!(s.carry_len(), 0);
        assert_eq!(s.samples_consumed(), 0);
    }

    #[test]
    fn try_push_returns_same_as_push_on_success() {
        let mut a = fixture();
        let mut b = fixture();
        let chunk = synth_audio(5, 2 * a.window_samples(), 16_000);

        let out_push = a.push(&chunk);
        let out_try = b.try_push(&chunk).expect("try_push ok");

        assert_eq!(out_push.len(), out_try.len());
        for ((t1, v1), (t2, v2)) in out_push.iter().zip(out_try.iter()) {
            assert_eq!(t1.0, t2.0);
            assert_eq!(v1, v2);
        }
    }

    // ---------- Offline / streaming bit-exactness ----------

    #[test]
    fn streaming_matches_offline_on_full_buffer() {
        let cfg = small_cfg();
        let n = 5 * (cfg.window_secs * cfg.sample_rate as f32) as usize;
        let audio = synth_audio(7, n, cfg.sample_rate);

        // Offline.
        let mut off = crate::neural::test_support::passthrough_embedder(cfg.clone()).unwrap();
        let buf = crate::AudioBuffer {
            samples: &audio,
            rate: crate::SampleRate::HZ_16000,
        };
        let off_fp = off.extract(buf).unwrap();

        // Streaming, single big push.
        let mut s = passthrough_streaming(cfg).unwrap();
        let s_out = s.push(&audio);

        assert_eq!(off_fp.embeddings.len(), s_out.len());
        for (e, (t, v)) in off_fp.embeddings.iter().zip(s_out.iter()) {
            assert_eq!(e.t_start.0, t.0);
            assert_eq!(&e.vector, v);
        }
    }

    #[test]
    fn streaming_chunk_size_invariant() {
        // Same audio, fed in many different chunk patterns, must
        // produce the same embedding sequence.
        let cfg = small_cfg();
        let n = 8 * (cfg.window_secs * cfg.sample_rate as f32) as usize + 23;
        let audio = synth_audio(11, n, cfg.sample_rate);

        let reference = {
            let mut s = passthrough_streaming(cfg.clone()).unwrap();
            s.push(&audio)
        };

        for chunk_size in [1, 7, 17, 256, 1024, 8_191] {
            let mut s = passthrough_streaming(cfg.clone()).unwrap();
            let mut collected = Vec::new();
            let mut start = 0;
            while start < audio.len() {
                let end = (start + chunk_size).min(audio.len());
                collected.extend(s.push(&audio[start..end]));
                start = end;
            }
            assert_eq!(
                collected.len(),
                reference.len(),
                "chunk_size={chunk_size}: count mismatch",
            );
            for ((t1, v1), (t2, v2)) in collected.iter().zip(reference.iter()) {
                assert_eq!(t1.0, t2.0, "chunk_size={chunk_size}: timestamp drift");
                assert_eq!(v1, v2, "chunk_size={chunk_size}: embedding drift");
            }
        }
    }

    #[test]
    fn overlapping_window_streaming_matches_offline() {
        // hop_secs < window_secs → overlapping windows. Streaming must
        // still match offline.
        let mut cfg = small_cfg();
        cfg.window_secs = 0.5; // 8000 samples
        cfg.hop_secs = 0.25; // 4000 samples → 2:1 overlap
        let n = 4 * (cfg.window_secs * cfg.sample_rate as f32) as usize;
        let audio = synth_audio(13, n, cfg.sample_rate);

        let mut off = crate::neural::test_support::passthrough_embedder(cfg.clone()).unwrap();
        let buf = crate::AudioBuffer {
            samples: &audio,
            rate: crate::SampleRate::HZ_16000,
        };
        let off_fp = off.extract(buf).unwrap();

        let mut s = passthrough_streaming(cfg).unwrap();
        let s_out = s.push(&audio);

        assert_eq!(off_fp.embeddings.len(), s_out.len());
        for (e, (t, v)) in off_fp.embeddings.iter().zip(s_out.iter()) {
            assert_eq!(e.t_start.0, t.0);
            assert_eq!(&e.vector, v);
        }
    }

    #[test]
    fn l2_normalization_actually_normalises() {
        let mut cfg = small_cfg();
        cfg.l2_normalize = true;
        let mut s = passthrough_streaming(cfg.clone()).unwrap();
        let chunk = synth_audio(17, s.window_samples(), 16_000);
        let out = s.push(&chunk);
        let v = &out[0].1;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "expected L2 norm ~1, got {norm}",);
    }

    #[test]
    fn try_push_with_callback_matches_try_push_collected() {
        let mut a = fixture();
        let mut b = fixture();
        let chunk = synth_audio(23, 3 * a.window_samples(), 16_000);

        let collected_via_vec = a.try_push(&chunk).unwrap();

        let mut collected_via_cb: Vec<(TimestampMs, Vec<f32>)> = Vec::new();
        let n = b
            .try_push_with(&chunk, |t, v| collected_via_cb.push((t, v.to_vec())))
            .unwrap();

        assert_eq!(n, collected_via_vec.len());
        assert_eq!(collected_via_cb.len(), collected_via_vec.len());
        for ((t1, v1), (t2, v2)) in collected_via_vec.iter().zip(collected_via_cb.iter()) {
            assert_eq!(t1.0, t2.0);
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn no_l2_normalization_preserves_magnitude() {
        let mut cfg = small_cfg();
        cfg.l2_normalize = false;
        let mut s = passthrough_streaming(cfg.clone()).unwrap();
        let chunk = synth_audio(19, s.window_samples(), 16_000);
        let out = s.push(&chunk);
        let v = &out[0].1;
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Log-mel of any non-trivial signal has |·| well away from 1.0.
        assert!(
            (norm - 1.0).abs() > 0.5,
            "unexpected near-unit norm without L2: {norm}",
        );
    }
}
