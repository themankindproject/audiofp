//! Fingerprinter traits.
//!
//! Two traits cover the two ways `audiofp` produces fingerprints:
//!
//! - [`Fingerprinter`] — feed a whole [`AudioBuffer`] and get its full
//!   output. Suited to enrolment / batch jobs.
//! - [`StreamingFingerprinter`] — push samples as they arrive and receive
//!   fingerprints whenever the algorithm has enough material. Suited to
//!   live capture.
//!
//! Concrete implementations live in feature-gated modules
//! (`fp_classical::Wang`, `neural::ResonaFp`, …).
//!
//! [`AudioBuffer`]: crate::AudioBuffer

use alloc::vec::Vec;

use crate::{AudioBuffer, Result, TimestampMs};

/// Offline (whole-buffer) fingerprinter.
///
/// Implementations are stateful between calls only insofar as they may
/// cache scratch buffers — the fingerprint of `extract(a)` does not depend
/// on any previous call.
///
/// # Example
///
/// ```
/// use audiofp::{AudioBuffer, Fingerprinter, Result, SampleRate};
///
/// /// A toy fingerprinter that just sums absolute samples.
/// struct Energy;
///
/// impl Fingerprinter for Energy {
///     type Output = f32;
///     type Config = ();
///
///     fn name(&self) -> &'static str { "energy-v0" }
///     fn config(&self) -> &Self::Config { &() }
///     fn required_sample_rate(&self) -> u32 { 16_000 }
///     fn min_samples(&self) -> usize { 16_000 }
///     fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
///         Ok(audio.samples.iter().map(|s| s.abs()).sum())
///     }
/// }
///
/// let mut fp = Energy;
/// let samples = vec![0.0_f32; 16_000];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_16000 };
/// assert_eq!(fp.extract(buf).unwrap(), 0.0);
/// ```
pub trait Fingerprinter {
    /// The fingerprint produced by this extractor (e.g. `Vec<WangHash>`).
    type Output;

    /// Per-instance configuration this fingerprinter exposes to callers.
    type Config: Clone + Send + Sync;

    /// Stable identifier for the algorithm and version, e.g. `"wang-v1"`.
    /// Useful when persisting fingerprints alongside the producer name.
    ///
    /// **Versioning contract:** the returned string is guaranteed to be
    /// stable as long as the bytes of the produced hashes are stable.
    /// A change that alters hash bytes (algorithm tweak, parameter bump,
    /// representation change) **must** bump the version suffix
    /// (`wang-v1` → `wang-v2`, etc.) in the same release. Persisted
    /// fingerprints can then be invalidated, migrated, or rejected
    /// based on the producer name without ambiguity.
    fn name(&self) -> &'static str;

    /// Borrow the configuration this instance was built with.
    fn config(&self) -> &Self::Config;

    /// Sample rate, in hertz, the fingerprinter expects its input at.
    /// Resampling is the caller's responsibility.
    fn required_sample_rate(&self) -> u32;

    /// Minimum buffer length, in samples, required to extract anything.
    /// Calls with shorter inputs return [`AfpError::AudioTooShort`].
    ///
    /// [`AfpError::AudioTooShort`]: crate::AfpError::AudioTooShort
    fn min_samples(&self) -> usize;

    /// Compute the fingerprint of `audio`.
    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output>;
}

/// Streaming fingerprinter that emits zero-or-more frames per push.
///
/// Implementations must be **non-blocking** and **bounded-allocation**:
/// any buffers needed for sustained operation are allocated at construction,
/// not inside [`StreamingFingerprinter::push`]. This makes them suitable
/// for invocation from realtime audio callbacks (when invoked through
/// `audiofp`'s streaming orchestrator).
///
/// # Example
///
/// ```
/// use audiofp::{StreamingFingerprinter, TimestampMs};
///
/// struct EveryThird { count: usize }
///
/// impl StreamingFingerprinter for EveryThird {
///     type Frame = u32;
///     fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, u32)> {
///         let mut out = Vec::new();
///         for s in samples {
///             self.count += 1;
///             if self.count % 3 == 0 {
///                 out.push((TimestampMs(self.count as u64), s.to_bits()));
///             }
///         }
///         out
///     }
///     fn flush(&mut self) -> Vec<(TimestampMs, u32)> { Vec::new() }
///     fn latency_ms(&self) -> u32 { 0 }
/// }
///
/// let mut fp = EveryThird { count: 0 };
/// assert_eq!(fp.push(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).len(), 2);
/// ```
pub trait StreamingFingerprinter {
    /// One unit of fingerprint material the stream emits.
    type Frame;

    /// Feed PCM samples and return any fingerprints that became available
    /// during this push.
    ///
    /// Must not block and must not allocate beyond the per-instance
    /// budget set at construction.
    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)>;

    /// Drain any pending fingerprint material at end-of-stream.
    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)>;

    /// Conservative upper bound on emission latency: from the time a
    /// sample enters [`push`] to the time the fingerprint covering it
    /// is returned.
    ///
    /// [`push`]: StreamingFingerprinter::push
    fn latency_ms(&self) -> u32;

    /// Feed PCM samples and invoke `callback` for each fingerprint frame
    /// that became available during this push.
    ///
    /// Zero-allocation variant: the callback receives `(TimestampMs, &Frame)`
    /// by reference, avoiding the `Vec` allocation of [`push`]. Returns the
    /// number of frames emitted.
    ///
    /// Default implementation delegates to [`push`] and iterates the result.
    /// Implementors should override for true zero-allocation behaviour.
    ///
    /// [`push`]: StreamingFingerprinter::push
    fn push_with<F>(&mut self, samples: &[f32], mut callback: F) -> usize
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        let frames = self.push(samples);
        let n = frames.len();
        for (t, frame) in frames {
            callback(t, &frame);
        }
        n
    }

    /// Drain any pending fingerprint material at end-of-stream, invoking
    /// `callback` for each frame.
    ///
    /// Zero-allocation variant of [`flush`]. Returns the number of frames
    /// emitted.
    ///
    /// Default implementation delegates to [`flush`] and iterates the result.
    ///
    /// [`flush`]: StreamingFingerprinter::flush
    fn flush_with<F>(&mut self, mut callback: F) -> usize
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        let frames = self.flush();
        let n = frames.len();
        for (t, frame) in frames {
            callback(t, &frame);
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy streaming fingerprinter that emits one `u32` frame per
    /// `push` sample. Its `push_with` / `flush_with` are the default
    /// trait methods — these tests pin the default-impl delegation
    /// contract that downstream implementations rely on.
    struct CountByThree {
        count: u32,
        buffered: Vec<u32>,
    }

    impl CountByThree {
        fn new() -> Self {
            Self {
                count: 0,
                buffered: Vec::new(),
            }
        }
    }

    impl StreamingFingerprinter for CountByThree {
        type Frame = u32;

        fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, u32)> {
            let mut out = Vec::new();
            for _ in samples {
                self.count += 1;
                if self.count % 3 == 0 {
                    out.push((TimestampMs(self.count as u64), self.count));
                }
            }
            self.buffered.extend(out.iter().map(|(_, f)| *f));
            out
        }

        fn flush(&mut self) -> Vec<(TimestampMs, u32)> {
            // Pretend there are 2 frames left buffered.
            let pending: Vec<u32> = self.buffered.drain(..).collect();
            pending
                .into_iter()
                .map(|v| (TimestampMs(v as u64 + 100), v))
                .collect()
        }

        fn latency_ms(&self) -> u32 {
            0
        }
    }

    #[test]
    fn push_with_default_impl_matches_push() {
        let samples = vec![0.0_f32; 10];

        // Collect push() output.
        let mut fp = CountByThree::new();
        let mut a: Vec<(TimestampMs, u32)> = Vec::new();
        a.extend(fp.push(&samples));
        a.extend(fp.push(&[]));

        // Collect push_with() output (default impl delegates to push).
        let mut fp = CountByThree::new();
        let mut b: Vec<(TimestampMs, u32)> = Vec::new();
        fp.push_with(&samples, |t, f| b.push((t, *f)));
        fp.push_with(&[], |t, f| b.push((t, *f)));

        assert_eq!(
            a.len(),
            b.len(),
            "push_with must call back in the same order as push yields"
        );
        assert_eq!(a, b, "push_with must mirror push output exactly");
    }

    #[test]
    fn flush_with_default_impl_matches_flush() {
        let mut fp = CountByThree::new();
        let samples = vec![0.0_f32; 9];
        let _ = fp.push(&samples);
        // 3 frames emitted (count=3,6,9), 0 buffered (the toy
        // implementation also keeps its own copy in `buffered`).
        let pending: Vec<_> = fp.flush();

        let mut fp = CountByThree::new();
        let _ = fp.push(&samples);
        let mut collected = Vec::new();
        let n = fp.flush_with(|t, f| collected.push((t, *f)));

        assert_eq!(n, pending.len());
        assert_eq!(collected, pending, "flush_with must mirror flush");
    }

    #[test]
    fn push_with_reports_emitted_count() {
        let mut fp = CountByThree::new();
        // 9 samples → 3 emitted (count=3,6,9).
        let n = fp.push_with(&[0.0_f32; 9], |_, _| {});
        assert_eq!(n, 3);
        // 2 more samples → 0 emitted (count was 9, not divisible by 3).
        let n = fp.push_with(&[0.0_f32; 2], |_, _| {});
        assert_eq!(n, 0);
    }
}
