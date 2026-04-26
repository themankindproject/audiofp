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
}
