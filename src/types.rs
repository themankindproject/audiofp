//! Shared value types used across the `afp` crate.

use core::num::NonZeroU32;

/// A sample rate in hertz, guaranteed non-zero.
///
/// Use one of the `HZ_*` constants for the rates `afp` supports out of the
/// box, or [`SampleRate::new`] to validate an arbitrary value.
///
/// # Example
///
/// ```
/// use afp::SampleRate;
///
/// assert_eq!(SampleRate::HZ_44100.hz(), 44_100);
/// assert!(SampleRate::new(0).is_none());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SampleRate(pub NonZeroU32);

impl SampleRate {
    /// 8 kHz — the rate `afp`'s classical fingerprinters consume.
    pub const HZ_8000: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(8_000)) };

    /// 11.025 kHz.
    pub const HZ_11025: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(11_025)) };

    /// 16 kHz — typical speech rate.
    pub const HZ_16000: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(16_000)) };

    /// 22.05 kHz — common for music workflows.
    pub const HZ_22050: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(22_050)) };

    /// 44.1 kHz — CD-quality audio.
    pub const HZ_44100: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(44_100)) };

    /// 48 kHz — DAT / professional audio.
    pub const HZ_48000: SampleRate =
        unsafe { SampleRate(NonZeroU32::new_unchecked(48_000)) };

    /// Build a [`SampleRate`] from any non-zero `u32`.
    ///
    /// Returns `None` if `hz == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use afp::SampleRate;
    ///
    /// assert_eq!(SampleRate::new(32_000).unwrap().hz(), 32_000);
    /// assert!(SampleRate::new(0).is_none());
    /// ```
    #[must_use]
    pub const fn new(hz: u32) -> Option<SampleRate> {
        match NonZeroU32::new(hz) {
            Some(n) => Some(SampleRate(n)),
            None => None,
        }
    }

    /// Return the rate in hertz.
    ///
    /// # Example
    ///
    /// ```
    /// use afp::SampleRate;
    ///
    /// assert_eq!(SampleRate::HZ_48000.hz(), 48_000);
    /// ```
    #[must_use]
    pub const fn hz(self) -> u32 {
        self.0.get()
    }
}

/// A borrowed view of a mono PCM buffer in `[-1.0, 1.0]`.
///
/// Channel mixing is the caller's job — every public `afp` API takes mono
/// `f32`. Multi-channel inputs must be downmixed (helpers will live in the
/// streaming module once it lands).
///
/// # Example
///
/// ```
/// use afp::{AudioBuffer, SampleRate};
///
/// let samples = vec![0.0_f32; 16_000];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_16000 };
/// assert_eq!(buf.samples.len(), 16_000);
/// assert_eq!(buf.rate.hz(), 16_000);
/// ```
#[derive(Clone, Debug)]
pub struct AudioBuffer<'a> {
    /// Mono samples in `[-1.0, 1.0]`. Out-of-range values are not rejected
    /// here; downstream code clips or normalises as needed.
    pub samples: &'a [f32],

    /// Sample rate the samples were captured at.
    pub rate: SampleRate,
}

/// A timestamp in milliseconds since the start of a stream.
///
/// `u64` gives roughly 584 million years of headroom — long enough.
///
/// # Example
///
/// ```
/// use afp::TimestampMs;
///
/// let t = TimestampMs(1_500);
/// assert_eq!(t.0, 1_500);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimestampMs(pub u64);
