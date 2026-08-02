//! Shared value types used across the `audiofp` crate.

use core::num::NonZeroU32;

/// A sample rate in hertz, guaranteed non-zero.
///
/// Use one of the `HZ_*` constants for the rates `audiofp` supports out of the
/// box, or [`SampleRate::new`] to validate an arbitrary value.
///
/// # Example
///
/// ```
/// use audiofp::SampleRate;
///
/// assert_eq!(SampleRate::HZ_44100.hz(), 44_100);
/// assert!(SampleRate::new(0).is_none());
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SampleRate(pub NonZeroU32);

impl SampleRate {
    /// 5 kHz — the rate [`Haitsma`](crate::classical::Haitsma) consumes.
    pub const HZ_5000: SampleRate = SampleRate(NonZeroU32::new(5_000).unwrap());

    /// 8 kHz — the rate [`Wang`](crate::classical::Wang) and
    /// [`Panako`](crate::classical::Panako) consume.
    pub const HZ_8000: SampleRate = SampleRate(NonZeroU32::new(8_000).unwrap());

    /// 11.025 kHz.
    pub const HZ_11025: SampleRate = SampleRate(NonZeroU32::new(11_025).unwrap());

    /// 16 kHz — typical speech rate; AudioSeal watermark default.
    pub const HZ_16000: SampleRate = SampleRate(NonZeroU32::new(16_000).unwrap());

    /// 22.05 kHz — common for music workflows.
    pub const HZ_22050: SampleRate = SampleRate(NonZeroU32::new(22_050).unwrap());

    /// 44.1 kHz — CD-quality audio.
    pub const HZ_44100: SampleRate = SampleRate(NonZeroU32::new(44_100).unwrap());

    /// 48 kHz — DAT / professional audio.
    pub const HZ_48000: SampleRate = SampleRate(NonZeroU32::new(48_000).unwrap());

    /// Build a [`SampleRate`] from any non-zero `u32`.
    ///
    /// Returns `None` if `hz == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use audiofp::SampleRate;
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
    /// use audiofp::SampleRate;
    ///
    /// assert_eq!(SampleRate::HZ_48000.hz(), 48_000);
    /// ```
    #[must_use]
    pub const fn hz(self) -> u32 {
        self.0.get()
    }
}

/// A timestamp in milliseconds since the start of a stream.
///
/// `u64` gives roughly 584 million years of headroom — long enough.
///
/// # Example
///
/// ```
/// use audiofp::TimestampMs;
///
/// let t = TimestampMs(1_500);
/// assert_eq!(t.0, 1_500);
/// ```
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimestampMs(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_constants_match_their_names() {
        assert_eq!(SampleRate::HZ_5000.hz(), 5_000);
        assert_eq!(SampleRate::HZ_8000.hz(), 8_000);
        assert_eq!(SampleRate::HZ_11025.hz(), 11_025);
        assert_eq!(SampleRate::HZ_16000.hz(), 16_000);
        assert_eq!(SampleRate::HZ_22050.hz(), 22_050);
        assert_eq!(SampleRate::HZ_44100.hz(), 44_100);
        assert_eq!(SampleRate::HZ_48000.hz(), 48_000);
    }

    #[test]
    fn sample_rate_eq() {
        let a = SampleRate::HZ_44100;
        let b = SampleRate::new(44_100).unwrap();
        let c = SampleRate::HZ_48000;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sample_rate_new_rejects_zero() {
        assert!(SampleRate::new(0).is_none());
        assert_eq!(SampleRate::new(1).unwrap().hz(), 1);
    }

    #[test]
    fn timestamp_ord() {
        let a = TimestampMs(100);
        let b = TimestampMs(200);
        assert!(a < b);
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Less);
        assert_eq!(b.cmp(&a), core::cmp::Ordering::Greater);
        assert_eq!(a.cmp(&a), core::cmp::Ordering::Equal);
    }
}
