//! Tapered analysis windows for STFT.
//!
//! The three classics (Hann, Hamming, Blackman) are emitted in their
//! **periodic** form (period `N`, not `N-1`) — that's what librosa /
//! `scipy.signal.get_window(..., fftbins=True)` use for spectral analysis,
//! and matching that convention is necessary for librosa-parity tests.

use alloc::vec::Vec;

use libm::cosf;

/// Selects which tapered window to apply to each STFT frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WindowKind {
    /// `0.5 - 0.5 cos(2π n / N)`. Smooth, low spectral leakage; the default
    /// for almost every audio fingerprinting algorithm.
    Hann,

    /// `0.54 - 0.46 cos(2π n / N)`. Slightly narrower main lobe than Hann
    /// at the cost of higher sidelobes.
    Hamming,

    /// `0.42 - 0.5 cos(2π n / N) + 0.08 cos(4π n / N)`. Strongest sidelobe
    /// suppression of the three; widest main lobe.
    Blackman,
}

/// Compute a window of length `n` of the given kind.
///
/// Returns an empty `Vec` when `n == 0` and a single-sample `[1.0]` when
/// `n == 1`. Otherwise the result has length `n` and is non-negative.
///
/// # Example
///
/// ```
/// use audiofp::dsp::windows::{make_window, WindowKind};
///
/// let w = make_window(WindowKind::Hann, 1024);
/// assert_eq!(w.len(), 1024);
/// // Periodic Hann is zero at index 0.
/// assert!(w[0].abs() < 1e-6);
/// // Peaks near the centre.
/// assert!(w[512] > 0.99);
/// ```
#[must_use]
pub fn make_window(kind: WindowKind, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return alloc::vec![1.0];
    }

    let mut w = Vec::with_capacity(n);
    let n_f = n as f32;
    let two_pi = 2.0 * core::f32::consts::PI;
    let four_pi = 4.0 * core::f32::consts::PI;

    for k in 0..n {
        let kf = k as f32;
        let v = match kind {
            WindowKind::Hann => 0.5 - 0.5 * cosf(two_pi * kf / n_f),
            WindowKind::Hamming => 0.54 - 0.46 * cosf(two_pi * kf / n_f),
            WindowKind::Blackman => {
                0.42 - 0.5 * cosf(two_pi * kf / n_f) + 0.08 * cosf(four_pi * kf / n_f)
            }
        };
        w.push(v);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_starts_at_zero_and_peaks_in_middle() {
        let w = make_window(WindowKind::Hann, 1024);
        assert!(w[0].abs() < 1e-6);
        assert!(w[512] > 0.99);
        // Symmetric around the midpoint (periodic windows still have the
        // mirrored shape; index 0 is paired with index N which wraps around).
        for k in 1..512 {
            let mirror = 1024 - k;
            assert!((w[k] - w[mirror]).abs() < 1e-5, "k={k}");
        }
    }

    #[test]
    fn hamming_min_is_above_zero() {
        let w = make_window(WindowKind::Hamming, 512);
        // Hamming bottoms out at 0.08, never zero.
        assert!(w[0] >= 0.07);
    }

    #[test]
    fn blackman_endpoints_are_close_to_zero() {
        let w = make_window(WindowKind::Blackman, 512);
        // Periodic Blackman at n=0 evaluates to 0.42 - 0.5 + 0.08 = 0.0.
        assert!(w[0].abs() < 1e-5);
    }

    #[test]
    fn empty_and_single_element_edges() {
        assert!(make_window(WindowKind::Hann, 0).is_empty());
        assert_eq!(make_window(WindowKind::Hann, 1), alloc::vec![1.0]);
    }

    #[test]
    fn periodic_windows_have_zero_at_endpoint_only_for_hann_blackman() {
        // Hann and Blackman bottom out at zero at index 0 (periodic form).
        // Hamming has a non-zero floor (0.08).
        let h = make_window(WindowKind::Hann, 256);
        let b = make_window(WindowKind::Blackman, 256);
        assert!(h[0].abs() < 1e-5);
        assert!(b[0].abs() < 1e-5);
        let hm = make_window(WindowKind::Hamming, 256);
        assert!(hm[0] > 0.05, "Hamming floor too low: {}", hm[0]);
    }

    #[test]
    fn window_lengths_match_request() {
        for kind in [WindowKind::Hann, WindowKind::Hamming, WindowKind::Blackman] {
            for n in [2_usize, 4, 16, 100, 1024, 4096] {
                let w = make_window(kind, n);
                assert_eq!(w.len(), n, "kind={kind:?} n={n}");
            }
        }
    }

    #[test]
    fn windows_are_non_negative() {
        for kind in [WindowKind::Hann, WindowKind::Hamming, WindowKind::Blackman] {
            let w = make_window(kind, 1024);
            for (i, &v) in w.iter().enumerate() {
                assert!(v >= -1e-6, "kind={kind:?} idx={i} v={v}");
            }
        }
    }
}
