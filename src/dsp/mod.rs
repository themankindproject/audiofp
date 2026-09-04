//! Digital signal processing primitives.
//!
//! Each fingerprinter in [`crate::classical`] composes a fixed pipeline
//! out of these primitives, but they're public so users can build their
//! own analysis chains on top of `audiofp`.
//!
//! | Module                | Purpose                                                  |
//! | --------------------- | -------------------------------------------------------- |
//! | [`windows`]           | Tapered Hann / Hamming / Blackman generators             |
//! | [`stft`]              | Pre-planned real-input STFT with reusable scratch        |
//! | [`mel`]               | Triangular mel filterbank (HTK + Slaney scales)          |
//! | [`peaks`]             | 2-D peak picker (Lemire monotonic-deque rolling max)     |
//! | [`resample`]          | Linear and windowed-sinc Kaiser resamplers               |
//!
//! All modules compile under `no_std + alloc` so they can be reused on
//! hosted targets without `std`. Bare-metal embedded support is on the
//! roadmap (currently blocked by `rustfft` transitively requiring
//! `num-traits/std`).

pub mod mel;
pub mod peaks;
pub mod resample;
pub(crate) mod simd;
pub mod stft;
pub mod windows;

/// Conversion factor: `10·log10(x) = DB_LOG2_FACTOR·log2(x)`.
///
/// Used by the Wang and Panako front-ends to compute dB magnitude from
/// power spectra without a `log10` call (`log2` is faster on x86 via a
/// single hardware instruction).
pub(crate) const DB_LOG2_FACTOR: f32 = 10.0 / core::f32::consts::LOG2_10;

/// Convert a power spectrum slice to dB in-place using SIMD via `wide`.
///
/// Equivalent to:
/// ```ignore
/// for v in buf.iter_mut() {
///     *v = DB_LOG2_FACTOR * v.max(floor).log2();
/// }
/// ```
///
/// Uses `wide::f32x8` to process 8 elements at a time with vectorized
/// `max` and `log2`. Produces bit-identical results to the scalar path
/// because `wide::f32x8::log2()` implements the same IEEE-754 log2
/// computation as the scalar `f32::log2()`.
#[inline]
pub(crate) fn power_to_db_wide(buf: &mut [f32], floor: f32) {
    simd::db_into(buf, floor, DB_LOG2_FACTOR);
}

/// SIMD-accelerated dot product: `sum(a[i] * b[i])` via `wide::f32x8`.
///
/// Processes 8 elements at a time with fused multiply-add, then reduces.
/// Used by the polyphase resampler and mel filterbank hot paths.
#[inline]
pub(crate) fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    simd::dot_core(a, b)
}

/// SIMD-accelerated squared dot product: `sum(a[i] * b[i]²)` via `wide::f32x8`.
///
/// Processes 8 elements at a time: squares `b`, then fused multiply-adds
/// with `a`. Used by the mel filterbank `log_mel` hot path to avoid a
/// separate power-spectrum allocation when starting from magnitudes.
#[inline]
pub(crate) fn dot_sq_wide(a: &[f32], b: &[f32]) -> f32 {
    simd::dot_sq_core(a, b)
}
