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
pub mod stft;
pub mod windows;

/// Conversion factor: `10·log10(x) = DB_LOG2_FACTOR·log2(x)`.
///
/// Used by the Wang and Panako front-ends to compute dB magnitude from
/// power spectra without a `log10` call (`log2` is faster on x86 via a
/// single hardware instruction).
pub(crate) const DB_LOG2_FACTOR: f32 = 10.0 / core::f32::consts::LOG2_10;
