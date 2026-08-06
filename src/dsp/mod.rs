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
    use wide::f32x8;

    let n = buf.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let floor_v = f32x8::splat(floor);
    let factor_v = f32x8::splat(DB_LOG2_FACTOR);

    for i in 0..chunks {
        let off = i * 8;
        let v = f32x8::new(
            buf[off..off + 8]
                .try_into()
                .expect("buf chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        let clamped = v.max(floor_v);
        let db = factor_v * clamped.log2();
        buf[off..off + 8].copy_from_slice(db.as_array());
    }

    // Scalar tail.
    for v in &mut buf[tail_start..] {
        *v = DB_LOG2_FACTOR * v.max(floor).log2();
    }
}

/// SIMD-accelerated dot product: `sum(a[i] * b[i])` via `wide::f32x8`.
///
/// Processes 8 elements at a time with fused multiply-add, then reduces.
/// Used by the polyphase resampler and mel filterbank hot paths.
#[inline]
pub(crate) fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let va = f32x8::new(
            a[off..off + 8]
                .try_into()
                .expect("a chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        let vb = f32x8::new(
            b[off..off + 8]
                .try_into()
                .expect("b chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        acc = va.mul_add(vb, acc);
    }

    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += a[i] * b[i];
    }
    sum
}

/// SIMD-accelerated squared dot product: `sum(a[i] * b[i]²)` via `wide::f32x8`.
///
/// Processes 8 elements at a time: squares `b`, then fused multiply-adds
/// with `a`. Used by the mel filterbank `log_mel` hot path to avoid a
/// separate power-spectrum allocation when starting from magnitudes.
#[inline]
pub(crate) fn dot_sq_wide(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let va = f32x8::new(
            a[off..off + 8]
                .try_into()
                .expect("a chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        let vb = f32x8::new(
            b[off..off + 8]
                .try_into()
                .expect("b chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        let vb_sq = vb * vb;
        acc = va.mul_add(vb_sq, acc);
    }

    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += a[i] * (b[i] * b[i]);
    }
    sum
}
