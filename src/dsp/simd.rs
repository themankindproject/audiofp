//! Shared 8-wide SIMD helpers (`wide::f32x8`).
//!
//! Every DSP hot loop in the crate iterates `n/8` full chunks plus a
//! scalar tail. This module owns that skeleton once so there is a single
//! place to tune vector width, rounding behaviour, and fallback paths.
//!
//! All functions are `#[inline]` and `no_std + alloc` compatible; the
//! per-callsite wrappers in [`super`] (`dot_wide`, `power_to_db_wide`),
//! [`super::stft`], [`crate::neural::embedder`], and
//! [`crate::matching`] keep their names and delegate here, so behaviour
//! (including `wide`'s bit-identical `log2`/`sqrt`/`mul_add` semantics)
//! is unchanged.

use wide::f32x8;

/// Load 8 floats starting at `off`. The caller guarantees
/// `off + 8 <= s.len()` (all call sites iterate `n/8` complete chunks).
#[inline]
pub(crate) fn load8(s: &[f32], off: usize) -> f32x8 {
    f32x8::new(
        s[off..off + 8]
            .try_into()
            .expect("simd chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
    )
}

/// Store 8 floats starting at `off`.
#[inline]
pub(crate) fn store8(dst: &mut [f32], off: usize, v: f32x8) {
    dst[off..off + 8].copy_from_slice(v.as_array());
}

/// Core dot product: `sum(a[i] * b[i])` via fused multiply-add, 8-wide
/// with a scalar tail.
#[inline]
pub(crate) fn dot_core(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        acc = load8(a, off).mul_add(load8(b, off), acc);
    }

    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Core sum-of-squares: `sum(v[i] * v[i])`, 8-wide with a scalar tail.
///
/// Kept separate from [`dot_core`] (rather than `dot_core(v, v)`) so the
/// hot loop loads each chunk once instead of twice.
///
/// Used only by the `neural`-gated embedder/matcher; the allow keeps
/// non-neural builds warning-free while keeping the canonical sumsq here.
#[allow(dead_code)]
#[inline]
pub(crate) fn sumsq_core(v: &[f32]) -> f32 {
    let n = v.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let x = load8(v, off);
        acc = x.mul_add(x, acc);
    }

    let mut sumsq = acc.reduce_add();
    for &x in &v[tail_start..] {
        sumsq += x * x;
    }
    sumsq
}

/// Core squared dot product: `sum(a[i] * b[i]^2)`, 8-wide with a scalar
/// tail. Used by the mel filterbank `log_mel` path to avoid a separate
/// power-spectrum allocation when starting from magnitudes.
#[inline]
pub(crate) fn dot_sq_core(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let vb = load8(b, off);
        acc = load8(a, off).mul_add(vb * vb, acc);
    }

    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += a[i] * (b[i] * b[i]);
    }
    sum
}

/// Core elementwise multiply: `dst[i] = src[i] * win[i]`, 8-wide with a
/// scalar tail. Entirely safe code.
#[inline]
pub(crate) fn mul_into(src: &[f32], win: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), win.len());
    debug_assert_eq!(src.len(), dst.len());

    let n = src.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    for i in 0..chunks {
        let off = i * 8;
        store8(dst, off, load8(src, off) * load8(win, off));
    }

    for i in tail_start..n {
        dst[i] = src[i] * win[i];
    }
}

/// Core power-to-dB conversion: `buf[i] = factor * max(buf[i], floor).log2()`
/// in place, 8-wide with a scalar tail.
///
/// Produces bit-identical results to the scalar path because
/// `wide::f32x8::log2()` implements the same IEEE-754 log2 computation
/// as the scalar `f32::log2()`.
#[inline]
pub(crate) fn db_into(buf: &mut [f32], floor: f32, factor: f32) {
    let n = buf.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let floor_v = f32x8::splat(floor);
    let factor_v = f32x8::splat(factor);

    for i in 0..chunks {
        let off = i * 8;
        let clamped = load8(buf, off).max(floor_v);
        store8(buf, off, factor_v * clamped.log2());
    }

    // Scalar tail.
    for v in &mut buf[tail_start..] {
        *v = factor * v.max(floor).log2();
    }
}

/// Load the real and imaginary parts of 8 complex spectrum bins.
#[inline]
pub(crate) fn load_complex8(complex: &[num_complex::Complex<f32>], off: usize) -> (f32x8, f32x8) {
    let re = f32x8::new([
        complex[off].re,
        complex[off + 1].re,
        complex[off + 2].re,
        complex[off + 3].re,
        complex[off + 4].re,
        complex[off + 5].re,
        complex[off + 6].re,
        complex[off + 7].re,
    ]);
    let im = f32x8::new([
        complex[off].im,
        complex[off + 1].im,
        complex[off + 2].im,
        complex[off + 3].im,
        complex[off + 4].im,
        complex[off + 5].im,
        complex[off + 6].im,
        complex[off + 7].im,
    ]);
    (re, im)
}

/// Core complex power: `dst[i] = re^2 + im^2`, 8-wide with a scalar tail.
///
/// Vectorises the sqrt that the scalar path must take through
/// `libm::sqrtf`, which cannot be auto-vectorised. `f32x8::sqrt` is the
/// hardware IEEE sqrt (or the same musl-derived software sqrt in `wide`'s
/// no-SIMD fallback), so on default builds results are bit-identical to
/// the scalar loop. On FMA builds the `mul_add` fuses one rounding.
#[inline]
pub(crate) fn complex_power_into(complex: &[num_complex::Complex<f32>], dst: &mut [f32]) {
    complex_power_impl::<false>(complex, dst);
}

/// Core complex magnitude: `dst[i] = sqrt(re^2 + im^2)`, 8-wide with a
/// scalar tail. See [`complex_power_into`] for numerics notes.
#[inline]
pub(crate) fn complex_magnitude_into(complex: &[num_complex::Complex<f32>], dst: &mut [f32]) {
    complex_power_impl::<true>(complex, dst);
}

/// Shared implementation, specialized at compile time on `SQRT` so the
/// hot loop contains no branch (the pre-split code passed a runtime
/// `sqrt: bool`, which relies on the predictor / loop versioning).
#[inline]
fn complex_power_impl<const SQRT: bool>(complex: &[num_complex::Complex<f32>], dst: &mut [f32]) {
    debug_assert_eq!(complex.len(), dst.len());

    let n = complex.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    for i in 0..chunks {
        let off = i * 8;
        let (re, im) = load_complex8(complex, off);
        let power = re.mul_add(re, im * im);
        store8(dst, off, if SQRT { power.sqrt() } else { power });
    }

    // Scalar tail.
    for i in tail_start..n {
        let c = &complex[i];
        let p = c.re * c.re + c.im * c.im;
        dst[i] = if SQRT { libm::sqrtf(p) } else { p };
    }
}
