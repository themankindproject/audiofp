//! Shared 8-wide SIMD helpers (`wide::f32x8`).
//!
//! Every DSP hot loop in the crate iterates `n/8` full chunks plus a
//! scalar tail. This module owns that skeleton once so there is a single
//! place to tune vector width, rounding behaviour, and fallback paths.
//!
//! All functions are `#[inline]` and `no_std + alloc` compatible; the
//! per-callsite wrappers in [`super`] (`dot_wide`, `power_to_db_wide`),
//! [`super::stft`], [`crate::neural::embedder`], and
//! [`crate::matching`] keep their names and delegate here.
//!
//! # Cross-platform determinism
//!
//! Every kernel here produces the same bits on every target. Only
//! operations IEEE 754 defines exactly are used — lane-wise `+ - * sqrt`
//! (each rounded once), comparisons, selects, and bit manipulation — in
//! an order that does not depend on the target's SIMD width. In
//! particular:
//!
//! - No fused multiply-add. `wide`'s `mul_add` is one rounding on NEON
//!   and on FMA-enabled x86 builds but two roundings elsewhere, so it is
//!   written out as `a * b + c`. Rust never contracts that into an FMA.
//! - Horizontal sums use a fixed lane order ([`hsum8`]); `wide`'s
//!   `reduce_add` changes order on AVX builds.
//! - The vector `log2` is [`log2_lanes`], not `wide`'s `log2`: `wide`
//!   evaluates its polynomial with `mul_add` and documents its precision
//!   as platform- and version-dependent.
//! - Scalar tails call `libm` rather than `std`'s float methods, which
//!   defer to the platform C library.
//!
//! The reference output is the one the crate has always produced on
//! baseline x86_64 (SSE2, no FMA), so golden hashes are unchanged there.

use bytemuck::cast;
use wide::{f32x8, i32x8, u32x8};

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

/// Horizontal sum of the 8 lanes in a fixed order:
/// `(((l0 + l1) + l2) + l3) + (((l4 + l5) + l6) + l7)`.
///
/// This is the order `wide::f32x8::reduce_add` uses on SSE2, NEON, WASM,
/// and scalar builds; its AVX path pairs lanes differently, so the order
/// is pinned here instead of inherited.
#[inline]
fn hsum8(v: f32x8) -> f32 {
    let l = v.to_array();
    let lo = ((l[0] + l[1]) + l[2]) + l[3];
    let hi = ((l[4] + l[5]) + l[6]) + l[7];
    lo + hi
}

/// Core dot product: `sum(a[i] * b[i])`, 8-wide with a scalar tail.
///
/// Products and sums are rounded separately (no FMA) and lanes are
/// reduced with [`hsum8`], so the result is identical on every target.
#[inline]
pub(crate) fn dot_core(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        acc = load8(a, off) * load8(b, off) + acc;
    }

    let mut sum = hsum8(acc);
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
        acc = x * x + acc;
    }

    let mut sumsq = hsum8(acc);
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
        acc = load8(a, off) * (vb * vb) + acc;
    }

    let mut sum = hsum8(acc);
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

/// Core power-to-dB conversion: `buf[i] = factor * log2(max(buf[i], floor))`
/// in place, 8-wide with a scalar tail.
///
/// The vector path uses [`log2_lanes`] and the scalar tail uses
/// `libm::log2f`; the two can differ by 1 ULP, but each is the same on
/// every target. See [`power_to_db_wide`](crate::dsp::power_to_db_wide).
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
        store8(buf, off, factor_v * log2_lanes(clamped));
    }

    // Scalar tail.
    for v in &mut buf[tail_start..] {
        *v = factor * libm::log2f(v.max(floor));
    }
}

/// Row-wise [`db_into`] that reproduces the rounding of one [`db_into`]
/// call over a larger buffer this row is part of.
///
/// `db_into` converts every element of a buffer 8-wide except its last
/// `len % 8` elements, which take the scalar path (the two can differ by
/// 1 ULP). Each vector lane depends only on its own input — the ops are
/// lane-wise and the special-value fixups in [`log2_lanes`] are per-lane
/// selects — so an element's result is fixed by its value and by which
/// side of that split it falls on, not by its neighbours. Elements
/// `[0, scalar_from)` of `row` are converted on the vector path and
/// `[scalar_from, len)` on the scalar path; with `scalar_from` taken from
/// the enclosing buffer's split, the output is bit-identical to having
/// converted the whole buffer at once. This lets a front-end convert one
/// spectrogram row at a time without materialising the spectrogram.
#[inline]
pub(crate) fn db_into_split(row: &mut [f32], floor: f32, factor: f32, scalar_from: usize) {
    let scalar_from = scalar_from.min(row.len());
    let vector_part = scalar_from - scalar_from % 8;
    db_into_vector_lanes(&mut row[..vector_part], floor, factor);
    if vector_part < scalar_from {
        // Remainder of the vector region: run it through one padded vector
        // so it gets vector-path rounding. The pad value is irrelevant
        // (lanes are independent); 1.0 keeps the lanes finite.
        let rem = &mut row[vector_part..scalar_from];
        let mut lanes = [1.0_f32; 8];
        lanes[..rem.len()].copy_from_slice(rem);
        db_into_vector_lanes(&mut lanes, floor, factor);
        rem.copy_from_slice(&lanes[..rem.len()]);
    }
    for v in &mut row[scalar_from..] {
        *v = factor * libm::log2f(v.max(floor));
    }
}

/// Vector-path body of [`db_into`]; `buf.len()` must be a multiple of 8.
#[inline]
fn db_into_vector_lanes(buf: &mut [f32], floor: f32, factor: f32) {
    debug_assert_eq!(buf.len() % 8, 0);
    let floor_v = f32x8::splat(floor);
    let factor_v = f32x8::splat(factor);
    for off in (0..buf.len()).step_by(8) {
        let clamped = load8(buf, off).max(floor_v);
        store8(buf, off, factor_v * log2_lanes(clamped));
    }
}

/// Lane-wise base-2 logarithm, identical on every target.
///
/// This is the Cephes-derived `logf` algorithm `wide` 1.6 uses for
/// `f32x8::log2` (`ln(x) * LOG2_E`), with each `mul_add` / `mul_neg_add`
/// written out as a separately rounded multiply and add. On baseline
/// x86_64 — no FMA, which is what `wide` compiles `mul_add` to there — the
/// result is bit-identical to `wide`'s for every input; on NEON and
/// FMA-enabled x86 `wide` fuses those steps, which is the cross-platform
/// divergence this function exists to remove.
///
/// Special values follow `wide`: `+inf -> +inf`, `0` and subnormals ->
/// `-inf`, negative inputs and `-inf` -> NaN, NaN -> NaN.
// The coefficients are copied verbatim from `wide` so each parses to the
// identical `f32`.
#[allow(clippy::excessive_precision)]
#[inline]
pub(crate) fn log2_lanes(x1: f32x8) -> f32x8 {
    let half = f32x8::splat(0.5);
    let one = f32x8::ONE;
    let p0 = f32x8::splat(3.3333331174E-1);
    let p1 = f32x8::splat(-2.4999993993E-1);
    let p2 = f32x8::splat(2.0000714765E-1);
    let p3 = f32x8::splat(-1.6668057665E-1);
    let p4 = f32x8::splat(1.4249322787E-1);
    let p5 = f32x8::splat(-1.2420140846E-1);
    let p6 = f32x8::splat(1.1676998740E-1);
    let p7 = f32x8::splat(-1.1514610310E-1);
    let p8 = f32x8::splat(7.0376836292E-2);
    let ln2_hi = f32x8::splat(0.693359375);
    let ln2_lo = f32x8::splat(-2.12194440e-4);
    let smallest_normal = f32x8::splat(f32::MIN_POSITIVE);

    let bits: u32x8 = cast(x1);
    // Mantissa scaled into [0.5, 1).
    let x: f32x8 = cast((bits & u32x8::splat(0x007F_FFFF)) | u32x8::splat(0x3F00_0000));
    // Unbiased exponent as a float: (2^23 + biased) - (2^23 + 127), exact.
    let pow2_23 = f32x8::splat(8_388_608.0);
    let biased: f32x8 = cast((bits >> 23_u32) | cast::<f32x8, u32x8>(pow2_23));
    let e = biased - (pow2_23 + f32x8::splat(127.0));

    let mask = x.simd_gt(f32x8::SQRT_2 * half);
    let x = (!mask).select(x + x, x);
    let fe = mask.select(e + one, e);
    let x = x - one;

    // 8th-degree polynomial with `wide`'s `polynomial_8!` grouping.
    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;
    let hi = x2 * (p7 * x + p6) + (x * p5 + p4);
    let lo = x8 * p8 + (x2 * (x * p3 + p2) + (x * p1 + p0));
    let poly = x4 * hi + lo;

    let res = x2 * x * poly;
    let res = fe * ln2_lo + res;
    let res = res + (x - x2 * half);
    let res = fe * ln2_hi + res;

    let overflow = !x1.is_finite();
    let underflow = x1.simd_lt(smallest_normal);
    let res = if (overflow | underflow).any() {
        let exponent_bits: i32x8 = cast(x1);
        let is_zero_or_subnormal: f32x8 =
            cast((exponent_bits & i32x8::splat(0x7F80_0000)).simd_eq(i32x8::splat(0)));
        let nan = f32x8::splat(f32::from_bits(0x7FC0_0101));
        let res = underflow.select(nan, res);
        let res = is_zero_or_subnormal.select(f32x8::splat(f32::NEG_INFINITY), res);
        let res = overflow.select(x1, res);
        (overflow & x1.is_sign_negative()).select(nan, res)
    } else {
        res
    };
    res * f32x8::LOG2_E
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
/// no-SIMD fallback) and both squares are rounded before the add, so the
/// vector and scalar paths are bit-identical on every target.
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
        let power = re * re + im * im;
        store8(dst, off, if SQRT { power.sqrt() } else { power });
    }

    // Scalar tail.
    for i in tail_start..n {
        let c = &complex[i];
        let p = c.re * c.re + c.im * c.im;
        dst[i] = if SQRT { libm::sqrtf(p) } else { p };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    fn xorshift(seed: u32) -> impl FnMut() -> u32 {
        let mut x = seed;
        move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        }
    }

    /// Random values in (-1, 1) built from exact operations.
    fn random_vec(seed: u32, n: usize) -> Vec<f32> {
        let mut next = xorshift(seed);
        (0..n)
            .map(|_| next() as i32 as f32 * (1.0 / 2_147_483_648.0))
            .collect()
    }

    /// Reference for the 8-lane accumulate + [`hsum8`] + scalar-tail shape
    /// shared by `dot_core`, `sumsq_core`, and `dot_sq_core`: `term(i)`
    /// is the product for element `i`, `fused` selects a single-rounding
    /// accumulate (what `wide`'s `mul_add` does on NEON / FMA builds).
    fn lane_sum_ref(n: usize, term: impl Fn(usize) -> (f32, f32), fused: bool) -> f32 {
        let mut lanes = [0.0_f32; 8];
        let chunks = n / 8;
        for c in 0..chunks {
            for (l, lane) in lanes.iter_mut().enumerate() {
                let (x, y) = term(c * 8 + l);
                *lane = if fused {
                    libm::fmaf(x, y, *lane)
                } else {
                    x * y + *lane
                };
            }
        }
        let mut sum = (((lanes[0] + lanes[1]) + lanes[2]) + lanes[3])
            + (((lanes[4] + lanes[5]) + lanes[6]) + lanes[7]);
        for i in chunks * 8..n {
            let (x, y) = term(i);
            sum += x * y;
        }
        sum
    }

    /// `wide`'s AVX `reduce_add` lane order, for the sensitivity check.
    fn avx_order(l: [f32; 8]) -> f32 {
        ((l[0] + l[4]) + (l[2] + l[6])) + ((l[1] + l[5]) + (l[3] + l[7]))
    }

    #[test]
    fn reductions_are_unfused_and_in_fixed_lane_order() {
        for (seed, n) in [(1_u32, 8_usize), (2, 13), (3, 64), (4, 257), (5, 1031)] {
            let a = random_vec(seed, n);
            let b = random_vec(seed.wrapping_mul(0x9E37_79B9), n);
            let dot = |i: usize| (a[i], b[i]);
            let sq = |i: usize| (a[i], a[i]);
            let dsq = |i: usize| (a[i], b[i] * b[i]);
            let bits = f32::to_bits;
            assert_eq!(
                bits(dot_core(&a, &b)),
                bits(lane_sum_ref(n, dot, false)),
                "dot n={n}"
            );
            assert_eq!(
                bits(sumsq_core(&a)),
                bits(lane_sum_ref(n, sq, false)),
                "sumsq n={n}"
            );
            assert_eq!(
                bits(dot_sq_core(&a, &b)),
                bits(lane_sum_ref(n, dsq, false)),
                "dot_sq n={n}"
            );
        }
        // The data must be able to tell fused from unfused accumulation and
        // the pinned lane order from AVX's, or the test above proves nothing.
        let a = random_vec(7, 1031);
        let b = random_vec(8, 1031);
        let dot = |i: usize| (a[i], b[i]);
        assert_ne!(
            lane_sum_ref(1031, dot, false).to_bits(),
            lane_sum_ref(1031, dot, true).to_bits(),
            "inputs do not distinguish fused accumulation"
        );
        let order_differs = a
            .chunks_exact(8)
            .zip(b.chunks_exact(8))
            .filter(|(x, y)| {
                let lanes: [f32; 8] = core::array::from_fn(|l| x[l] * 1e4 + y[l]);
                hsum8(f32x8::new(lanes)).to_bits() != avx_order(lanes).to_bits()
            })
            .count();
        assert!(order_differs > 0, "inputs do not distinguish lane order");
    }

    #[test]
    fn complex_power_rounds_each_square() {
        let re = random_vec(11, 1027);
        let im = random_vec(12, 1027);
        let spec: Vec<num_complex::Complex<f32>> = re
            .iter()
            .zip(&im)
            .map(|(&r, &i)| num_complex::Complex::new(r, i))
            .collect();
        let mut power = alloc::vec![0.0_f32; spec.len()];
        let mut mag = alloc::vec![0.0_f32; spec.len()];
        complex_power_into(&spec, &mut power);
        complex_magnitude_into(&spec, &mut mag);
        let mut fused_differs = false;
        for (k, c) in spec.iter().enumerate() {
            let want = c.re * c.re + c.im * c.im;
            assert_eq!(power[k].to_bits(), want.to_bits(), "power bin {k}");
            assert_eq!(
                mag[k].to_bits(),
                libm::sqrtf(want).to_bits(),
                "magnitude bin {k}"
            );
            fused_differs |= libm::fmaf(c.re, c.re, c.im * c.im).to_bits() != want.to_bits();
        }
        assert!(fused_differs, "inputs do not distinguish fused evaluation");
    }

    const NAN_LOG: u32 = 0x7FC0_0101;

    /// Scalar statement of [`log2_lanes`]; `fused` evaluates every
    /// multiply-add with one rounding, as `wide::f32x8::log2` does on NEON
    /// and FMA-enabled x86.
    #[allow(clippy::excessive_precision)]
    fn log2_ref(x1: f32, fused: bool) -> f32 {
        let ma = |a: f32, b: f32, c: f32| {
            if fused {
                libm::fmaf(a, b, c)
            } else {
                a * b + c
            }
        };
        let bits = x1.to_bits();
        let x = f32::from_bits((bits & 0x007F_FFFF) | 0x3F00_0000);
        let e = f32::from_bits((bits >> 23) | 8_388_608.0_f32.to_bits()) - (8_388_608.0 + 127.0);
        let gt = x > core::f32::consts::SQRT_2 * 0.5;
        let x = if gt { x } else { x + x };
        let fe = if gt { e + 1.0 } else { e };
        let x = x - 1.0;
        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;
        let hi = ma(
            x2,
            ma(-1.1514610310E-1, x, 1.1676998740E-1),
            ma(x, -1.2420140846E-1, 1.4249322787E-1),
        );
        let lo = ma(
            x8,
            7.0376836292E-2,
            ma(
                x2,
                ma(x, -1.6668057665E-1, 2.0000714765E-1),
                ma(x, -2.4999993993E-1, 3.3333331174E-1),
            ),
        );
        let res = x2 * x * ma(x4, hi, lo);
        let res = ma(fe, -2.12194440e-4, res);
        let res = res + ma(-x2, 0.5, x);
        let mut res = ma(fe, 0.693359375, res);
        if !x1.is_finite() || x1 < f32::MIN_POSITIVE {
            if x1 < f32::MIN_POSITIVE {
                res = f32::from_bits(NAN_LOG);
            }
            if bits & 0x7F80_0000 == 0 {
                res = f32::NEG_INFINITY;
            }
            if !x1.is_finite() {
                res = if x1.is_sign_negative() {
                    f32::from_bits(NAN_LOG)
                } else {
                    x1
                };
            }
        }
        res * core::f32::consts::LOG2_E
    }

    /// Every 4099th bit pattern (plus the special values) — about 1M
    /// inputs spread over every exponent, sign, and NaN payload class.
    fn log2_sweep() -> impl Iterator<Item = f32> {
        (0..=u32::MAX / 4099)
            .map(|i| f32::from_bits(i * 4099))
            .chain([
                0.0,
                -0.0,
                1.0,
                f32::MIN_POSITIVE,
                f32::MAX,
                f32::INFINITY,
                -f32::INFINITY,
                f32::NAN,
            ])
    }

    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    #[test]
    fn log2_lanes_matches_unfused_reference() {
        let inputs: Vec<f32> = log2_sweep().collect();
        let mut fused_differs = 0usize;
        for chunk in inputs.chunks(8) {
            let mut lanes = [1.0_f32; 8];
            lanes[..chunk.len()].copy_from_slice(chunk);
            let got = log2_lanes(f32x8::new(lanes)).to_array();
            for (l, &x) in chunk.iter().enumerate() {
                let want = log2_ref(x, false);
                assert!(
                    same(got[l], want),
                    "log2({x:e} = {:#x}): {} vs {want}",
                    x.to_bits(),
                    got[l]
                );
                fused_differs += usize::from(!same(log2_ref(x, true), want));
            }
        }
        // `wide`'s fused evaluation disagrees on some of these inputs (56 of
        // the ~1.05M), so a regression to it cannot pass on NEON / FMA
        // targets.
        assert!(
            fused_differs > 0,
            "inputs do not distinguish fused evaluation"
        );
    }

    /// `log2_lanes` reproduces `wide::f32x8::log2` for every `f32` on the
    /// targets where `wide` does not fuse — the historical reference. Run
    /// with `cargo test --release --lib log2_lanes_matches_wide -- --ignored`
    /// (all 2^32 inputs).
    #[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
    #[test]
    #[ignore = "exhaustive: ~2^32 evaluations"]
    fn log2_lanes_matches_wide_log2_exhaustively() {
        let mut lanes = [0.0_f32; 8];
        for hi in 0..=(u32::MAX >> 3) {
            for (l, lane) in lanes.iter_mut().enumerate() {
                *lane = f32::from_bits((hi << 3) | l as u32);
            }
            let v = f32x8::new(lanes);
            let (ours, theirs) = (log2_lanes(v).to_array(), v.log2().to_array());
            for (l, (o, t)) in ours.iter().zip(&theirs).enumerate() {
                assert!(same(*o, *t), "input {:#x}", (hi << 3) | l as u32);
            }
        }
    }

    #[test]
    fn db_into_split_rows_match_whole_buffer_bit_for_bit() {
        let factor = crate::dsp::DB_LOG2_FACTOR;
        let floor = 1e-12_f32;
        let mut x: u32 = 0xD1B;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x
        };
        for n_bins in [1usize, 3, 7, 8, 9, 15, 16, 17, 129, 513, 1025] {
            for n_rows in [1usize, 2, 3, 5, 8] {
                // Powers spanning the floor, subnormals, zeros, and large values.
                let flat: Vec<f32> = (0..n_rows * n_bins)
                    .map(|_| match next() % 7 {
                        0 => 0.0,
                        1 => 1e-40,
                        2 => 1e-13,
                        _ => f32::from_bits(next() % 0x7F00_0000),
                    })
                    .collect();
                let mut whole = flat.clone();
                db_into(&mut whole, floor, factor);

                let total = flat.len();
                let simd_end = total - total % 8;
                let mut rows = flat.clone();
                for (r, row) in rows.chunks_exact_mut(n_bins).enumerate() {
                    db_into_split(row, floor, factor, simd_end.saturating_sub(r * n_bins));
                }
                let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&rows), bits(&whole), "n_bins={n_bins} n_rows={n_rows}");
            }
        }
    }
}
