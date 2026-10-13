//! Non-overlapping log-magnitude QIM.
//!
//! Each payload bit owns a fixed set of bins for the whole clip. The
//! detector votes per block: averaging dB across blocks first walks off
//! the lattice. Polarity matches the bakeoff: closer to the bit-1 lattice
//! (half a step) decodes as bit 1.

use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use rustfft::FftPlannerScalar;

use crate::dsp::stft::RealFft;

use super::N_FFT;
use super::keying::keyed_bins;
use super::stft::modulus;

const Q_DB: f32 = 6.0;
const BINS_PER_BIT: usize = 4;
const BIN_LO: usize = 20;
const BIN_HI: usize = 512;
const OFFSET_STEP: usize = 64;

/// True when `n_bits` can each take [`BINS_PER_BIT`] bins inside the carrier band.
pub(crate) fn fits(n_bits: usize) -> bool {
    n_bits.saturating_mul(BINS_PER_BIT) <= BIN_HI - BIN_LO
}

pub(crate) fn embed(samples: &[f32], bits: &[u8], key: &[u8]) -> Vec<f32> {
    let n_bits = bits.len().max(1);
    let order = qim_bins(key, n_bits);
    let mut planner = FftPlannerScalar::<f32>::new();
    let fwd = planner.plan_fft_forward(N_FFT);
    let inv = planner.plan_fft_inverse(N_FFT);
    let scratch_len = fwd
        .get_inplace_scratch_len()
        .max(inv.get_inplace_scratch_len());
    let mut scratch = vec![Complex::new(0.0, 0.0); scratch_len];
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut y = samples.to_vec();
    let mut pos = 0usize;
    while pos + N_FFT <= samples.len() {
        for i in 0..N_FFT {
            buf[i] = Complex::new(samples[pos + i], 0.0);
        }
        fwd.process_with_scratch(&mut buf, &mut scratch);
        for (b, &bit) in bits.iter().enumerate() {
            let target = if bit == 1 { Q_DB * 0.5 } else { 0.0 };
            for k in 0..BINS_PER_BIT {
                let idx = b + k * n_bits;
                if idx >= order.len() {
                    break;
                }
                let bin = order[idx];
                let mag = modulus(buf[bin]);
                if mag < 1e-6 {
                    continue;
                }
                let db = 20.0 * libm::log10f(mag);
                let new_mag = libm::powf(10.0, quantize_db(db, target, Q_DB) / 20.0).max(1e-7);
                set_herm_mag(&mut buf, bin, new_mag);
            }
        }
        inv.process_with_scratch(&mut buf, &mut scratch);
        let scale = 1.0 / N_FFT as f32;
        for i in 0..N_FFT {
            y[pos + i] = buf[i].re * scale;
        }
        pos += N_FFT;
    }
    y
}

/// Recovered bits, mean absolute lattice vote, and the winning sample offset.
pub(crate) fn detect(samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64, usize) {
    if n_bits == 0 || samples.len() < N_FFT {
        return (vec![0; n_bits], 0.0, 0);
    }
    let order = qim_bins(key, n_bits);
    if order.is_empty() {
        return (vec![0; n_bits], 0.0, 0);
    }
    let q = f64::from(Q_DB);
    // Detect consumes magnitudes of a real block only, so it uses the
    // crate's real-input FFT (half-length complex FFT + twiddle pass)
    // rather than a full complex transform: the classical fingerprinters
    // already run on this backend, and it is the target-independent one.
    let mut fft = RealFft::new(N_FFT);
    let mut time = vec![0.0f32; N_FFT];
    let mut spec = vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1];
    let mut sum = vec![0.0f64; n_bits];
    let mut cnt = vec![0u32; n_bits];
    let mut corr = vec![0.0f64; n_bits];
    let mut best = vec![0.0f64; n_bits];
    let mut best_e = f64::MIN;
    let mut best_offset = 0usize;
    let mut offset = 0usize;
    while offset < N_FFT && offset + N_FFT <= samples.len() {
        block_votes(
            samples,
            offset,
            &order,
            n_bits,
            q,
            &mut Spectrum {
                fft: &mut fft,
                time: &mut time,
                spec: &mut spec,
            },
            &mut sum,
            &mut cnt,
            &mut corr,
        );
        let energy: f64 = corr.iter().map(|c| c.abs()).sum();
        if energy > best_e {
            best_e = energy;
            best.copy_from_slice(&corr);
            best_offset = offset;
        }
        offset += OFFSET_STEP;
    }
    let bits: Vec<u8> = best.iter().map(|&c| u8::from(c >= 0.0)).collect();
    let mean = best.iter().map(|c| c.abs()).sum::<f64>() / n_bits as f64;
    (bits, mean, best_offset)
}

fn qim_bins(key: &[u8], n_bits: usize) -> Vec<usize> {
    keyed_bins(
        key,
        b"qim-block-bins",
        BIN_LO,
        BIN_HI,
        n_bits.saturating_mul(BINS_PER_BIT),
    )
}

fn lattice_dist(db: f64, offset: f64, q: f64) -> f64 {
    let x = (db - offset) / q;
    (x - x.round()).abs()
}

fn quantize_db(db: f32, target: f32, q: f32) -> f32 {
    let step = ((db - target) / q).round();
    step * q + target
}

fn set_herm_mag(buf: &mut [Complex<f32>], bin: usize, new_mag: f32) {
    let n = buf.len();
    if bin == 0 || bin >= n / 2 {
        return;
    }
    let m = modulus(buf[bin]);
    if m < 1e-8 {
        return;
    }
    buf[bin] *= new_mag / m;
    buf[n - bin] = buf[bin].conj();
}

struct Spectrum<'a> {
    fft: &'a mut RealFft,
    time: &'a mut [f32],
    spec: &'a mut [Complex<f32>],
}

#[allow(clippy::too_many_arguments)]
fn block_votes(
    samples: &[f32],
    offset: usize,
    order: &[usize],
    n_bits: usize,
    q: f64,
    spec: &mut Spectrum<'_>,
    sum: &mut [f64],
    cnt: &mut [u32],
    out: &mut [f64],
) {
    sum.fill(0.0);
    cnt.fill(0);
    let mut pos = offset;
    while pos + N_FFT <= samples.len() {
        spec.time.copy_from_slice(&samples[pos..pos + N_FFT]);
        spec.fft.process(spec.time, spec.spec);
        for b in 0..n_bits {
            let mut vote = 0.0f64;
            let mut n = 0.0f64;
            for k in 0..BINS_PER_BIT {
                let idx = b + k * n_bits;
                if idx >= order.len() {
                    break;
                }
                let bin = order[idx];
                if bin == 0 || bin >= N_FFT / 2 {
                    continue;
                }
                let mag = modulus(spec.spec[bin]).max(1e-7);
                let db = f64::from(20.0 * libm::log10f(mag));
                vote += lattice_dist(db, 0.0, q) - lattice_dist(db, q * 0.5, q);
                n += 1.0;
            }
            if n > 0.0 {
                sum[b] += vote / n;
                cnt[b] += 1;
            }
        }
        pos += N_FFT;
    }
    for b in 0..n_bits {
        out[b] = if cnt[b] > 0 {
            sum[b] / f64::from(cnt[b])
        } else {
            0.0
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The detector runs on [`RealFft`]; the embedder (and the bakeoff
    /// reference) uses a full complex FFT. Pin that both backends produce
    /// the same magnitudes in the carrier band, so the substitution cannot
    /// move a lattice vote.
    #[test]
    fn real_fft_matches_complex_fft_in_carrier_band() {
        let fwd = FftPlannerScalar::<f32>::new().plan_fft_forward(N_FFT);
        let mut scratch = vec![Complex::new(0.0, 0.0); fwd.get_inplace_scratch_len()];
        let mut rfft = RealFft::new(N_FFT);
        let mut worst = 0.0f64;
        let mut x = 0x1234_5678u32;
        for _ in 0..8 {
            let mut time = vec![0.0f32; N_FFT];
            for t in &mut time {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *t = (x >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            }

            let mut complex: Vec<Complex<f32>> =
                time.iter().map(|&s| Complex::new(s, 0.0)).collect();
            fwd.process_with_scratch(&mut complex, &mut scratch);

            let mut real_time = time.clone();
            let mut real = vec![Complex::new(0.0, 0.0); N_FFT / 2 + 1];
            rfft.process(&mut real_time, &mut real);

            for bin in BIN_LO..BIN_HI.min(N_FFT / 2) {
                let a = modulus(complex[bin]);
                let b = modulus(real[bin]);
                worst = worst.max(f64::from((a - b).abs() / a.max(1e-30)));
            }
        }
        // Measured worst gap across eight frames: 4.8e-6 relative
        // (≈ 40 f32 ulps, −106 dB) — five orders below the 6 dB lattice
        // step, so a vote cannot move.
        assert!(worst < 1e-5, "real/complex magnitude gap {worst:.3e}");
    }
}
