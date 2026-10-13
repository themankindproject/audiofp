//! Non-overlapping log-magnitude QIM.
//!
//! Each payload bit owns a fixed set of bins for the whole clip. The
//! detector votes per block: averaging dB across blocks first walks off
//! the lattice. Polarity matches the bakeoff: closer to the bit-1 lattice
//! (half a step) decodes as bit 1.

use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use rustfft::{Fft, FftPlannerScalar};

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
    let mut planner = FftPlannerScalar::<f32>::new();
    let fwd = planner.plan_fft_forward(N_FFT);
    let mut scratch = vec![Complex::new(0.0, 0.0); fwd.get_inplace_scratch_len()];
    let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
    let mut best = vec![0.0f64; n_bits];
    let mut best_e = f64::MIN;
    let mut best_offset = 0usize;
    let mut offset = 0usize;
    while offset < N_FFT && offset + N_FFT <= samples.len() {
        let corr = block_votes(
            samples,
            offset,
            &order,
            n_bits,
            q,
            &mut Spectrum {
                fwd: fwd.as_ref(),
                scratch: &mut scratch,
                buf: &mut buf,
            },
        );
        let energy: f64 = corr.iter().map(|c| c.abs()).sum();
        if energy > best_e {
            best_e = energy;
            best = corr;
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
    fwd: &'a dyn Fft<f32>,
    scratch: &'a mut [Complex<f32>],
    buf: &'a mut [Complex<f32>],
}

fn block_votes(
    samples: &[f32],
    offset: usize,
    order: &[usize],
    n_bits: usize,
    q: f64,
    spec: &mut Spectrum<'_>,
) -> Vec<f64> {
    let mut sum = vec![0.0f64; n_bits];
    let mut cnt = vec![0u32; n_bits];
    let mut pos = offset;
    while pos + N_FFT <= samples.len() {
        for (i, sample) in spec.buf.iter_mut().enumerate() {
            *sample = Complex::new(samples[pos + i], 0.0);
        }
        spec.fwd.process_with_scratch(spec.buf, spec.scratch);
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
                let mag = modulus(spec.buf[bin]).max(1e-7);
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
    let mut corr = vec![0.0f64; n_bits];
    for b in 0..n_bits {
        if cnt[b] > 0 {
            corr[b] = sum[b] / f64::from(cnt[b]);
        }
    }
    corr
}
