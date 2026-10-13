//! Overlapped log-magnitude DSSS with per-bin host subtraction.
//!
//! Geometry matches the bakeoff `log_host`: FFT 2048, hop 512, ±3 dB chips,
//! 300 bins from bin 10 to 75% of the spectrum. The detector searches every
//! bit phase and keeps the one with the largest absolute correlation.

use alloc::vec;
use alloc::vec::Vec;

use super::keying::{ChaCha, derive_key, keyed_bins};
use super::stft::{ComplexStft, modulus};
use super::{HOP, N_FFT};

const DB_ALPHA: f32 = 3.0;
const BIN_LO: usize = 10;
const BIN_HI_FRAC: f32 = 0.75;
const BINS_PER_FRAME: usize = 300;

pub(crate) fn embed(samples: &[f32], bits: &[u8], key: &[u8]) -> Vec<f32> {
    let mut stft = ComplexStft::new(N_FFT, HOP);
    let (mut spec, n_frames) = stft.analyze(samples);
    let nb = stft.n_bins();
    let order = log_order(key, nb);
    let n_bits = bits.len().max(1);
    for f in 0..n_frames {
        let bit = bits[f % n_bits];
        let sign = bit_sign(bit);
        let mut rng = ChaCha::new(&derive_key(key, b"log-host-spread"), (f % n_bits) as u64);
        for &bin in &order {
            let chip = hard_chip(&mut rng);
            let cell = &mut spec[f * nb + bin];
            let mag = modulus(*cell);
            if mag < 1e-7 {
                continue;
            }
            let db = 20.0 * libm::log10f(mag);
            let new_mag = libm::powf(10.0, (db + sign * chip * DB_ALPHA) / 20.0).max(1e-7);
            *cell *= new_mag / mag;
        }
    }
    stft.synthesize(&spec, n_frames, samples.len())
}

/// Recovered bits, mean absolute correlation, and the winning bit phase.
pub(crate) fn detect(samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64, usize) {
    if n_bits == 0 {
        return (Vec::new(), 0.0, 0);
    }
    let mut stft = ComplexStft::new(N_FFT, HOP);
    let (spec, n_frames) = stft.analyze(samples);
    let nb = stft.n_bins();
    let order = log_order(key, nb);
    if n_frames == 0 || order.is_empty() {
        return (vec![0; n_bits], 0.0, 0);
    }
    let n_ord = order.len();
    let mut db = vec![0.0f64; n_frames * n_ord];
    for f in 0..n_frames {
        for (i, &bin) in order.iter().enumerate() {
            let mag = modulus(spec[f * nb + bin]).max(1e-7);
            db[f * n_ord + i] = f64::from(20.0 * libm::log10f(mag));
        }
    }
    let mut mean = vec![0.0f64; n_ord];
    for f in 0..n_frames {
        for i in 0..n_ord {
            mean[i] += db[f * n_ord + i];
        }
    }
    let inv = 1.0 / n_frames as f64;
    for m in &mut mean {
        *m *= inv;
    }
    pick_phase(n_bits, |ph0| {
        correlate(&db, &mean, n_frames, n_ord, n_bits, ph0, key)
    })
}

fn log_order(key: &[u8], nb: usize) -> Vec<usize> {
    let bin_hi = (nb as f32 * BIN_HI_FRAC) as usize;
    keyed_bins(key, b"log-host-bins", BIN_LO, bin_hi, BINS_PER_FRAME)
}

fn correlate(
    db: &[f64],
    mean: &[f64],
    n_frames: usize,
    n_ord: usize,
    n_bits: usize,
    ph0: usize,
    key: &[u8],
) -> Vec<f64> {
    let mut corr = vec![0.0f64; n_bits];
    for f in 0..n_frames {
        let b = (f + ph0) % n_bits;
        let mut rng = ChaCha::new(&derive_key(key, b"log-host-spread"), b as u64);
        for i in 0..n_ord {
            let chip = f64::from(hard_chip(&mut rng));
            corr[b] += (db[f * n_ord + i] - mean[i]) * chip;
        }
    }
    corr
}

fn pick_phase(n_bits: usize, correlate_at: impl Fn(usize) -> Vec<f64>) -> (Vec<u8>, f64, usize) {
    let mut best_phase = 0usize;
    let mut best = correlate_at(0);
    let mut best_e: f64 = best.iter().map(|c| c.abs()).sum();
    for ph0 in 1..n_bits {
        let corr = correlate_at(ph0);
        let energy: f64 = corr.iter().map(|c| c.abs()).sum();
        if energy > best_e {
            best_e = energy;
            best = corr;
            best_phase = ph0;
        }
    }
    let bits: Vec<u8> = best.iter().map(|&c| u8::from(c >= 0.0)).collect();
    let mean = best.iter().map(|c| c.abs()).sum::<f64>() / n_bits.max(1) as f64;
    (bits, mean, best_phase)
}

fn bit_sign(bit: u8) -> f32 {
    if bit == 0 { -1.0 } else { 1.0 }
}

fn hard_chip(rng: &mut ChaCha) -> f32 {
    if rng.u32() & 1 == 0 { -1.0 } else { 1.0 }
}
