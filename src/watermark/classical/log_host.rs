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

/// Linear gain for a +3 dB chip: `10^(3/20)`.
const GAIN_UP: f32 = 1.412_537_6;
/// Linear gain for a −3 dB chip: `10^(-3/20)`.
const GAIN_DOWN: f32 = 0.707_945_8;
const BIN_LO: usize = 10;
const BIN_HI_FRAC: f32 = 0.75;
const BINS_PER_FRAME: usize = 300;

pub(crate) fn embed(samples: &[f32], bits: &[u8], key: &[u8]) -> Vec<f32> {
    let mut stft = ComplexStft::new(N_FFT, HOP);
    let (mut spec, n_frames) = stft.analyze(samples);
    let nb = stft.n_bins();
    let order = log_order(key, nb);
    let n_bits = bits.len().max(1);
    let n_ord = order.len();
    let chips = chips(key, n_bits, n_ord);
    for f in 0..n_frames {
        let b = f % n_bits;
        let row = &chips[b * n_ord..(b + 1) * n_ord];
        // Bit 0 flips every chip; the chip itself picks the ±3 dB direction.
        let negate = bits[b] == 0;
        for (i, &bin) in order.iter().enumerate() {
            let cell = &mut spec[f * nb + bin];
            let mag = modulus(*cell);
            if mag < 1e-7 {
                continue;
            }
            let gain = if (row[i] > 0.0) != negate {
                GAIN_UP
            } else {
                GAIN_DOWN
            };
            let new_mag = (mag * gain).max(1e-7);
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
    let mut mean = vec![0.0f64; n_ord];
    for f in 0..n_frames {
        for (i, &bin) in order.iter().enumerate() {
            let mag = modulus(spec[f * nb + bin]).max(1e-7);
            let v = f64::from(20.0 * libm::log10f(mag));
            db[f * n_ord + i] = v;
            mean[i] += v;
        }
    }
    let inv = 1.0 / n_frames as f64;
    for m in &mut mean {
        *m *= inv;
    }
    let chips = chips(key, n_bits, n_ord);
    pick_phase(&db, &mean, n_frames, n_bits, &chips)
}

/// One ±1 chip row per bit, drawn from that bit's domain-separated stream.
///
/// The rows repeat every frame, so embed and detect consume them from this
/// table instead of rebuilding a keystream per frame — the previous shape
/// allocated one `derive_key` buffer per (phase, frame) pair.
fn chips(key: &[u8], n_bits: usize, n_ord: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_bits * n_ord);
    for b in 0..n_bits {
        let mut rng = ChaCha::new(&derive_key(key, b"log-host-spread"), b as u64);
        for _ in 0..n_ord {
            out.push(hard_chip(&mut rng));
        }
    }
    out
}

fn log_order(key: &[u8], nb: usize) -> Vec<usize> {
    let bin_hi = (nb as f32 * BIN_HI_FRAC) as usize;
    keyed_bins(key, b"log-host-bins", BIN_LO, bin_hi, BINS_PER_FRAME)
}

fn pick_phase(
    db: &[f64],
    mean: &[f64],
    n_frames: usize,
    n_bits: usize,
    chips: &[f32],
) -> (Vec<u8>, f64, usize) {
    let n_ord = mean.len();
    let mut corr = vec![0.0f64; n_bits];
    let mut best = vec![0.0f64; n_bits];
    let mut best_e = f64::NEG_INFINITY;
    let mut best_phase = 0usize;
    for ph0 in 0..n_bits {
        corr.fill(0.0);
        for f in 0..n_frames {
            let b = (f + ph0) % n_bits;
            let row = &chips[b * n_ord..(b + 1) * n_ord];
            for i in 0..n_ord {
                corr[b] += (db[f * n_ord + i] - mean[i]) * f64::from(row[i]);
            }
        }
        let energy: f64 = corr.iter().map(|c| c.abs()).sum();
        if energy > best_e {
            best_e = energy;
            best.copy_from_slice(&corr);
            best_phase = ph0;
        }
    }
    let bits: Vec<u8> = best.iter().map(|&c| u8::from(c >= 0.0)).collect();
    let mean = best.iter().map(|c| c.abs()).sum::<f64>() / n_bits.max(1) as f64;
    (bits, mean, best_phase)
}

fn hard_chip(rng: &mut ChaCha) -> f32 {
    if rng.u32() & 1 == 0 { -1.0 } else { 1.0 }
}
