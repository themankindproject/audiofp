//! Signal degradations and fingerprint-overlap measures shared by the
//! robustness tests (`real_audio`, `robustness`).
#![allow(dead_code)]

use std::collections::HashSet;

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::{Fingerprinter, SampleRate};

/// Add uniform noise at the requested signal-to-noise ratio (dB).
///
/// Uniform `[-1, 1]` noise has RMS `1/√3`; we scale it to the target
/// noise RMS computed from the signal RMS and the requested SNR.
pub fn add_noise(samples: &[f32], snr_db: f32, seed: u32) -> Vec<f32> {
    let signal_power: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    let signal_rms = signal_power.sqrt();
    let noise_rms = signal_rms / 10f32.powf(snr_db / 20.0);
    let noise_amp = noise_rms * 3f32.sqrt();

    let mut x = seed.max(1);
    samples
        .iter()
        .map(|s| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let n = (x as i32 as f32) / (i32::MAX as f32) * noise_amp;
            s + n
        })
        .collect()
}

/// 1-pole IIR lowpass with a normalised cutoff `f_c / sr`.
///
/// Cutoff = 0.05 means the −3 dB point is 5 % of the sample rate
/// (i.e. 400 Hz at 8 kHz, mimicking a fairly aggressive HF rolloff).
pub fn lowpass(samples: &[f32], cutoff_normalised: f32) -> Vec<f32> {
    let alpha = 1.0 - (-2.0 * std::f32::consts::PI * cutoff_normalised).exp();
    let mut y = 0.0_f32;
    samples
        .iter()
        .map(|&s| {
            y = alpha * s + (1.0 - alpha) * y;
            y
        })
        .collect()
}

pub fn jaccard<T: std::hash::Hash + Eq>(a: &HashSet<T>, b: &HashSet<T>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f32 / union as f32
}

pub fn wang_hash_set(samples: &[f32], sr: u32) -> HashSet<u32> {
    let mut wang = Wang::default();
    wang.extract(samples, SampleRate::new(sr).expect("non-zero rate"))
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

pub fn panako_hash_set(samples: &[f32], sr: u32) -> HashSet<u32> {
    let mut p = Panako::default();
    p.extract(samples, SampleRate::new(sr).expect("non-zero rate"))
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

pub fn haitsma_frames(samples: &[f32], sr: u32) -> Vec<u32> {
    let mut h = Haitsma::default();
    h.extract(samples, SampleRate::new(sr).expect("non-zero rate"))
        .unwrap()
        .frames
}

/// Bit-level Hamming similarity for Haitsma — frames are aligned 1-to-1
/// with the clean reference, so we can compare them position by position.
pub fn haitsma_similarity(clean: &[u32], dirty: &[u32]) -> f32 {
    let n = clean.len().min(dirty.len());
    if n == 0 {
        return 0.0;
    }
    let total_bits = (n as u32) * 32;
    let matching: u32 = clean[..n]
        .iter()
        .zip(dirty[..n].iter())
        .map(|(a, b)| 32 - (a ^ b).count_ones())
        .sum();
    matching as f32 / total_bits as f32
}
