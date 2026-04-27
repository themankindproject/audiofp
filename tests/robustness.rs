//! Synthetic robustness tests.
//!
//! These tests **do not** put audio through a real codec (MP3 / AAC /
//! Opus). That would need committed audio fixtures plus an external
//! encoder (ffmpeg or similar) and is on the roadmap for a later release.
//!
//! Instead they apply two reproducible perturbations — additive Gaussian
//! noise at a known SNR, and a single-pole IIR lowpass — and verify that
//! each classical fingerprinter retains a minimum Jaccard overlap with
//! the clean reference. Together they cover the two main ways lossy
//! codecs degrade the signal:
//!
//! 1. **Noise injection** (≈ quantisation + entropy-coding artefacts):
//!    `add_noise` mixes uniform noise into the input at a target SNR.
//! 2. **Highband attenuation** (≈ MP3/AAC's joint-stereo and Opus's
//!    SILK-mode HF rolloff): `lowpass` runs a 1-pole IIR.
//!
//! Thresholds are calibrated empirically and chosen to be loose enough
//! that platform-specific f32 jitter won't cause flakes, but tight
//! enough that a real regression would trip them. Tighten them as the
//! algorithms improve.

use std::collections::HashSet;

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

const SECS: f32 = 10.0;

/// Same xorshift32 + two-tone synthesiser as the regression goldens.
fn synth(seed: u32, sr: u32) -> Vec<f32> {
    let n = (sr as f32 * SECS) as usize;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        let s = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
            + noise;
        out.push(s);
    }
    out
}

/// Add uniform noise at the requested signal-to-noise ratio (dB).
///
/// Uniform `[-1, 1]` noise has RMS `1/√3`; we scale it to the target
/// noise RMS computed from the signal RMS and the requested SNR.
fn add_noise(samples: &[f32], snr_db: f32, seed: u32) -> Vec<f32> {
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
fn lowpass(samples: &[f32], cutoff_normalised: f32) -> Vec<f32> {
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

fn jaccard<T: std::hash::Hash + Eq>(a: &HashSet<T>, b: &HashSet<T>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f32 / union as f32
}

fn wang_hash_set(samples: &[f32]) -> HashSet<u32> {
    let mut wang = Wang::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::HZ_8000,
    };
    wang.extract(buf)
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn panako_hash_set(samples: &[f32]) -> HashSet<u32> {
    let mut p = Panako::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::HZ_8000,
    };
    p.extract(buf)
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn haitsma_frames(samples: &[f32]) -> Vec<u32> {
    let mut h = Haitsma::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::new(5_000).unwrap(),
    };
    h.extract(buf).unwrap().frames
}

/// Bit-level Hamming similarity for Haitsma — frames are aligned 1-to-1
/// with the clean reference, so we can compare them position by position.
fn haitsma_similarity(clean: &[u32], dirty: &[u32]) -> f32 {
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

// ---- Wang ----

#[test]
fn wang_robust_to_30db_noise() {
    // Synthetic two-tone audio is noise-fragile — most "peaks" outside the
    // 880/1320 Hz tones come from the noise floor wiggles, so 30 dB SNR
    // shifts the peak set significantly. Real broadband music recovers
    // far more hashes; calibrate the threshold accordingly.
    let clean = synth(0xCAFE, 8_000);
    let dirty = add_noise(&clean, 30.0, 0xBEEF);
    let overlap = jaccard(&wang_hash_set(&clean), &wang_hash_set(&dirty));
    assert!(
        overlap >= 0.05,
        "Wang Jaccard at 30 dB SNR = {overlap:.3} (threshold 0.05)",
    );
}

#[test]
fn wang_robust_to_lowpass() {
    let clean = synth(0xCAFE, 8_000);
    // Cutoff 0.20 of the sample rate ≈ 1.6 kHz at 8 kHz — well above the
    // tones (880 Hz, 1320 Hz) so the spectral peaks survive.
    let dirty = lowpass(&clean, 0.20);
    let overlap = jaccard(&wang_hash_set(&clean), &wang_hash_set(&dirty));
    assert!(
        overlap >= 0.20,
        "Wang Jaccard under lowpass = {overlap:.3} (threshold 0.20)",
    );
}

// ---- Panako ----

#[test]
fn panako_robust_to_30db_noise() {
    let clean = synth(0xCAFE, 8_000);
    let dirty = add_noise(&clean, 30.0, 0xBEEF);
    let overlap = jaccard(&panako_hash_set(&clean), &panako_hash_set(&dirty));
    assert!(
        overlap >= 0.03,
        "Panako Jaccard at 30 dB SNR = {overlap:.3} (threshold 0.03)",
    );
}

#[test]
fn panako_robust_to_lowpass() {
    let clean = synth(0xCAFE, 8_000);
    let dirty = lowpass(&clean, 0.20);
    let overlap = jaccard(&panako_hash_set(&clean), &panako_hash_set(&dirty));
    assert!(
        overlap >= 0.10,
        "Panako Jaccard under lowpass = {overlap:.3} (threshold 0.10)",
    );
}

// ---- Haitsma ----

#[test]
fn haitsma_robust_to_30db_noise() {
    let clean = synth(0xCAFE, 5_000);
    let dirty = add_noise(&clean, 30.0, 0xBEEF);
    let sim = haitsma_similarity(&haitsma_frames(&clean), &haitsma_frames(&dirty));
    assert!(
        sim >= 0.75,
        "Haitsma bit similarity at 30 dB SNR = {sim:.3} (threshold 0.75)",
    );
}

#[test]
fn haitsma_robust_to_lowpass() {
    let clean = synth(0xCAFE, 5_000);
    // 25 % of sr at 5 kHz = 1.25 kHz, comfortably inside the 300–2000 Hz band.
    let dirty = lowpass(&clean, 0.25);
    let sim = haitsma_similarity(&haitsma_frames(&clean), &haitsma_frames(&dirty));
    assert!(
        sim >= 0.80,
        "Haitsma bit similarity under lowpass = {sim:.3} (threshold 0.80)",
    );
}
