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

mod common;

use common::degrade::{
    add_noise, haitsma_frames, haitsma_similarity, jaccard, lowpass, panako_hash_set, wang_hash_set,
};

const SECS: f32 = 10.0;

/// The shared two-tone signal, [`SECS`] long.
fn synth(seed: u32, sr: u32) -> Vec<f32> {
    common::audio_gen::two_tone(seed, sr, (sr as f32 * SECS) as usize)
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
    let overlap = jaccard(&wang_hash_set(&clean, 8_000), &wang_hash_set(&dirty, 8_000));
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
    let overlap = jaccard(&wang_hash_set(&clean, 8_000), &wang_hash_set(&dirty, 8_000));
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
    let overlap = jaccard(
        &panako_hash_set(&clean, 8_000),
        &panako_hash_set(&dirty, 8_000),
    );
    assert!(
        overlap >= 0.03,
        "Panako Jaccard at 30 dB SNR = {overlap:.3} (threshold 0.03)",
    );
}

#[test]
fn panako_robust_to_lowpass() {
    let clean = synth(0xCAFE, 8_000);
    let dirty = lowpass(&clean, 0.20);
    let overlap = jaccard(
        &panako_hash_set(&clean, 8_000),
        &panako_hash_set(&dirty, 8_000),
    );
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
    let sim = haitsma_similarity(
        &haitsma_frames(&clean, 5_000),
        &haitsma_frames(&dirty, 5_000),
    );
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
    let sim = haitsma_similarity(
        &haitsma_frames(&clean, 5_000),
        &haitsma_frames(&dirty, 5_000),
    );
    assert!(
        sim >= 0.80,
        "Haitsma bit similarity under lowpass = {sim:.3} (threshold 0.80)",
    );
}
