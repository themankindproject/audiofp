//! Integration tests using real audio assets to verify fingerprinter robustness.
//!
//! Real broadband music and speech are richer than synthetic two-tone signals
//! and provide a more realistic test of spectral peak survival and bit similarity.
#![cfg(feature = "std")]

use std::collections::HashSet;

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

// ---- Helper Functions ----

/// Add uniform noise at the requested signal-to-noise ratio (dB).
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

fn wang_hash_set(samples: &[f32], sr: u32) -> HashSet<u32> {
    let mut wang = Wang::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::new(sr).unwrap(),
    };
    wang.extract(buf)
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn panako_hash_set(samples: &[f32], sr: u32) -> HashSet<u32> {
    let mut p = Panako::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::new(sr).unwrap(),
    };
    p.extract(buf)
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn haitsma_frames(samples: &[f32], sr: u32) -> Vec<u32> {
    let mut h = Haitsma::default();
    let buf = AudioBuffer {
        samples,
        rate: SampleRate::new(sr).unwrap(),
    };
    h.extract(buf).unwrap().frames
}

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

// ---- Tests ----

#[test]
fn real_audio_speech_robustness() {
    let path = "tests/assets/speech.ogg";
    let clean = decode_to_mono_at(path, 8_000).expect("failed to decode speech.ogg at 8kHz");
    
    // --- Wang ---
    // Test Noise Robustness (30 dB SNR)
    let noisy = add_noise(&clean, 30.0, 0x1234);
    let overlap_noise = jaccard(&wang_hash_set(&clean, 8_000), &wang_hash_set(&noisy, 8_000));
    assert!(
        overlap_noise >= 0.25,
        "Wang speech Jaccard at 30 dB SNR = {overlap_noise:.3} (threshold 0.25)",
    );

    // Test Lowpass Robustness
    let lowpassed = lowpass(&clean, 0.20);
    let overlap_lp = jaccard(&wang_hash_set(&clean, 8_000), &wang_hash_set(&lowpassed, 8_000));
    assert!(
        overlap_lp >= 0.50,
        "Wang speech Jaccard under lowpass = {overlap_lp:.3} (threshold 0.50)",
    );

    // --- Panako ---
    // Test Noise Robustness (30 dB SNR)
    let overlap_noise_panako = jaccard(&panako_hash_set(&clean, 8_000), &panako_hash_set(&noisy, 8_000));
    assert!(
        overlap_noise_panako >= 0.20,
        "Panako speech Jaccard at 30 dB SNR = {overlap_noise_panako:.3} (threshold 0.20)",
    );

    // Test Lowpass Robustness
    let overlap_lp_panako = jaccard(&panako_hash_set(&clean, 8_000), &panako_hash_set(&lowpassed, 8_000));
    assert!(
        overlap_lp_panako >= 0.40,
        "Panako speech Jaccard under lowpass = {overlap_lp_panako:.3} (threshold 0.40)",
    );
}

#[test]
fn real_audio_piano_robustness() {
    let path = "tests/assets/piano.ogg";
    let clean = decode_to_mono_at(path, 8_000).expect("failed to decode piano.ogg at 8kHz");

    // Test Lowpass Robustness for Wang
    let lowpassed = lowpass(&clean, 0.20);
    let overlap_lp_wang = jaccard(&wang_hash_set(&clean, 8_000), &wang_hash_set(&lowpassed, 8_000));
    assert!(
        overlap_lp_wang >= 0.50,
        "Wang piano Jaccard under lowpass = {overlap_lp_wang:.3} (threshold 0.50)",
    );

    // Test Lowpass Robustness for Panako
    let overlap_lp_panako = jaccard(&panako_hash_set(&clean, 8_000), &panako_hash_set(&lowpassed, 8_000));
    assert!(
        overlap_lp_panako >= 0.40,
        "Panako piano Jaccard under lowpass = {overlap_lp_panako:.3} (threshold 0.40)",
    );
}

#[test]
fn real_audio_haitsma_robustness() {
    let path = "tests/assets/speech.ogg";
    let clean = decode_to_mono_at(path, 5_000).expect("failed to decode speech.ogg at 5kHz");

    // Test Noise Robustness (30 dB SNR)
    let noisy = add_noise(&clean, 30.0, 0x1234);
    let clean_frames = haitsma_frames(&clean, 5_000);
    let noisy_frames = haitsma_frames(&noisy, 5_000);
    let sim_noise = haitsma_similarity(&clean_frames, &noisy_frames);
    assert!(
        sim_noise >= 0.85,
        "Haitsma bit similarity at 30 dB SNR = {sim_noise:.3} (threshold 0.85)",
    );

    // Test Lowpass Robustness
    let lowpassed = lowpass(&clean, 0.25);
    let lp_frames = haitsma_frames(&lowpassed, 5_000);
    let sim_lp = haitsma_similarity(&clean_frames, &lp_frames);
    assert!(
        sim_lp >= 0.88,
        "Haitsma bit similarity under lowpass = {sim_lp:.3} (threshold 0.88)",
    );
}

#[test]
fn real_audio_silence_handling() {
    let silence = vec![0.0_f32; 8000 * 5]; // 5 seconds of silence at 8kHz

    // Wang should not panic, and should produce zero hashes
    let w_hashes = wang_hash_set(&silence, 8_000);
    assert_eq!(w_hashes.len(), 0);

    // Panako should not panic, and should produce zero hashes
    let p_hashes = panako_hash_set(&silence, 8_000);
    assert_eq!(p_hashes.len(), 0);

    // Haitsma should not panic, and should produce frames (since energy difference of 0 is just constant bits)
    let silence_5k = vec![0.0_f32; 5000 * 5];
    let h_frames = haitsma_frames(&silence_5k, 5_000);
    assert!(!h_frames.is_empty());
}
