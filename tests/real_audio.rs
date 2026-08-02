//! Integration tests using real audio assets to verify fingerprinter robustness.
//!
//! Real broadband music and speech are richer than synthetic two-tone signals
//! and provide a more realistic test of spectral peak survival and bit similarity.
#![cfg(feature = "std")]

use std::collections::HashSet;

use audiofp::classical::{Haitsma, Panako, StreamingHaitsma, StreamingPanako, StreamingWang, Wang};
use audiofp::io::{decode_to_mono, decode_to_mono_at};
use audiofp::{Fingerprinter, SampleRate, StreamingFingerprinter};

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
    wang.extract(samples, SampleRate::new(sr).unwrap())
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn panako_hash_set(samples: &[f32], sr: u32) -> HashSet<u32> {
    let mut p = Panako::default();
    p.extract(samples, SampleRate::new(sr).unwrap())
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn haitsma_frames(samples: &[f32], sr: u32) -> Vec<u32> {
    let mut h = Haitsma::default();
    h.extract(samples, SampleRate::new(sr).unwrap()).unwrap().frames
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
    let overlap_lp = jaccard(
        &wang_hash_set(&clean, 8_000),
        &wang_hash_set(&lowpassed, 8_000),
    );
    assert!(
        overlap_lp >= 0.50,
        "Wang speech Jaccard under lowpass = {overlap_lp:.3} (threshold 0.50)",
    );

    // --- Panako ---
    // Test Noise Robustness (30 dB SNR)
    let overlap_noise_panako = jaccard(
        &panako_hash_set(&clean, 8_000),
        &panako_hash_set(&noisy, 8_000),
    );
    assert!(
        overlap_noise_panako >= 0.20,
        "Panako speech Jaccard at 30 dB SNR = {overlap_noise_panako:.3} (threshold 0.20)",
    );

    // Test Lowpass Robustness
    let overlap_lp_panako = jaccard(
        &panako_hash_set(&clean, 8_000),
        &panako_hash_set(&lowpassed, 8_000),
    );
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
    let overlap_lp_wang = jaccard(
        &wang_hash_set(&clean, 8_000),
        &wang_hash_set(&lowpassed, 8_000),
    );
    assert!(
        overlap_lp_wang >= 0.50,
        "Wang piano Jaccard under lowpass = {overlap_lp_wang:.3} (threshold 0.50)",
    );

    // Test Lowpass Robustness for Panako
    let overlap_lp_panako = jaccard(
        &panako_hash_set(&clean, 8_000),
        &panako_hash_set(&lowpassed, 8_000),
    );
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

#[test]
fn real_audio_streaming_equivalence() {
    let path = "tests/assets/speech.ogg";
    let samples_8k = decode_to_mono_at(path, 8_000).expect("failed to decode at 8kHz");

    // --- Wang ---
    let mut wang_offline = Wang::default();
    let off_wang = wang_offline
        .extract(&samples_8k, SampleRate::HZ_8000,
        )
        .unwrap()
        .hashes;

    let mut wang_stream = StreamingWang::default();
    let mut online_wang = Vec::new();

    let chunk_sizes = [128, 512, 1024, 256, 2048, 128];
    let mut cursor = 0;
    while cursor < samples_8k.len() {
        let chunk_len = chunk_sizes[cursor % chunk_sizes.len()].min(samples_8k.len() - cursor);
        let end = cursor + chunk_len;
        online_wang.extend(
            wang_stream
                .push(&samples_8k[cursor..end]).unwrap().into_iter()
                .map(|(_, h)| h),
        );
        cursor = end;
    }
    online_wang.extend(wang_stream.flush().unwrap().into_iter().map(|(_, h)| h));

    let mut a_wang = off_wang;
    let mut b_wang = online_wang;
    a_wang.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
    b_wang.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
    assert_eq!(a_wang, b_wang);

    // --- Panako ---
    let mut panako_offline = Panako::default();
    let off_panako = panako_offline
        .extract(&samples_8k, SampleRate::HZ_8000,
        )
        .unwrap()
        .hashes;

    let mut panako_stream = StreamingPanako::default();
    let mut online_panako = Vec::new();
    let mut cursor = 0;
    while cursor < samples_8k.len() {
        let chunk_len = chunk_sizes[cursor % chunk_sizes.len()].min(samples_8k.len() - cursor);
        let end = cursor + chunk_len;
        online_panako.extend(
            panako_stream
                .push(&samples_8k[cursor..end]).unwrap().into_iter()
                .map(|(_, h)| h),
        );
        cursor = end;
    }
    online_panako.extend(panako_stream.flush().unwrap().into_iter().map(|(_, h)| h));

    let mut a_panako = off_panako;
    let mut b_panako = online_panako;
    a_panako.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    b_panako.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
    assert_eq!(a_panako, b_panako);

    // --- Haitsma ---
    let samples_5k = decode_to_mono_at(path, 5_000).expect("failed to decode at 5kHz");
    let mut haitsma_offline = Haitsma::default();
    let off_haitsma = haitsma_offline
        .extract(&samples_5k, SampleRate::HZ_5000,
        )
        .unwrap()
        .frames;

    let mut haitsma_stream = StreamingHaitsma::default();
    let mut online_haitsma = Vec::new();
    let mut cursor = 0;
    while cursor < samples_5k.len() {
        let chunk_len = chunk_sizes[cursor % chunk_sizes.len()].min(samples_5k.len() - cursor);
        let end = cursor + chunk_len;
        online_haitsma.extend(
            haitsma_stream
                .push(&samples_5k[cursor..end]).unwrap().into_iter()
                .map(|(_, h)| h),
        );
        cursor = end;
    }
    online_haitsma.extend(haitsma_stream.flush().unwrap().into_iter().map(|(_, h)| h));

    assert_eq!(off_haitsma, online_haitsma);
}

#[test]
fn real_audio_resampler_and_channel_verification() {
    let path = "tests/assets/speech.ogg";

    // 1. decode_to_mono downmixes channels and yields native sampling rate
    let (native_samples, native_sr) = decode_to_mono(path).expect("failed to decode mono");
    assert!(native_sr > 0);
    assert!(!native_samples.is_empty());

    // 2. Resample using SincResampler directly
    let resampler = audiofp::dsp::resample::SincResampler::new(native_sr, 8_000);
    let resampled_manually = resampler.process(&native_samples);

    // 3. Compare with decode_to_mono_at output
    let resampled_auto = decode_to_mono_at(path, 8_000).expect("failed to decode at 8kHz");

    assert_eq!(resampled_manually.len(), resampled_auto.len());
    let diff: f32 = resampled_manually
        .iter()
        .zip(resampled_auto.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff < 1e-4, "Resampler difference too large: {}", diff);
}
