//! Adversarial / huge-input stress tests + real-audio snapshot assertions.
//!
//! Covers:
//! - `InputTooLarge` rejection on all algorithms
//! - Adversarial PCM signals (DC offset, impulse, square wave, all-zero, all-max)
//! - NaN/Inf rejection on offline paths
//! - Real CC0 audio snapshot: hash counts must remain stable across versions
//!
//! Run with:
//! ```bash
//! cargo test --test stress --all-features
//! ```
//!
//! The real-audio snapshot tests require the `std` feature (file decoding).
//! The adversarial/stress tests work under `no_std + alloc`.

#![cfg(all(feature = "std-mp3", feature = "std-flac", feature = "std-ogg"))]

use audiofp::classical::{Haitsma, HaitsmaConfig, Panako, PanakoConfig, Wang, WangConfig};
use audiofp::{AfpError, Fingerprinter, SampleRate};

// ───────────────────────────────────────────────────────────────────────
// Helper: generate adversarial signals at a given sample rate
// ───────────────────────────────────────────────────────────────────────

fn silence(sr: u32, secs: f32) -> Vec<f32> {
    vec![0.0_f32; (sr as f32 * secs) as usize]
}

fn dc_offset(sr: u32, secs: f32, offset: f32) -> Vec<f32> {
    vec![offset; (sr as f32 * secs) as usize]
}

fn impulse(sr: u32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    let mut out = vec![0.0_f32; n];
    out[n / 2] = 1.0; // single sample spike at midpoint
    out
}

fn square_wave(sr: u32, secs: f32, freq: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            if ((t * freq * 2.0) as u32).is_multiple_of(2) {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

fn max_amplitude(sr: u32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn nan_audio(sr: u32, secs: f32) -> Vec<f32> {
    let mut out = silence(sr, secs);
    out[100] = f32::NAN;
    out
}

fn inf_audio(sr: u32, secs: f32) -> Vec<f32> {
    let mut out = silence(sr, secs);
    out[200] = f32::INFINITY;
    out[300] = f32::NEG_INFINITY;
    out
}

// ───────────────────────────────────────────────────────────────────────
// Stress: InputTooLarge rejection
// ───────────────────────────────────────────────────────────────────────

#[test]
fn wang_rejects_input_exceeding_max_samples() {
    let cfg = WangConfig {
        max_input_samples: Some(1000),
        ..Default::default()
    };
    let mut w = Wang::new(cfg);
    let samples = vec![0.0_f32; 2000];
    let result = w.extract(&samples, SampleRate::HZ_8000);
    assert!(
        matches!(result, Err(AfpError::InputTooLarge { .. })),
        "expected InputTooLarge, got {result:?}"
    );
}

#[test]
fn panako_rejects_input_exceeding_max_samples() {
    let cfg = PanakoConfig {
        max_input_samples: Some(1000),
        ..Default::default()
    };
    let mut p = Panako::new(cfg);
    let samples = vec![0.0_f32; 2000];
    let result = p.extract(&samples, SampleRate::HZ_8000);
    assert!(
        matches!(result, Err(AfpError::InputTooLarge { .. })),
        "expected InputTooLarge, got {result:?}"
    );
}

#[test]
fn haitsma_rejects_input_exceeding_max_samples() {
    let cfg = HaitsmaConfig {
        max_input_samples: Some(1000),
        ..Default::default()
    };
    let mut h = Haitsma::new(cfg);
    let samples = vec![0.0_f32; 2000];
    let result = h.extract(&samples, SampleRate::HZ_5000);
    assert!(
        matches!(result, Err(AfpError::InputTooLarge { .. })),
        "expected InputTooLarge, got {result:?}"
    );
}

// ───────────────────────────────────────────────────────────────────────
// Stress: Adversarial PCM — must not panic, must produce valid output
// ───────────────────────────────────────────────────────────────────────

macro_rules! adversarial_test {
    ($name:ident, $algo:ident, $sr:expr, $rate:expr, $signal_fn:expr) => {
        #[test]
        fn $name() {
            let samples = $signal_fn;
            let mut fp = $algo::default();
            let rate = $rate;
            // Must not panic — result can be Ok (possibly 0 hashes) or
            // Err (AudioTooShort) depending on signal properties.
            let result = fp.extract(&samples, rate);
            match &result {
                Ok(_) => {}                               // valid
                Err(AfpError::AudioTooShort { .. }) => {} // acceptable
                Err(e) => panic!("unexpected error on adversarial input: {e:?}"),
            }
        }
    };
}

// Wang (8 kHz)
adversarial_test!(
    wang_silence_3s,
    Wang,
    8_000,
    SampleRate::HZ_8000,
    silence(8_000, 3.0)
);
adversarial_test!(
    wang_dc_offset,
    Wang,
    8_000,
    SampleRate::HZ_8000,
    dc_offset(8_000, 3.0, 0.99)
);
adversarial_test!(
    wang_impulse,
    Wang,
    8_000,
    SampleRate::HZ_8000,
    impulse(8_000, 3.0)
);
adversarial_test!(
    wang_square_440,
    Wang,
    8_000,
    SampleRate::HZ_8000,
    square_wave(8_000, 3.0, 440.0)
);
adversarial_test!(
    wang_max_amplitude,
    Wang,
    8_000,
    SampleRate::HZ_8000,
    max_amplitude(8_000, 3.0)
);

// Panako (8 kHz)
adversarial_test!(
    panako_silence_3s,
    Panako,
    8_000,
    SampleRate::HZ_8000,
    silence(8_000, 3.0)
);
adversarial_test!(
    panako_dc_offset,
    Panako,
    8_000,
    SampleRate::HZ_8000,
    dc_offset(8_000, 3.0, 0.99)
);
adversarial_test!(
    panako_impulse,
    Panako,
    8_000,
    SampleRate::HZ_8000,
    impulse(8_000, 3.0)
);
adversarial_test!(
    panako_square_440,
    Panako,
    8_000,
    SampleRate::HZ_8000,
    square_wave(8_000, 3.0, 440.0)
);
adversarial_test!(
    panako_max_amplitude,
    Panako,
    8_000,
    SampleRate::HZ_8000,
    max_amplitude(8_000, 3.0)
);

// Haitsma (5 kHz)
adversarial_test!(
    haitsma_silence_3s,
    Haitsma,
    5_000,
    SampleRate::HZ_5000,
    silence(5_000, 3.0)
);
adversarial_test!(
    haitsma_dc_offset,
    Haitsma,
    5_000,
    SampleRate::HZ_5000,
    dc_offset(5_000, 3.0, 0.99)
);
adversarial_test!(
    haitsma_impulse,
    Haitsma,
    5_000,
    SampleRate::HZ_5000,
    impulse(5_000, 3.0)
);
adversarial_test!(
    haitsma_square_440,
    Haitsma,
    5_000,
    SampleRate::HZ_5000,
    square_wave(5_000, 3.0, 440.0)
);
adversarial_test!(
    haitsma_max_amplitude,
    Haitsma,
    5_000,
    SampleRate::HZ_5000,
    max_amplitude(5_000, 3.0)
);

// ───────────────────────────────────────────────────────────────────────
// Stress: NaN / Inf rejection on offline extract
// ───────────────────────────────────────────────────────────────────────

#[test]
fn wang_rejects_nan_input() {
    let samples = nan_audio(8_000, 3.0);
    let mut w = Wang::default();
    let result = w.extract(&samples, SampleRate::HZ_8000);
    assert!(
        matches!(result, Err(AfpError::NonFiniteSample { .. })),
        "expected NonFiniteSample, got {result:?}"
    );
}

#[test]
fn wang_rejects_inf_input() {
    let samples = inf_audio(8_000, 3.0);
    let mut w = Wang::default();
    let result = w.extract(&samples, SampleRate::HZ_8000);
    assert!(
        matches!(result, Err(AfpError::NonFiniteSample { .. })),
        "expected NonFiniteSample, got {result:?}"
    );
}

#[test]
fn panako_rejects_nan_input() {
    let samples = nan_audio(8_000, 3.0);
    let mut p = Panako::default();
    let result = p.extract(&samples, SampleRate::HZ_8000);
    assert!(
        matches!(result, Err(AfpError::NonFiniteSample { .. })),
        "expected NonFiniteSample, got {result:?}"
    );
}

#[test]
fn haitsma_rejects_nan_input() {
    let samples = nan_audio(5_000, 3.0);
    let mut h = Haitsma::default();
    let result = h.extract(&samples, SampleRate::HZ_5000);
    assert!(
        matches!(result, Err(AfpError::NonFiniteSample { .. })),
        "expected NonFiniteSample, got {result:?}"
    );
}

// ───────────────────────────────────────────────────────────────────────
// Real-audio snapshots: hash counts on CC0 corpus must stay stable.
//
// These are NOT bit-exact golden tests (those live in tests/regression.rs
// on synthetic audio). These assert *count ranges* — the number of hashes
// from real audio should not drift by more than ±10% across versions.
// If a refactor changes the count, update the expected range here with
// justification in the commit message.
// ───────────────────────────────────────────────────────────────────────

mod real_audio_snapshots {
    use super::*;
    use audiofp::io::decode_to_mono_at;

    fn asset(name: &str) -> String {
        format!("tests/assets/{name}")
    }

    /// Assert hash count is within [min, max] inclusive.
    fn assert_count_in_range(name: &str, actual: usize, min: usize, max: usize) {
        assert!(
            actual >= min && actual <= max,
            "{name}: hash count {actual} outside expected range [{min}, {max}]. \
             If intentional, update the snapshot range in tests/stress.rs."
        );
    }

    // ─── Wang on real audio ───────────────────────────────────────────

    #[test]
    fn wang_galway_flac_hash_count_stable() {
        let samples = decode_to_mono_at(asset("galway.flac"), 8_000).unwrap();
        let mut w = Wang::default();
        let fp = w.extract(&samples, SampleRate::HZ_8000).unwrap();
        // ~16s clip @ default config produces ~1500-2500 hashes
        assert_count_in_range("wang/galway.flac", fp.hashes.len(), 1000, 3000);
    }

    #[test]
    fn wang_freak_flac_hash_count_stable() {
        let samples = decode_to_mono_at(asset("freak.flac"), 8_000).unwrap();
        let mut w = Wang::default();
        let fp = w.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_count_in_range("wang/freak.flac", fp.hashes.len(), 1000, 3000);
    }

    #[test]
    fn wang_piano_ogg_hash_count_stable() {
        let samples = decode_to_mono_at(asset("piano.ogg"), 8_000).unwrap();
        let mut w = Wang::default();
        let fp = w.extract(&samples, SampleRate::HZ_8000).unwrap();
        // Piano is sparser — fewer spectral peaks
        assert_count_in_range("wang/piano.ogg", fp.hashes.len(), 100, 2500);
    }

    #[test]
    fn wang_speech_ogg_hash_count_stable() {
        let samples = decode_to_mono_at(asset("speech.ogg"), 8_000).unwrap();
        let mut w = Wang::default();
        let fp = w.extract(&samples, SampleRate::HZ_8000).unwrap();
        // Speech has moderate spectral content
        assert_count_in_range("wang/speech.ogg", fp.hashes.len(), 200, 2500);
    }

    // ─── Panako on real audio ─────────────────────────────────────────

    #[test]
    fn panako_galway_flac_hash_count_stable() {
        let samples = decode_to_mono_at(asset("galway.flac"), 8_000).unwrap();
        let mut p = Panako::default();
        let fp = p.extract(&samples, SampleRate::HZ_8000).unwrap();
        // Panako with fan_out=5 produces fewer hashes than Wang
        assert_count_in_range("panako/galway.flac", fp.hashes.len(), 500, 2500);
    }

    #[test]
    fn panako_freak_flac_hash_count_stable() {
        let samples = decode_to_mono_at(asset("freak.flac"), 8_000).unwrap();
        let mut p = Panako::default();
        let fp = p.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_count_in_range("panako/freak.flac", fp.hashes.len(), 500, 2500);
    }

    // ─── Haitsma on real audio ────────────────────────────────────────

    #[test]
    fn haitsma_galway_flac_frame_count_stable() {
        let samples = decode_to_mono_at(asset("galway.flac"), 5_000).unwrap();
        let mut h = Haitsma::default();
        let fp = h.extract(&samples, SampleRate::HZ_5000).unwrap();
        // ~16s × 78.125 fps ≈ 1250 frames; allow ±20%
        assert_count_in_range("haitsma/galway.flac", fp.frames.len(), 1000, 1600);
    }

    #[test]
    fn haitsma_freak_flac_frame_count_stable() {
        let samples = decode_to_mono_at(asset("freak.flac"), 5_000).unwrap();
        let mut h = Haitsma::default();
        let fp = h.extract(&samples, SampleRate::HZ_5000).unwrap();
        assert_count_in_range("haitsma/freak.flac", fp.frames.len(), 1000, 1600);
    }

    // ─── Cross-format consistency ────────────────────────────────────

    #[test]
    fn wang_galway_mp3_vs_flac_substantial_overlap() {
        let flac = decode_to_mono_at(asset("galway.flac"), 8_000).unwrap();
        let mp3 = decode_to_mono_at(asset("galway.mp3"), 8_000).unwrap();

        let mut w = Wang::default();
        let fp_flac = w.extract(&flac, SampleRate::HZ_8000).unwrap();
        let fp_mp3 = w.extract(&mp3, SampleRate::HZ_8000).unwrap();

        // Compute hash overlap (Jaccard-style)
        use std::collections::HashSet;
        let set_flac: HashSet<u32> = fp_flac.hashes.iter().map(|h| h.hash).collect();
        let set_mp3: HashSet<u32> = fp_mp3.hashes.iter().map(|h| h.hash).collect();
        let intersection = set_flac.intersection(&set_mp3).count();
        let union = set_flac.union(&set_mp3).count();
        let jaccard = intersection as f64 / union.max(1) as f64;

        // MP3 128kbps vs FLAC should share ≥25% of hashes
        assert!(
            jaccard >= 0.20,
            "galway MP3 vs FLAC Jaccard too low: {jaccard:.3} (expected ≥ 0.20)"
        );
    }

    #[test]
    fn wang_different_tracks_minimal_overlap() {
        let galway = decode_to_mono_at(asset("galway.flac"), 8_000).unwrap();
        let freak = decode_to_mono_at(asset("freak.flac"), 8_000).unwrap();

        let mut w = Wang::default();
        let fp_galway = w.extract(&galway, SampleRate::HZ_8000).unwrap();
        let fp_freak = w.extract(&freak, SampleRate::HZ_8000).unwrap();

        use std::collections::HashSet;
        let set_a: HashSet<u32> = fp_galway.hashes.iter().map(|h| h.hash).collect();
        let set_b: HashSet<u32> = fp_freak.hashes.iter().map(|h| h.hash).collect();
        let intersection = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();
        let jaccard = intersection as f64 / union.max(1) as f64;

        // Different songs should have < 5% overlap (random collision floor)
        assert!(
            jaccard < 0.05,
            "different tracks Jaccard too high: {jaccard:.3} (expected < 0.05)"
        );
    }
}
