//! Degradation robustness on real audio (expanded corpus).
//!
//! The `*_noisy15.wav` fixtures (15 dB SNR white-noise overlays,
//! deterministic seed — see `tests/assets/CREDITS.md`) must still match
//! their clean masters on all three classical matchers: this is a real
//! robustness gate, not a characterisation.
//!
//! The `*_fast2.wav` (+2% speed) and `*_pitch1.wav` (+1 semitone)
//! fixtures are pinned as *informational non-matches*: landmark and
//! frame-hash methods are not tempo/pitch invariant, so they correctly
//! reject today. If a future change makes them match, these tests fail
//! loudly — update them deliberately, not silently.
//!
//! Fixtures cover two genres (contemporary `galway`, jazz-funk
//! `acidjazz`) so the gates are not tuned to one recording.
#![cfg(all(
    feature = "std-wav",
    feature = "std-mp3",
    feature = "std-flac",
    feature = "std-ogg"
))]

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangMatchConfig,
    WangMatcher,
};
use audiofp::{Fingerprinter, SampleRate};

/// (label, clean master, noisy variant).
const NOISE_PAIRS: &[(&str, &str, &str)] = &[
    (
        "galway",
        "tests/assets/galway.wav",
        "tests/assets/galway_noisy15.wav",
    ),
    (
        "acidjazz",
        "tests/assets/acidjazz.wav",
        "tests/assets/acidjazz_noisy15.wav",
    ),
];

/// (label, clean master, speed/pitch variant): expected non-match.
const INVARIANCE_PAIRS: &[(&str, &str, &str)] = &[
    (
        "galway/fast2",
        "tests/assets/galway.wav",
        "tests/assets/galway_fast2.wav",
    ),
    (
        "galway/pitch1",
        "tests/assets/galway.wav",
        "tests/assets/galway_pitch1.wav",
    ),
    (
        "acidjazz/fast2",
        "tests/assets/acidjazz.wav",
        "tests/assets/acidjazz_fast2.wav",
    ),
    (
        "acidjazz/pitch1",
        "tests/assets/acidjazz.wav",
        "tests/assets/acidjazz_pitch1.wav",
    ),
];

fn wang_fp(path: &str) -> audiofp::classical::WangFingerprint {
    let samples = decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    Wang::default()
        .extract(&samples, SampleRate::HZ_8000)
        .unwrap_or_else(|e| panic!("extract {path}: {e}"))
}

fn panako_fp(path: &str) -> audiofp::classical::PanakoFingerprint {
    let samples = decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    Panako::default()
        .extract(&samples, SampleRate::HZ_8000)
        .unwrap_or_else(|e| panic!("extract {path}: {e}"))
}

fn haitsma_fp(path: &str) -> audiofp::classical::HaitsmaFingerprint {
    let samples = decode_to_mono_at(path, 5_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
    Haitsma::default()
        .extract(&samples, SampleRate::HZ_5000)
        .unwrap_or_else(|e| panic!("extract {path}: {e}"))
}

#[test]
fn wang_matches_15db_noise_on_real_audio() {
    let matcher = WangMatcher::new(WangMatchConfig::default());
    for (label, clean, noisy) in NOISE_PAIRS {
        let r = matcher.match_one(&wang_fp(clean), &wang_fp(noisy));
        assert!(
            r.is_match,
            "Wang: {label} @15dB SNR must match: score={:.3}, prom={:.1}",
            r.score, r.prominence,
        );
    }
}

#[test]
fn panako_matches_15db_noise_on_real_audio() {
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());
    for (label, clean, noisy) in NOISE_PAIRS {
        let r = matcher.match_one(&panako_fp(clean), &panako_fp(noisy));
        assert!(
            r.is_match,
            "Panako: {label} @15dB SNR must match: score={:.3}",
            r.score,
        );
    }
}

#[test]
fn haitsma_matches_15db_noise_on_real_audio() {
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
    for (label, clean, noisy) in NOISE_PAIRS {
        let r = matcher.match_one(&haitsma_fp(clean), &haitsma_fp(noisy));
        assert!(
            r.is_match,
            "Haitsma: {label} @15dB SNR must match: score={:.3}",
            r.score,
        );
    }
}

// ─── Informational invariance pins ────────────────────────────────────
// These assert TODAY's behaviour (rejection). A future tempo/pitch
// invariant matcher must update them deliberately.

#[test]
fn wang_rejects_2pct_speed_today() {
    let matcher = WangMatcher::new(WangMatchConfig::default());
    for (label, clean, degraded) in INVARIANCE_PAIRS {
        let r = matcher.match_one(&wang_fp(clean), &wang_fp(degraded));
        assert!(
            !r.is_match,
            "Wang: {label} informational pin tripped (now matches, score={:.3}): \
             tempo/pitch invariance changed — update this test deliberately",
            r.score,
        );
    }
}

#[test]
fn panako_rejects_2pct_speed_today() {
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());
    for (label, clean, degraded) in INVARIANCE_PAIRS {
        let r = matcher.match_one(&panako_fp(clean), &panako_fp(degraded));
        assert!(
            !r.is_match,
            "Panako: {label} informational pin tripped (now matches, score={:.3}): \
             tempo/pitch invariance changed — update this test deliberately",
            r.score,
        );
    }
}

#[test]
fn haitsma_rejects_2pct_speed_today() {
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
    for (label, clean, degraded) in INVARIANCE_PAIRS {
        let r = matcher.match_one(&haitsma_fp(clean), &haitsma_fp(degraded));
        assert!(
            !r.is_match,
            "Haitsma: {label} informational pin tripped (now matches, score={:.3}): \
             tempo/pitch invariance changed — update this test deliberately",
            r.score,
        );
    }
}
