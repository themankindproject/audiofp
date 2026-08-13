//! Threshold calibration on the real CC0 corpus (issue #104).
//!
//! Runs the three matchers over the real-audio catalog and prints the
//! score/prominence/BER distributions for same-track cross-codec pairs
//! (should match) and cross-track pairs (should not match). The output is
//! the data behind the "Recommended thresholds" section of
//! `ROBUSTNESS.md`.
//!
//! Run with:
//!
//! ```bash
//! cargo test --test threshold_calibration -- --ignored --nocapture
//! ```
//!
//! The test itself only pins separation: every positive pair must clear
//! the defaults and every negative pair must fail them. The printed sweep
//! shows how much headroom the defaults have before FP/FN appear.

#![cfg(all(
    feature = "std-wav",
    feature = "std-mp3",
    feature = "std-flac",
    feature = "std-ogg"
))]

use audiofp::classical::{
    Haitsma, HaitsmaFingerprint, Panako, PanakoFingerprint, Wang, WangFingerprint,
};
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangMatchConfig,
    WangMatcher,
};
use audiofp::{Fingerprinter, SampleRate};

/// One entry: (logical track id, file path). Mirrors
/// `tests/matching_real_audio.rs` so both suites exercise the same corpus.
const CATALOG: &[(usize, &str)] = &[
    (0, "tests/assets/galway.wav"),
    (0, "tests/assets/galway.mp3"),
    (0, "tests/assets/galway.flac"),
    (0, "tests/assets/galway.ogg"),
    (1, "tests/assets/freak.wav"),
    (1, "tests/assets/freak.mp3"),
    (1, "tests/assets/freak.flac"),
    (1, "tests/assets/freak.ogg"),
    (2, "tests/assets/piano.ogg"),
    (3, "tests/assets/speech.ogg"),
    (4, "tests/assets/catalog/bach_goldberg_aria.ogg"),
    (5, "tests/assets/catalog/bach_goldberg_var4.ogg"),
    (6, "tests/assets/catalog/beethoven_coriolan.ogg"),
    (7, "tests/assets/catalog/beethoven_egmont.ogg"),
    (8, "tests/assets/catalog/beethoven_eroica_mvt1.ogg"),
    (9, "tests/assets/catalog/dvorak_american_mvt1.ogg"),
    (10, "tests/assets/catalog/grieg_morning.ogg"),
];

const NUM_TRACKS: usize = 11;

/// (score, prominence) per pair; positives (same-track) and negatives
/// (cross-track) separately.
type PairScores = (Vec<(f32, f32)>, Vec<(f32, f32)>);

fn load_wang_catalog() -> Vec<(usize, WangFingerprint)> {
    let mut wang = Wang::default();
    CATALOG
        .iter()
        .map(|&(track, path)| {
            let samples =
                decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let fp = wang
                .extract(&samples, SampleRate::HZ_8000)
                .unwrap_or_else(|e| panic!("extract {path}: {e}"));
            (track, fp)
        })
        .collect()
}

fn load_haitsma_catalog() -> Vec<(usize, HaitsmaFingerprint)> {
    let mut haitsma = Haitsma::default();
    CATALOG
        .iter()
        .map(|&(track, path)| {
            let samples =
                decode_to_mono_at(path, 5_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let fp = haitsma
                .extract(&samples, SampleRate::HZ_5000)
                .unwrap_or_else(|e| panic!("extract {path}: {e}"));
            (track, fp)
        })
        .collect()
}

fn load_panako_catalog() -> Vec<(usize, PanakoFingerprint)> {
    let mut panako = Panako::default();
    CATALOG
        .iter()
        .map(|&(track, path)| {
            let samples =
                decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let fp = panako
                .extract(&samples, SampleRate::HZ_8000)
                .unwrap_or_else(|e| panic!("extract {path}: {e}"));
            (track, fp)
        })
        .collect()
}

/// (score, prominence) for every evaluated pair.
fn wang_pair_scores() -> PairScores {
    let catalog = load_wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig {
        // Accept everything so the raw score/prominence is observable.
        min_votes: 1,
        min_score: 0.0,
        min_prominence: 0.0,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                positives.push((r.score, r.prominence));
            }
        }
    }

    let mut negatives = Vec::new();
    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            negatives.push((r.score, r.prominence));
        }
    }

    (positives, negatives)
}

/// (score, prominence) for every evaluated pair.
fn haitsma_pair_scores() -> PairScores {
    let catalog = load_haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        max_ber: 1.0,
        min_overlap_frames: 1,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                positives.push((r.score, r.prominence));
            }
        }
    }

    let mut negatives = Vec::new();
    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            negatives.push((r.score, r.prominence));
        }
    }

    (positives, negatives)
}

/// (score, prominence) for every evaluated pair.
fn panako_pair_scores() -> PairScores {
    let catalog = load_panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig {
        min_votes: 1,
        min_score: 0.0,
        min_prominence: 0.0,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                positives.push((r.score, r.prominence));
            }
        }
    }

    let mut negatives = Vec::new();
    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            negatives.push((r.score, r.prominence));
        }
    }

    (positives, negatives)
}

/// Print a distribution summary and the largest threshold that still
/// separates positives from negatives with zero FP/FN.
fn report(name: &str, positives: &[(f32, f32)], negatives: &[(f32, f32)]) {
    let min_pos_score = positives.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
    let min_pos_prom = positives.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
    let max_neg_score = negatives.iter().map(|p| p.0).fold(0.0_f32, f32::max);
    let max_neg_prom = negatives.iter().map(|p| p.1).fold(0.0_f32, f32::max);

    println!("── {name} ──────────────────────────────────────────");
    println!(
        "  positives: n={}  score [{min_pos_score:.3}, 1.00]  prom [{min_pos_prom:.1}, …]",
        positives.len()
    );
    println!(
        "  negatives: n={}  score [0.00, {max_neg_score:.3}]  prom [0.0, {max_neg_prom:.1}]",
        negatives.len()
    );
    // Separation margin at the current defaults: largest config value that
    // still gives zero FP and zero FN simultaneously.
    println!(
        "  separation: score margin {:.3} (min_pos − max_neg), prom margin {:.1}",
        min_pos_score - max_neg_score,
        min_pos_prom - max_neg_prom,
    );
}

#[test]
#[ignore = "runs the full real-audio decode + sweep; prints calibration table"]
fn print_threshold_calibration_table() {
    let (wang_pos, wang_neg) = wang_pair_scores();
    report("Wang", &wang_pos, &wang_neg);

    let (haitsma_pos, haitsma_neg) = haitsma_pair_scores();
    report("Haitsma", &haitsma_pos, &haitsma_neg);

    let (panako_pos, panako_neg) = panako_pair_scores();
    report("Panako", &panako_pos, &panako_neg);
}

#[test]
fn wang_defaults_separate_all_pairs() {
    let catalog = load_wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig::default());

    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(r.is_match, "Wang: same-track pair must match: {r:?}");
            }
        }
    }

    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            assert!(
                !r.is_match,
                "Wang: cross-track pair must not match: {} vs {} → {r:?}",
                reps[i].0, reps[j].0,
            );
        }
    }
}

#[test]
fn haitsma_defaults_separate_all_pairs() {
    let catalog = load_haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());

    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(r.is_match, "Haitsma: same-track pair must match: {r:?}");
            }
        }
    }

    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            assert!(
                !r.is_match,
                "Haitsma: cross-track pair must not match: {} vs {} → {r:?}",
                reps[i].0, reps[j].0,
            );
        }
    }
}

#[test]
fn panako_defaults_separate_all_pairs() {
    let catalog = load_panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    for track in [0usize, 1] {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let r = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(r.is_match, "Panako: same-track pair must match: {r:?}");
            }
        }
    }

    let reps: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();
    for i in 0..reps.len() {
        for j in (i + 1)..reps.len() {
            let r = matcher.match_one(&reps[i].1, &reps[j].1);
            assert!(
                !r.is_match,
                "Panako: cross-track pair must not match: {} vs {} → {r:?}",
                reps[i].0, reps[j].0,
            );
        }
    }
}
