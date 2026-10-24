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
//! The default-config separation asserts (every positive pair clears the
//! defaults, every negative pair fails them) live in
//! `tests/matching_real_audio.rs`; this file only prints the sweep that
//! shows how much headroom the defaults have before FP/FN appear.

#![cfg(all(
    feature = "std-wav",
    feature = "std-mp3",
    feature = "std-flac",
    feature = "std-ogg"
))]

mod common;

use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangMatchConfig,
    WangMatcher,
};
use common::catalog::{
    MULTI_VARIANT_TRACKS, NUM_TRACKS, haitsma_catalog, panako_catalog, wang_catalog,
};

/// (score, prominence) per pair; positives (same-track) and negatives
/// (cross-track) separately.
type PairScores = (Vec<(f32, f32)>, Vec<(f32, f32)>);

/// (score, prominence) for every evaluated pair.
fn wang_pair_scores() -> PairScores {
    let catalog = wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig {
        // Accept everything so the raw score/prominence is observable.
        min_votes: 1,
        min_score: 0.0,
        min_prominence: 0.0,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in MULTI_VARIANT_TRACKS {
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
    let catalog = haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        max_ber: 1.0,
        min_overlap_frames: 1,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in MULTI_VARIANT_TRACKS {
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
    let catalog = panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig {
        min_votes: 1,
        min_score: 0.0,
        min_prominence: 0.0,
        ..Default::default()
    });

    let mut positives = Vec::new();
    for track in MULTI_VARIANT_TRACKS {
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
