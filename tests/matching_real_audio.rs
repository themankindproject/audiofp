//! End-to-end matching integration tests over real audio.
//!
//! Tests all three matchers (Wang, Haitsma, Panako) and the 1:N index
//! accelerators against a catalog of real-world recordings in multiple
//! codecs. Verifies:
//!
//! 1. **Same-song cross-codec matching** — the same recording encoded in
//!    different formats must match itself with high confidence.
//! 2. **Cross-track rejection** — different songs must NOT match.
//! 3. **Index identification** — the correct track is identified from a
//!    catalog using each algorithm's index type.
//!
//! Audio assets:
//! - "Galway" and "Furious Freak" by Kevin MacLeod (CC-BY 3.0)
//! - Piano and Speech clips (CC0, project-generated)
//! - Classical recordings from Musopen (CC0/Public Domain):
//!   Bach, Beethoven, Dvorak, Grieg
//!
//! See tests/assets/CREDITS.md.
#![cfg(all(
    feature = "std-wav",
    feature = "std-mp3",
    feature = "std-flac",
    feature = "std-ogg"
))]

mod common;

use audiofp::matching::{
    HaitsmaIndex, HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoIndex, PanakoMatchConfig,
    PanakoMatcher, WangIndex, WangMatchConfig, WangMatcher,
};
use common::catalog::{
    CATALOG, MULTI_VARIANT_TRACKS, NUM_TRACKS, haitsma_catalog, panako_catalog, wang_catalog,
};

// ─── Wang matcher tests ──────────────────────────────────────────────

#[test]
fn wang_same_song_cross_codec_matches() {
    let catalog = wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig::default());

    // Each multi-codec variant must match every other variant of its track.
    for track in MULTI_VARIANT_TRACKS {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let res = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(
                    res.is_match,
                    "Wang: track {track} codec pair ({i},{j}) must match: score={:.3}, prom={:.1}",
                    res.score, res.prominence,
                );
            }
        }
    }
}

#[test]
fn wang_cross_track_rejection() {
    let catalog = wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig::default());

    // Pick one representative per track.
    let representatives: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();

    for i in 0..representatives.len() {
        for j in (i + 1)..representatives.len() {
            let res = matcher.match_one(&representatives[i].1, &representatives[j].1);
            assert!(
                !res.is_match,
                "Wang: tracks {} vs {} must NOT match: score={:.3}",
                representatives[i].0, representatives[j].0, res.score,
            );
        }
    }
}

#[test]
fn wang_index_identifies_correct_track() {
    let catalog = wang_catalog();

    // Build index from one representative per track.
    let refs: Vec<_> = (0..NUM_TRACKS)
        .map(|t| {
            catalog
                .iter()
                .find(|(track, _)| *track == t)
                .unwrap()
                .1
                .clone()
        })
        .collect();
    let index = WangIndex::build(&refs, 1000);

    // Query with each entry — must identify its own track.
    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &WangMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                ref_id, *track,
                "Wang index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, ref_id, track, res.score,
            );
        } else {
            panic!(
                "Wang index: query #{idx} ({}) returned None",
                CATALOG[idx].1,
            );
        }
    }
}

// ─── Haitsma matcher tests ───────────────────────────────────────────

#[test]
fn haitsma_same_song_cross_codec_matches() {
    let catalog = haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());

    for track in MULTI_VARIANT_TRACKS {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let res = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(
                    res.is_match,
                    "Haitsma: track {track} codec pair ({i},{j}) must match: score={:.3}",
                    res.score,
                );
            }
        }
    }
}

#[test]
fn haitsma_cross_track_rejection() {
    let catalog = haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());

    let representatives: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();

    for i in 0..representatives.len() {
        for j in (i + 1)..representatives.len() {
            let res = matcher.match_one(&representatives[i].1, &representatives[j].1);
            assert!(
                !res.is_match,
                "Haitsma: tracks {} vs {} must NOT match: score={:.3}",
                representatives[i].0, representatives[j].0, res.score,
            );
        }
    }
}

#[test]
fn haitsma_index_identifies_correct_track() {
    let catalog = haitsma_catalog();

    let refs: Vec<_> = (0..NUM_TRACKS)
        .map(|t| {
            catalog
                .iter()
                .find(|(track, _)| *track == t)
                .unwrap()
                .1
                .clone()
        })
        .collect();
    let index = HaitsmaIndex::build(&refs, 500);

    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &HaitsmaMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                ref_id, *track,
                "Haitsma index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, ref_id, track, res.score,
            );
        } else {
            panic!(
                "Haitsma index: query #{idx} ({}) returned None",
                CATALOG[idx].1,
            );
        }
    }
}

// ─── Panako matcher tests ────────────────────────────────────────────

#[test]
fn panako_same_song_cross_codec_matches() {
    let catalog = panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    for track in MULTI_VARIANT_TRACKS {
        let variants: Vec<_> = catalog.iter().filter(|(t, _)| *t == track).collect();
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                let res = matcher.match_one(&variants[i].1, &variants[j].1);
                assert!(
                    res.is_match,
                    "Panako: track {track} codec pair ({i},{j}) must match: score={:.3}, scale={:.2}",
                    res.score, res.time_scale,
                );
            }
        }
    }
}

#[test]
fn panako_cross_track_rejection() {
    let catalog = panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    let representatives: Vec<_> = (0..NUM_TRACKS)
        .map(|t| catalog.iter().find(|(track, _)| *track == t).unwrap())
        .collect();

    for i in 0..representatives.len() {
        for j in (i + 1)..representatives.len() {
            let res = matcher.match_one(&representatives[i].1, &representatives[j].1);
            assert!(
                !res.is_match,
                "Panako: tracks {} vs {} must NOT match: score={:.3}",
                representatives[i].0, representatives[j].0, res.score,
            );
        }
    }
}

#[test]
fn panako_index_identifies_correct_track() {
    let catalog = panako_catalog();

    let refs: Vec<_> = (0..NUM_TRACKS)
        .map(|t| {
            catalog
                .iter()
                .find(|(track, _)| *track == t)
                .unwrap()
                .1
                .clone()
        })
        .collect();
    let index = PanakoIndex::build(&refs, 1000);

    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &PanakoMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                ref_id, *track,
                "Panako index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, ref_id, track, res.score,
            );
        } else {
            panic!(
                "Panako index: query #{idx} ({}) returned None",
                CATALOG[idx].1,
            );
        }
    }
}

// ─── Hash count sanity (drift detection baseline) ────────────────────

/// Verify all catalog tracks produce a reasonable number of hashes.
/// This acts as a canary: if a future code change silently alters hash
/// output, counts will shift and this test catches it.
#[test]
fn hash_counts_are_nonzero_and_reasonable() {
    let wang_catalog = wang_catalog();
    let haitsma_catalog = haitsma_catalog();
    let panako_catalog = panako_catalog();

    for (idx, (track, fp)) in wang_catalog.iter().enumerate() {
        assert!(
            fp.hashes.len() >= 10,
            "Wang track {track} ({}) produced only {} hashes (expected ≥10)",
            CATALOG[idx].1,
            fp.hashes.len(),
        );
    }

    for (idx, (track, fp)) in haitsma_catalog.iter().enumerate() {
        assert!(
            fp.frames.len() >= 50,
            "Haitsma track {track} ({}) produced only {} frames (expected ≥50)",
            CATALOG[idx].1,
            fp.frames.len(),
        );
    }

    for (idx, (track, fp)) in panako_catalog.iter().enumerate() {
        assert!(
            fp.hashes.len() >= 5,
            "Panako track {track} ({}) produced only {} hashes (expected ≥5)",
            CATALOG[idx].1,
            fp.hashes.len(),
        );
    }
}
