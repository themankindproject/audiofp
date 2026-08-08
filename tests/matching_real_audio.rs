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
//!    small catalog using each algorithm's index type.
//!
//! Audio assets: "Galway" and "Furious Freak" by Kevin MacLeod (CC-BY 3.0),
//! plus CC0 piano and speech clips. See tests/assets/CREDITS.md.
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
    HaitsmaIndex, HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoIndex, PanakoMatchConfig,
    PanakoMatcher, WangIndex, WangMatchConfig, WangMatcher,
};
use audiofp::{Fingerprinter, SampleRate};

// ─── Catalog of real audio files ─────────────────────────────────────

/// Each entry: (logical track id, file path).
/// Track 0 = Galway variants, Track 1 = Freak variants,
/// Track 2 = Piano, Track 3 = Speech.
const CATALOG: &[(usize, &str)] = &[
    // Track 0: Galway in multiple codecs
    (0, "tests/assets/galway.wav"),
    (0, "tests/assets/galway.mp3"),
    (0, "tests/assets/galway.flac"),
    (0, "tests/assets/galway.ogg"),
    // Track 1: Freak in multiple codecs
    (1, "tests/assets/freak.wav"),
    (1, "tests/assets/freak.mp3"),
    (1, "tests/assets/freak.flac"),
    (1, "tests/assets/freak.ogg"),
    // Track 2: Piano
    (2, "tests/assets/piano.ogg"),
    // Track 3: Speech
    (3, "tests/assets/speech.ogg"),
];

fn load_wang_catalog() -> Vec<(usize, WangFingerprint)> {
    let mut wang = Wang::default();
    CATALOG
        .iter()
        .map(|&(track, path)| {
            let samples = decode_to_mono_at(path, 8_000)
                .unwrap_or_else(|e| panic!("decode {path}: {e}"));
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
            let samples = decode_to_mono_at(path, 5_000)
                .unwrap_or_else(|e| panic!("decode {path}: {e}"));
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
            let samples = decode_to_mono_at(path, 8_000)
                .unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let fp = panako
                .extract(&samples, SampleRate::HZ_8000)
                .unwrap_or_else(|e| panic!("extract {path}: {e}"));
            (track, fp)
        })
        .collect()
}

// ─── Wang matcher tests ──────────────────────────────────────────────

#[test]
fn wang_same_song_cross_codec_matches() {
    let catalog = load_wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig::default());

    // Each Galway variant must match every other Galway variant.
    let galway: Vec<_> = catalog.iter().filter(|(t, _)| *t == 0).collect();
    for i in 0..galway.len() {
        for j in (i + 1)..galway.len() {
            let res = matcher.match_one(&galway[i].1, &galway[j].1);
            assert!(
                res.is_match,
                "Wang: Galway codec pair ({i},{j}) must match: score={:.3}, prom={:.1}",
                res.score, res.prominence,
            );
        }
    }

    // Each Freak variant must match every other Freak variant.
    let freak: Vec<_> = catalog.iter().filter(|(t, _)| *t == 1).collect();
    for i in 0..freak.len() {
        for j in (i + 1)..freak.len() {
            let res = matcher.match_one(&freak[i].1, &freak[j].1);
            assert!(
                res.is_match,
                "Wang: Freak codec pair ({i},{j}) must match: score={:.3}, prom={:.1}",
                res.score, res.prominence,
            );
        }
    }
}

#[test]
fn wang_cross_track_rejection() {
    let catalog = load_wang_catalog();
    let matcher = WangMatcher::new(WangMatchConfig::default());

    // Pick one representative per track.
    let representatives: Vec<_> = [0usize, 1, 2, 3]
        .iter()
        .map(|&t| catalog.iter().find(|(track, _)| *track == t).unwrap())
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
    let catalog = load_wang_catalog();

    // Build index from all fingerprints.
    let fps: Vec<_> = catalog.iter().map(|(_, fp)| fp.clone()).collect();
    let index = WangIndex::build(&fps, 1000);

    // Query with each fingerprint — must identify its own track.
    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &WangMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                catalog[ref_id].0, *track,
                "Wang index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, catalog[ref_id].0, track, res.score,
            );
        } else {
            // Self-match must always return something.
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
    let catalog = load_haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());

    let galway: Vec<_> = catalog.iter().filter(|(t, _)| *t == 0).collect();
    for i in 0..galway.len() {
        for j in (i + 1)..galway.len() {
            let res = matcher.match_one(&galway[i].1, &galway[j].1);
            assert!(
                res.is_match,
                "Haitsma: Galway codec pair ({i},{j}) must match: score={:.3}",
                res.score,
            );
        }
    }

    let freak: Vec<_> = catalog.iter().filter(|(t, _)| *t == 1).collect();
    for i in 0..freak.len() {
        for j in (i + 1)..freak.len() {
            let res = matcher.match_one(&freak[i].1, &freak[j].1);
            assert!(
                res.is_match,
                "Haitsma: Freak codec pair ({i},{j}) must match: score={:.3}",
                res.score,
            );
        }
    }
}

#[test]
fn haitsma_cross_track_rejection() {
    let catalog = load_haitsma_catalog();
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());

    let representatives: Vec<_> = [0usize, 1, 2, 3]
        .iter()
        .map(|&t| catalog.iter().find(|(track, _)| *track == t).unwrap())
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
    let catalog = load_haitsma_catalog();

    let fps: Vec<_> = catalog.iter().map(|(_, fp)| fp.clone()).collect();
    let index = HaitsmaIndex::build(&fps, 500);

    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &HaitsmaMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                catalog[ref_id].0, *track,
                "Haitsma index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, catalog[ref_id].0, track, res.score,
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
    let catalog = load_panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    let galway: Vec<_> = catalog.iter().filter(|(t, _)| *t == 0).collect();
    for i in 0..galway.len() {
        for j in (i + 1)..galway.len() {
            let res = matcher.match_one(&galway[i].1, &galway[j].1);
            assert!(
                res.is_match,
                "Panako: Galway codec pair ({i},{j}) must match: score={:.3}, scale={:.2}",
                res.score, res.time_scale,
            );
        }
    }

    let freak: Vec<_> = catalog.iter().filter(|(t, _)| *t == 1).collect();
    for i in 0..freak.len() {
        for j in (i + 1)..freak.len() {
            let res = matcher.match_one(&freak[i].1, &freak[j].1);
            assert!(
                res.is_match,
                "Panako: Freak codec pair ({i},{j}) must match: score={:.3}, scale={:.2}",
                res.score, res.time_scale,
            );
        }
    }
}

#[test]
fn panako_cross_track_rejection() {
    let catalog = load_panako_catalog();
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    let representatives: Vec<_> = [0usize, 1, 2, 3]
        .iter()
        .map(|&t| catalog.iter().find(|(track, _)| *track == t).unwrap())
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
    let catalog = load_panako_catalog();

    let fps: Vec<_> = catalog.iter().map(|(_, fp)| fp.clone()).collect();
    let index = PanakoIndex::build(&fps, 1000);

    for (idx, (track, fp)) in catalog.iter().enumerate() {
        let result = index.query(fp, &PanakoMatchConfig::default());
        if let Some((ref_id, res)) = result {
            assert_eq!(
                catalog[ref_id].0, *track,
                "Panako index: query #{idx} ({}) identified as track {} (expected {}), score={:.3}",
                CATALOG[idx].1, catalog[ref_id].0, track, res.score,
            );
        } else {
            panic!(
                "Panako index: query #{idx} ({}) returned None",
                CATALOG[idx].1,
            );
        }
    }
}
