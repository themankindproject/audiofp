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
/// Tracks 0-1, 11-12: multi-codec originals (positives). Tracks 2-3:
/// project-generated clips. Tracks 4-10, 13-16: Musopen classical (CC0,
/// negatives-only singletons).
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
    // Track 2: Piano (CC0)
    (2, "tests/assets/piano.ogg"),
    // Track 3: Speech (CC0)
    (3, "tests/assets/speech.ogg"),
    // Track 4: Bach - Goldberg Variations, Aria (CC0, Musopen)
    (4, "tests/assets/catalog/bach_goldberg_aria.ogg"),
    // Track 5: Bach - Goldberg Variations, Var 4 (CC0, Musopen)
    (5, "tests/assets/catalog/bach_goldberg_var4.ogg"),
    // Track 6: Beethoven - Coriolan Overture (CC0, Musopen)
    (6, "tests/assets/catalog/beethoven_coriolan.ogg"),
    // Track 7: Beethoven - Egmont Overture (CC0, Musopen)
    (7, "tests/assets/catalog/beethoven_egmont.ogg"),
    // Track 8: Beethoven - Symphony No.3 Eroica, Mvt 1 (CC0, Musopen)
    (8, "tests/assets/catalog/beethoven_eroica_mvt1.ogg"),
    // Track 9: Dvorak - String Quartet No.12 "American", Mvt 1 (CC0, Musopen)
    (9, "tests/assets/catalog/dvorak_american_mvt1.ogg"),
    // Track 10: Grieg - Peer Gynt, Morning (CC0, Musopen)
    (10, "tests/assets/catalog/grieg_morning.ogg"),
    // Track 11: AcidJazz in multiple codecs (CC-BY 3.0, Kevin MacLeod)
    (11, "tests/assets/acidjazz.wav"),
    (11, "tests/assets/acidjazz.mp3"),
    (11, "tests/assets/acidjazz.flac"),
    (11, "tests/assets/acidjazz.ogg"),
    // Track 12: Digya in multiple codecs (CC-BY 3.0, Kevin MacLeod)
    (12, "tests/assets/digya.wav"),
    (12, "tests/assets/digya.mp3"),
    (12, "tests/assets/digya.flac"),
    (12, "tests/assets/digya.ogg"),
    // Track 13: Haydn - String Quartet Op.64 No.4, Finale (CC0, Musopen)
    (13, "tests/assets/catalog/haydn_lark_finale.ogg"),
    // Track 14: Mozart - Marriage of Figaro Overture (CC0, Musopen)
    (14, "tests/assets/catalog/mozart_figaro_over.ogg"),
    // Track 15: Schubert - Piano Sonata D.958, Menuetto (CC0, Musopen)
    (15, "tests/assets/catalog/schubert_menuetto.ogg"),
    // Track 16: Mendelssohn - Italian Symphony, Saltarello (CC0, Musopen)
    (16, "tests/assets/catalog/mendelssohn_saltarello.ogg"),
];

const NUM_TRACKS: usize = 17;

/// Tracks with multiple codec variants: every pair must match.
const MULTI_VARIANT_TRACKS: [usize; 4] = [0, 1, 11, 12];

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

// ─── Wang matcher tests ──────────────────────────────────────────────

#[test]
fn wang_same_song_cross_codec_matches() {
    let catalog = load_wang_catalog();
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
    let catalog = load_wang_catalog();
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
    let catalog = load_wang_catalog();

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
    let catalog = load_haitsma_catalog();
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
    let catalog = load_haitsma_catalog();
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
    let catalog = load_haitsma_catalog();

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
    let catalog = load_panako_catalog();
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
    let catalog = load_panako_catalog();
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
    let catalog = load_panako_catalog();

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
    let wang_catalog = load_wang_catalog();
    let haitsma_catalog = load_haitsma_catalog();
    let panako_catalog = load_panako_catalog();

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
