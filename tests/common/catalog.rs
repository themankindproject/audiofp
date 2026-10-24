//! The real-audio catalog shared by `matching_real_audio` and
//! `threshold_calibration`.
//!
//! Decoding and fingerprinting 29 files dominates both suites, so each
//! per-algorithm catalog is built at most once per test binary behind a
//! [`OnceLock`]; every test in that binary borrows the same slice.
#![allow(dead_code)]

use std::sync::OnceLock;

use audiofp::classical::{
    Haitsma, HaitsmaFingerprint, Panako, PanakoFingerprint, Wang, WangFingerprint,
};
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

/// Each entry: (logical track id, file path).
/// Tracks 0-1, 11-12: multi-codec originals (positives). Tracks 2-3:
/// project-generated clips. Tracks 4-10, 13-16: Musopen classical (CC0,
/// negatives-only singletons).
pub const CATALOG: &[(usize, &str)] = &[
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

pub const NUM_TRACKS: usize = 17;

/// Tracks with multiple codec variants: every pair must match.
pub const MULTI_VARIANT_TRACKS: [usize; 4] = [0, 1, 11, 12];

/// Decode every catalog file at `rate_hz` and fingerprint it with `fp`.
fn build<F: Fingerprinter>(mut fp: F, rate_hz: u32, rate: SampleRate) -> Vec<(usize, F::Output)> {
    CATALOG
        .iter()
        .map(|&(track, path)| {
            let samples =
                decode_to_mono_at(path, rate_hz).unwrap_or_else(|e| panic!("decode {path}: {e}"));
            let out = fp
                .extract(&samples, rate)
                .unwrap_or_else(|e| panic!("extract {path}: {e}"));
            (track, out)
        })
        .collect()
}

pub fn wang_catalog() -> &'static [(usize, WangFingerprint)] {
    static CELL: OnceLock<Vec<(usize, WangFingerprint)>> = OnceLock::new();
    CELL.get_or_init(|| build(Wang::default(), 8_000, SampleRate::HZ_8000))
}

pub fn haitsma_catalog() -> &'static [(usize, HaitsmaFingerprint)] {
    static CELL: OnceLock<Vec<(usize, HaitsmaFingerprint)>> = OnceLock::new();
    CELL.get_or_init(|| build(Haitsma::default(), 5_000, SampleRate::HZ_5000))
}

pub fn panako_catalog() -> &'static [(usize, PanakoFingerprint)] {
    static CELL: OnceLock<Vec<(usize, PanakoFingerprint)>> = OnceLock::new();
    CELL.get_or_init(|| build(Panako::default(), 8_000, SampleRate::HZ_8000))
}
