//! Golden regression tests on real audio assets.
//!
//! If any of these fail after a code change, it means hash output has
//! drifted — the algorithm version suffix must be bumped.
#![cfg(feature = "std")]

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

fn assert_golden_wang(asset: &str, golden_path: &str) {
    let samples = decode_to_mono_at(asset, 8_000).unwrap();
    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000).unwrap();
    let actual: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    let expected = std::fs::read(golden_path)
        .unwrap_or_else(|_| panic!("golden file missing: {golden_path}. Run: cargo test --test generate_real_goldens --all-features -- --ignored"));
    assert_eq!(
        actual,
        expected,
        "Wang hash drift detected on {asset}! {} hashes, golden has {} bytes",
        fp.hashes.len(),
        expected.len()
    );
}

fn assert_golden_panako(asset: &str, golden_path: &str) {
    let samples = decode_to_mono_at(asset, 8_000).unwrap();
    let mut panako = Panako::default();
    let fp = panako.extract(&samples, SampleRate::HZ_8000).unwrap();
    let actual: Vec<u8> = fp
        .hashes
        .iter()
        .flat_map(|h| h.hash.to_le_bytes())
        .collect();
    let expected = std::fs::read(golden_path)
        .unwrap_or_else(|_| panic!("golden file missing: {golden_path}."));
    assert_eq!(actual, expected, "Panako hash drift detected on {asset}!");
}

fn assert_golden_haitsma(asset: &str, golden_path: &str) {
    let samples = decode_to_mono_at(asset, 5_000).unwrap();
    let mut haitsma = Haitsma::default();
    let fp = haitsma.extract(&samples, SampleRate::HZ_5000).unwrap();
    let actual: Vec<u8> = fp.frames.iter().flat_map(|f| f.to_le_bytes()).collect();
    let expected = std::fs::read(golden_path)
        .unwrap_or_else(|_| panic!("golden file missing: {golden_path}."));
    assert_eq!(actual, expected, "Haitsma hash drift detected on {asset}!");
}

#[test]
fn wang_piano_golden() {
    assert_golden_wang("tests/assets/piano.ogg", "tests/goldens/wang_v1_piano.bin");
}

#[test]
fn wang_speech_golden() {
    assert_golden_wang(
        "tests/assets/speech.ogg",
        "tests/goldens/wang_v1_speech.bin",
    );
}

#[test]
fn panako_piano_golden() {
    assert_golden_panako(
        "tests/assets/piano.ogg",
        "tests/goldens/panako_v2_piano.bin",
    );
}

#[test]
fn panako_speech_golden() {
    assert_golden_panako(
        "tests/assets/speech.ogg",
        "tests/goldens/panako_v2_speech.bin",
    );
}

#[test]
fn haitsma_piano_golden() {
    assert_golden_haitsma(
        "tests/assets/piano.ogg",
        "tests/goldens/haitsma_v1_piano.bin",
    );
}

#[test]
fn haitsma_speech_golden() {
    assert_golden_haitsma(
        "tests/assets/speech.ogg",
        "tests/goldens/haitsma_v1_speech.bin",
    );
}
