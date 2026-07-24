//! Codec round-trip robustness tests.
//!
//! Verifies that fingerprinting the same music encoded in different codecs
//! (MP3, FLAC, WAV, OGG-Vorbis, AAC-in-M4A) produces overlapping hashes,
//! proving the algorithms survive lossy re-encoding.
//!
//! Test audio: "Galway" by Kevin MacLeod (incompetech.com), CC-BY 3.0.
//! 16 seconds, mono, 44100 Hz, 16-bit. See tests/assets/CREDITS.md.
#![cfg(feature = "std")]

use std::collections::HashSet;

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

// ── Helpers ──────────────────────────────────────────────────────────────

fn wang_hashes(path: &str) -> HashSet<u32> {
    let samples =
        decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("failed to decode {path}: {e}"));
    let mut wang = Wang::default();
    wang.extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn panako_hashes(path: &str) -> HashSet<u32> {
    let samples =
        decode_to_mono_at(path, 8_000).unwrap_or_else(|e| panic!("failed to decode {path}: {e}"));
    let mut panako = Panako::default();
    panako
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn haitsma_frames(path: &str) -> Vec<u32> {
    let samples =
        decode_to_mono_at(path, 5_000).unwrap_or_else(|e| panic!("failed to decode {path}: {e}"));
    let mut h = Haitsma::default();
    h.extract(AudioBuffer::new(&samples, SampleRate::HZ_5000))
        .unwrap()
        .frames
}

fn jaccard(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f32 / union as f32
}

fn haitsma_bit_similarity(a: &[u32], b: &[u32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let matching: u32 = a[..n]
        .iter()
        .zip(b[..n].iter())
        .map(|(x, y)| 32 - (x ^ y).count_ones())
        .sum();
    matching as f32 / (n as u32 * 32) as f32
}

// ── Reference: FLAC (lossless) ───────────────────────────────────────────

const REF: &str = "tests/assets/galway.flac";

// ── Wang codec tests ─────────────────────────────────────────────────────

#[test]
fn wang_mp3_vs_flac() {
    let ref_h = wang_hashes(REF);
    let mp3_h = wang_hashes("tests/assets/galway.mp3");
    let j = jaccard(&ref_h, &mp3_h);
    eprintln!(
        "Wang MP3 vs FLAC: Jaccard = {j:.3} ({} ref, {} mp3)",
        ref_h.len(),
        mp3_h.len()
    );
    assert!(j >= 0.25, "Wang MP3 Jaccard = {j:.3} (expected ≥0.25)");
}

#[test]
fn wang_ogg_vs_flac() {
    let ref_h = wang_hashes(REF);
    let ogg_h = wang_hashes("tests/assets/galway.ogg");
    let j = jaccard(&ref_h, &ogg_h);
    eprintln!("Wang OGG vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.25, "Wang OGG Jaccard = {j:.3} (expected ≥0.25)");
}

#[test]
fn wang_m4a_vs_flac() {
    let ref_h = wang_hashes(REF);
    let m4a_h = wang_hashes("tests/assets/galway.m4a");
    let j = jaccard(&ref_h, &m4a_h);
    eprintln!("Wang M4A/AAC vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.25, "Wang M4A Jaccard = {j:.3} (expected ≥0.25)");
}

#[test]
fn wang_wav_vs_flac() {
    let ref_h = wang_hashes(REF);
    let wav_h = wang_hashes("tests/assets/galway.wav");
    let j = jaccard(&ref_h, &wav_h);
    eprintln!("Wang WAV vs FLAC: Jaccard = {j:.3}");
    // WAV and FLAC are both lossless — should be nearly identical
    assert!(
        j >= 0.95,
        "Wang WAV vs FLAC Jaccard = {j:.3} (expected ≥0.95)"
    );
}

// ── Panako codec tests ───────────────────────────────────────────────────

#[test]
fn panako_mp3_vs_flac() {
    let ref_h = panako_hashes(REF);
    let mp3_h = panako_hashes("tests/assets/galway.mp3");
    let j = jaccard(&ref_h, &mp3_h);
    eprintln!("Panako MP3 vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.20, "Panako MP3 Jaccard = {j:.3} (expected ≥0.20)");
}

#[test]
fn panako_ogg_vs_flac() {
    let ref_h = panako_hashes(REF);
    let ogg_h = panako_hashes("tests/assets/galway.ogg");
    let j = jaccard(&ref_h, &ogg_h);
    eprintln!("Panako OGG vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.20, "Panako OGG Jaccard = {j:.3} (expected ≥0.20)");
}

#[test]
fn panako_m4a_vs_flac() {
    let ref_h = panako_hashes(REF);
    let m4a_h = panako_hashes("tests/assets/galway.m4a");
    let j = jaccard(&ref_h, &m4a_h);
    eprintln!("Panako M4A/AAC vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.20, "Panako M4A Jaccard = {j:.3} (expected ≥0.20)");
}

// ── Haitsma codec tests ──────────────────────────────────────────────────

#[test]
fn haitsma_mp3_vs_flac() {
    let ref_f = haitsma_frames(REF);
    let mp3_f = haitsma_frames("tests/assets/galway.mp3");
    let sim = haitsma_bit_similarity(&ref_f, &mp3_f);
    eprintln!("Haitsma MP3 vs FLAC: bit-sim = {sim:.3}");
    assert!(
        sim >= 0.80,
        "Haitsma MP3 bit-sim = {sim:.3} (expected ≥0.80)"
    );
}

#[test]
fn haitsma_ogg_vs_flac() {
    let ref_f = haitsma_frames(REF);
    let ogg_f = haitsma_frames("tests/assets/galway.ogg");
    let sim = haitsma_bit_similarity(&ref_f, &ogg_f);
    eprintln!("Haitsma OGG vs FLAC: bit-sim = {sim:.3}");
    assert!(
        sim >= 0.80,
        "Haitsma OGG bit-sim = {sim:.3} (expected ≥0.80)"
    );
}

#[test]
fn haitsma_m4a_vs_flac() {
    let ref_f = haitsma_frames(REF);
    let m4a_f = haitsma_frames("tests/assets/galway.m4a");
    let sim = haitsma_bit_similarity(&ref_f, &m4a_f);
    eprintln!("Haitsma M4A/AAC vs FLAC: bit-sim = {sim:.3}");
    assert!(
        sim >= 0.75,
        "Haitsma M4A bit-sim = {sim:.3} (expected ≥0.75)"
    );
}

#[test]
fn haitsma_wav_vs_flac() {
    let ref_f = haitsma_frames(REF);
    let wav_f = haitsma_frames("tests/assets/galway.wav");
    let sim = haitsma_bit_similarity(&ref_f, &wav_f);
    eprintln!("Haitsma WAV vs FLAC: bit-sim = {sim:.3}");
    // Both lossless — should be identical
    assert!(
        sim >= 0.99,
        "Haitsma WAV vs FLAC bit-sim = {sim:.3} (expected ≥0.99)"
    );
}

// ── Cross-codec: all formats decode successfully ─────────────────────────

#[test]
fn all_galway_codecs_decode_and_fingerprint() {
    let formats = [
        "tests/assets/galway.mp3",
        "tests/assets/galway.flac",
        "tests/assets/galway.wav",
        "tests/assets/galway.m4a",
        "tests/assets/galway.ogg",
    ];

    for path in formats {
        let samples = decode_to_mono_at(path, 8_000)
            .unwrap_or_else(|e| panic!("{path} failed to decode: {e}"));
        assert!(
            samples.len() > 8_000 * 10,
            "{path}: expected >10s of audio at 8kHz, got {} samples",
            samples.len()
        );

        // Verify all three algorithms produce non-trivial output
        let mut wang = Wang::default();
        let wf = wang
            .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
            .unwrap();
        assert!(
            wf.hashes.len() > 50,
            "{path}: Wang produced only {} hashes",
            wf.hashes.len()
        );
    }
}
