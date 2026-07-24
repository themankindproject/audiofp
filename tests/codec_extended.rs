//! Extended codec and resampler robustness tests.
//!
//! Adds to the base `codec_roundtrip.rs`:
//! - AIFF format decoding
//! - Two-track identification (Galway vs Furious Freak)
//! - Mono vs stereo downmix equivalence
//! - MP3 sample-rate ladder (8kHz–44.1kHz → 8kHz fingerprint)
//!
//! Audio: Kevin MacLeod (incompetech.com), CC-BY 3.0. See CREDITS.md.
#![cfg(feature = "std")]

use std::collections::HashSet;

use audiofp::classical::{Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn wang_hashes(path: &str) -> HashSet<u32> {
    let samples = decode_to_mono_at(path, 8_000)
        .unwrap_or_else(|e| panic!("failed to decode {path}: {e}"));
    let mut wang = Wang::default();
    wang.extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap()
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect()
}

fn jaccard(a: &HashSet<u32>, b: &HashSet<u32>) -> f32 {
    let union = a.union(b).count();
    if union == 0 {
        return 0.0;
    }
    a.intersection(b).count() as f32 / union as f32
}

// ═══════════════════════════════════════════════════════════════════════════
// AIFF format support
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn aiff_decodes_and_matches_flac() {
    let ref_h = wang_hashes("tests/assets/galway.flac");
    let aiff_h = wang_hashes("tests/assets/galway.aiff");
    let j = jaccard(&ref_h, &aiff_h);
    eprintln!("Wang AIFF vs FLAC: Jaccard = {j:.3}");
    // AIFF is lossless PCM in a different container — should be ~identical
    assert!(j >= 0.95, "AIFF vs FLAC Jaccard = {j:.3} (expected ≥0.95)");
}

// ═══════════════════════════════════════════════════════════════════════════
// Two-track identification: Galway vs Furious Freak
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn same_track_different_codecs_high_overlap() {
    // Furious Freak in MP3 vs FLAC — same song, different codec
    let freak_flac = wang_hashes("tests/assets/freak.flac");
    let freak_mp3 = wang_hashes("tests/assets/freak.mp3");
    let j = jaccard(&freak_flac, &freak_mp3);
    eprintln!("Freak MP3 vs FLAC: Jaccard = {j:.3}");
    assert!(j >= 0.20, "Same track cross-codec Jaccard = {j:.3} (expected ≥0.20)");
}

#[test]
fn different_tracks_near_zero_overlap() {
    // Galway vs Furious Freak — different songs, should have near-zero overlap
    let galway = wang_hashes("tests/assets/galway.flac");
    let freak = wang_hashes("tests/assets/freak.flac");
    let j = jaccard(&galway, &freak);
    eprintln!("Galway vs Freak: Jaccard = {j:.3} (should be near 0)");
    // Different songs should have <5% hash overlap (random collision level)
    assert!(
        j < 0.05,
        "Different tracks overlap = {j:.3} (expected <0.05 — fingerprints not discriminating!)"
    );
}

#[test]
fn identification_scenario_cross_codec() {
    // The real scenario: can we identify Galway-as-MP3 against a database
    // containing Galway-as-FLAC and Freak-as-FLAC?
    let query = wang_hashes("tests/assets/galway.mp3");
    let db_galway = wang_hashes("tests/assets/galway.flac");
    let db_freak = wang_hashes("tests/assets/freak.flac");

    let overlap_galway = jaccard(&query, &db_galway);
    let overlap_freak = jaccard(&query, &db_freak);

    eprintln!("Query=galway.mp3 vs DB galway.flac: {overlap_galway:.3}");
    eprintln!("Query=galway.mp3 vs DB freak.flac:  {overlap_freak:.3}");

    // The correct match should be significantly higher
    assert!(
        overlap_galway > overlap_freak * 5.0,
        "Identification failed: galway overlap ({overlap_galway:.3}) not >> freak overlap ({overlap_freak:.3})"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Mono vs stereo downmix equivalence
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mono_vs_stereo_same_track_high_overlap() {
    let mono_h = wang_hashes("tests/assets/galway.mp3");
    let stereo_h = wang_hashes("tests/assets/galway_stereo.mp3");
    let j = jaccard(&mono_h, &stereo_h);
    eprintln!("Wang mono vs stereo MP3: Jaccard = {j:.3}");
    // Stereo downmix of the same track should preserve most peaks.
    // Note: ESP-ADF stereo variant uses joint-stereo encoding which
    // allocates bits differently than the mono encode, so overlap is
    // lower than lossless container differences.
    assert!(
        j >= 0.25,
        "Mono vs stereo Jaccard = {j:.3} (expected ≥0.25)"
    );
}

#[test]
fn mono_vs_stereo_flac() {
    let mono_h = wang_hashes("tests/assets/galway.flac");
    let stereo_h = wang_hashes("tests/assets/galway_stereo.flac");
    let j = jaccard(&mono_h, &stereo_h);
    eprintln!("Wang mono vs stereo FLAC: Jaccard = {j:.3}");
    assert!(
        j >= 0.60,
        "Mono vs stereo FLAC Jaccard = {j:.3} (expected ≥0.60)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MP3 sample-rate ladder: fingerprint survives arbitrary source rates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn sample_rate_ladder_all_produce_hashes() {
    let rates = [
        ("tests/assets/freak_8000hz.mp3", 8000),
        ("tests/assets/freak_11025hz.mp3", 11025),
        ("tests/assets/freak_16000hz.mp3", 16000),
        ("tests/assets/freak_22050hz.mp3", 22050),
        ("tests/assets/freak_32000hz.mp3", 32000),
        ("tests/assets/freak_44100hz.mp3", 44100),
    ];

    for (path, native_sr) in rates {
        let samples = decode_to_mono_at(path, 8_000)
            .unwrap_or_else(|e| panic!("{path} (native {native_sr} Hz) decode failed: {e}"));
        let mut wang = Wang::default();
        let fp = wang
            .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
            .unwrap_or_else(|e| panic!("{path} extract failed: {e}"));
        assert!(
            fp.hashes.len() > 20,
            "{path} (native {native_sr} Hz): only {} hashes (expected >20)",
            fp.hashes.len()
        );
        eprintln!("{path} ({native_sr} Hz → 8kHz): {} hashes", fp.hashes.len());
    }
}

#[test]
fn sample_rate_ladder_overlap_with_reference() {
    // All sample-rate variants are the same song (Furious Freak).
    // When resampled to 8kHz and fingerprinted, they should have
    // meaningful overlap with the 44.1kHz reference.
    let ref_h = wang_hashes("tests/assets/freak_44100hz.mp3");

    let variants = [
        "tests/assets/freak_8000hz.mp3",
        "tests/assets/freak_11025hz.mp3",
        "tests/assets/freak_16000hz.mp3",
        "tests/assets/freak_22050hz.mp3",
        "tests/assets/freak_32000hz.mp3",
    ];

    for path in variants {
        let h = wang_hashes(path);
        let j = jaccard(&ref_h, &h);
        eprintln!("{path} vs 44.1kHz ref: Jaccard = {j:.3}");
        // Low source rates (8kHz) lose high-frequency peaks, so lower threshold
        assert!(
            j >= 0.05,
            "{path}: Jaccard vs 44.1kHz = {j:.3} (expected ≥0.05)"
        );
    }
}

#[test]
fn panako_sample_rate_ladder() {
    // Panako should also handle the sample-rate ladder
    let ref_samples = decode_to_mono_at("tests/assets/freak_44100hz.mp3", 8_000).unwrap();
    let mut panako = Panako::default();
    let ref_fp = panako
        .extract(AudioBuffer::new(&ref_samples, SampleRate::HZ_8000))
        .unwrap();
    let ref_set: HashSet<u32> = ref_fp.hashes.iter().map(|h| h.hash).collect();

    let variants = [
        "tests/assets/freak_8000hz.mp3",
        "tests/assets/freak_22050hz.mp3",
        "tests/assets/freak_32000hz.mp3",
    ];

    for path in variants {
        let samples = decode_to_mono_at(path, 8_000).unwrap();
        let fp = panako
            .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
            .unwrap();
        let h: HashSet<u32> = fp.hashes.iter().map(|h| h.hash).collect();
        let j = jaccard(&ref_set, &h);
        eprintln!("Panako {path} vs 44.1kHz: Jaccard = {j:.3}");
        assert!(
            fp.hashes.len() > 20,
            "Panako {path}: only {} hashes",
            fp.hashes.len()
        );
    }
}
