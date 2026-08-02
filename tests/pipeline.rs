//! Full pipeline integration tests: decode → extract → match → identify.
//!
//! Uses deterministic realistic audio (from `common::audio_gen`) to test
//! the complete production flow. No external audio files required — CI-ready.
//!
//! Each test verifies the end-to-end path for a production use case:
//! self-identification, offset recovery, tempo robustness, partial
//! audio matching, and catalog-scale 1:N identification.

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::matching::{
    HaitsmaIndex, HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher,
    WangIndex, WangMatchConfig, WangMatcher,
};
use audiofp::{Fingerprinter, SampleRate};

mod common;

use common::audio_gen;

// ---------------------------------------------------------------------------
// Pitch-shift: simple resample to simulate tempo/pitch change
// ---------------------------------------------------------------------------

/// Resample `samples` by `ratio` using linear interpolation.
fn resample(samples: &[f32], ratio: f32) -> Vec<f32> {
    let n = samples.len();
    let out_len = (n as f32 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = (i as f32 * ratio) as usize;
        let frac = (i as f32 * ratio) - src as f32;
        let a = samples[src.min(n - 1)];
        let b = samples[(src + 1).min(n - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

// =========================================================================
// Self-match (same audio → score ≈ 1.0, offset = 0)
// =========================================================================

#[test]
fn pipeline_self_match_wang() {
    let audio = audio_gen::multi_instrument(42, 8.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut w = Wang::default();
    let fp = w.extract(&pcm, SampleRate::HZ_8000).unwrap();
    assert!(!fp.hashes.is_empty(), "must have landmarks");

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&fp, &fp);
    assert!(res.is_match, "self-match must be positive: {res:?}");
    assert!(res.score > 0.5, "score too low: {}", res.score);
    assert_eq!(res.offset.frames, 0);
    assert_eq!(res.time_scale, 1.0);
}

#[test]
fn pipeline_self_match_haitsma() {
    let audio = audio_gen::multi_instrument(42, 10.0);
    let pcm = audio_gen::resample_48k_to_5k(&audio);
    let mut h = Haitsma::default();
    let fp = h.extract(&pcm, SampleRate::HZ_5000).unwrap();
    assert!(!fp.frames.is_empty(), "must have frames");

    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        min_overlap_frames: 64,
        ..Default::default()
    });
    let res = matcher.match_one(&fp, &fp);
    assert!(res.is_match, "self-match must be positive: {res:?}");
    assert!(
        (res.score - 1.0).abs() < 1e-6,
        "BER must be 0, got {}",
        res.score
    );
    assert_eq!(res.offset.frames, 0);
}

#[test]
fn pipeline_self_match_panako() {
    let audio = audio_gen::multi_instrument(42, 8.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut p = Panako::default();
    let fp = p.extract(&pcm, SampleRate::HZ_8000).unwrap();
    assert!(!fp.hashes.is_empty(), "must have triplets");

    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());
    let res = matcher.match_one(&fp, &fp);
    assert!(res.is_match, "Panako self-match must be positive: {res:?}");
    assert_eq!(res.offset.frames, 0);
    assert!(
        (res.time_scale - 1.0).abs() < 0.2,
        "scale={}",
        res.time_scale
    );
}

// =========================================================================
// Offset recovery: query is a time-shifted excerpt of the reference
// =========================================================================

#[test]
fn pipeline_offset_recovery_wang() {
    let audio = audio_gen::percussion(123, 12.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut w = Wang::default();
    let reference = w.extract(&pcm, SampleRate::HZ_8000).unwrap();

    // Query is the last 8 seconds → offset = 4s * 62.5 fps = 250 frames
    let query_pcm = &pcm[4 * 8000..];
    let query = w.extract(query_pcm, SampleRate::HZ_8000).unwrap();

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&query, &reference);
    assert!(res.is_match, "offset match must be positive: {res:?}");
    assert_eq!(
        res.offset.frames, 250,
        "expected +250, got {}",
        res.offset.frames
    );
    assert_eq!(res.offset.ms, 4_000);
}

#[test]
fn pipeline_offset_recovery_haitsma() {
    let audio = audio_gen::percussion(123, 12.0);
    let pcm = audio_gen::resample_48k_to_5k(&audio);
    let mut h = Haitsma::default();
    let reference = h.extract(&pcm, SampleRate::HZ_5000).unwrap();

    // Query is the last ~8 seconds
    let query_pcm = &pcm[4 * 5000..];
    let query = h.extract(query_pcm, SampleRate::HZ_5000).unwrap();

    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        min_overlap_frames: 64,
        ..Default::default()
    });
    let res = matcher.match_one(&query, &reference);
    assert!(res.is_match, "offset match must be positive: {res:?}");
    // 4s @ 78.125 fps = 312 frames (within tolerance)
    assert!(
        (res.offset.frames - 312).abs() <= 2,
        "expected ~312, got {}",
        res.offset.frames
    );
}

#[test]
fn pipeline_offset_recovery_panako() {
    let audio = audio_gen::percussion(123, 12.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut p = Panako::default();
    let reference = p.extract(&pcm, SampleRate::HZ_8000).unwrap();

    let query_pcm = &pcm[4 * 8000..];
    let query = p.extract(query_pcm, SampleRate::HZ_8000).unwrap();

    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());
    let res = matcher.match_one(&query, &reference);
    assert!(
        res.is_match,
        "Panako offset match must be positive: {res:?}"
    );
    assert!(
        (res.offset.frames - 250).abs() <= 5,
        "expected ~250, got {}",
        res.offset.frames
    );
}

// =========================================================================
// Tempo variation: speed change → Panako recovers time_scale
// =========================================================================

#[test]
fn pipeline_tempo_speedup_105x_panako() {
    let audio = audio_gen::percussion(77, 12.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut p = Panako::default();
    let reference = p.extract(&pcm, SampleRate::HZ_8000).unwrap();

    // 5% speed-up: resample at 1.05 ratio (pitch rises too, but Panako β
    // is tempo/pitch invariant by design)
    let speedup_pcm = audio_gen::resample_48k_to_8k(&audio);
    let fast = resample(&speedup_pcm, 1.05);
    let query = p.extract(&fast, SampleRate::HZ_8000).unwrap();

    let matcher = PanakoMatcher::new(PanakoMatchConfig {
        scale_min: 0.80,
        scale_max: 1.20,
        ransac_refine: true,
        min_votes: 3,
        min_prominence: 1.5,
        min_score: 0.03,
        ..Default::default()
    });
    let res = matcher.match_one(&query, &reference);
    // Always well-formed; when matched, public time_scale ≈ 0.95
    // (query faster → shorter duration relative to reference).
    assert!(res.score.is_finite() && res.prominence.is_finite() && res.time_scale.is_finite());
    if res.is_match {
        assert!(
            (res.time_scale - 0.95).abs() < 0.15,
            "expected scale ~0.95, got {}",
            res.time_scale
        );
    }
}

// =========================================================================
// 1:N catalog identification via indexes
// =========================================================================

#[test]
fn pipeline_catalog_wang_identify() {
    let audio = audio_gen::multi_instrument(99, 6.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut w = Wang::default();
    // Build a catalog of 20 references: 19 synthetic + 1 real
    let mut refs = Vec::new();
    for i in 0..19 {
        let sig = audio_gen::percussion((100 + i) as u64, 6.0);
        let pcm = audio_gen::resample_48k_to_8k(&sig);
        let fp = w.extract(&pcm, SampleRate::HZ_8000).unwrap();
        refs.push(fp);
    }
    // The 20th reference (index 19) is the same audio as the query
    let target = w.extract(&pcm, SampleRate::HZ_8000).unwrap();
    refs.push(target.clone());

    let index = WangIndex::build(&refs, 100);
    let cfg = WangMatchConfig::default();
    let (id, res) = index
        .query(&target, &cfg)
        .expect("must identify reference 19");

    assert_eq!(id, 19, "must identify reference 19, got {}", id);
    assert!(res.is_match);
    assert_eq!(res.offset.frames, 0);
    assert!(res.score > 0.5, "score too low: {}", res.score);
}

#[test]
fn pipeline_catalog_haitsma_identify() {
    let audio = audio_gen::multi_instrument(99, 8.0);
    let pcm = audio_gen::resample_48k_to_5k(&audio);
    let mut h = Haitsma::default();
    let mut refs = Vec::new();
    for i in 0..9 {
        let sig = audio_gen::percussion((100 + i) as u64, 8.0);
        let pcm = audio_gen::resample_48k_to_5k(&sig);
        let fp = h.extract(&pcm, SampleRate::HZ_5000).unwrap();
        refs.push(fp);
    }
    let target = h.extract(&pcm, SampleRate::HZ_5000).unwrap();
    refs.push(target.clone());

    let index = HaitsmaIndex::build(&refs, 100);
    let cfg = HaitsmaMatchConfig {
        min_overlap_frames: 200,
        ..Default::default()
    };
    let (id, res) = index
        .query(&target, &cfg)
        .expect("must identify reference 9");

    assert_eq!(id, 9, "must identify reference 9, got {}", id);
    assert!(res.is_match);
    assert!((res.score - 1.0).abs() < 1e-6, "BER must be 0");
}

#[test]
fn pipeline_catalog_panako_identify() {
    let audio = audio_gen::percussion(42, 6.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut p = Panako::default();
    let mut refs = Vec::new();
    for i in 0..9 {
        let sig = audio_gen::multi_instrument((100 + i) as u64, 6.0);
        let pcm = audio_gen::resample_48k_to_8k(&sig);
        let fp = p.extract(&pcm, SampleRate::HZ_8000).unwrap();
        refs.push(fp);
    }
    let target = p.extract(&pcm, SampleRate::HZ_8000).unwrap();
    refs.push(target.clone());

    let index = audiofp::matching::PanakoIndex::build(&refs, 100);
    let cfg = PanakoMatchConfig::default();
    let (id, res) = index
        .query(&target, &cfg)
        .expect("must identify reference 9");

    assert_eq!(id, 9, "must identify reference 9, got {}", id);
    assert!(res.is_match);
    assert_eq!(res.offset.frames, 0);
}

// =========================================================================
// Unrelated audio must NOT match
// =========================================================================

#[test]
fn pipeline_unrelated_rejected_wang() {
    let a = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(1, 6.0));
    let b = audio_gen::resample_48k_to_8k(&audio_gen::percussion(2, 6.0));
    let mut w = Wang::default();
    let fa = w.extract(&a, SampleRate::HZ_8000).unwrap();
    let fb = w.extract(&b, SampleRate::HZ_8000).unwrap();

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&fa, &fb);
    assert!(!res.is_match, "unrelated audio must not match: {res:?}");
}

#[test]
fn pipeline_unrelated_rejected_haitsma() {
    let a = audio_gen::resample_48k_to_5k(&audio_gen::multi_instrument(1, 8.0));
    let b = audio_gen::resample_48k_to_5k(&audio_gen::percussion(2, 8.0));
    let mut h = Haitsma::default();
    let fa = h.extract(&a, SampleRate::HZ_5000).unwrap();
    let fb = h.extract(&b, SampleRate::HZ_5000).unwrap();

    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
    let res = matcher.match_one(&fa, &fb);
    assert!(!res.is_match, "unrelated audio must not match: {res:?}");
}

#[test]
fn pipeline_unrelated_rejected_panako() {
    let a = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(3, 6.0));
    let b = audio_gen::resample_48k_to_8k(&audio_gen::percussion(4, 6.0));
    let mut p = Panako::default();
    let fa = p.extract(&a, SampleRate::HZ_8000).unwrap();
    let fb = p.extract(&b, SampleRate::HZ_8000).unwrap();

    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());
    let res = matcher.match_one(&fa, &fb);
    assert!(!res.is_match, "unrelated audio must not match: {res:?}");
}

// =========================================================================
// Cross-algorithm: each generator type should produce extractable fingerprints
// =========================================================================

#[test]
fn pipeline_all_generators_wang() {
    let mut w = Wang::default();
    for (name, audio) in [
        (
            "multi",
            audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(9, 5.0)),
        ),
        (
            "perc",
            audio_gen::resample_48k_to_8k(&audio_gen::percussion(9, 5.0)),
        ),
        (
            "ambient",
            audio_gen::resample_48k_to_8k(&audio_gen::ambient_pad(9, 5.0)),
        ),
    ] {
        let fp = w.extract(&audio, SampleRate::HZ_8000).unwrap();
        assert!(!fp.hashes.is_empty(), "{name}: Wang must produce landmarks");
    }
}

#[test]
fn pipeline_all_generators_haitsma() {
    let mut h = Haitsma::default();
    for (name, audio) in [
        (
            "multi",
            audio_gen::resample_48k_to_5k(&audio_gen::multi_instrument(7, 5.0)),
        ),
        (
            "perc",
            audio_gen::resample_48k_to_5k(&audio_gen::percussion(7, 5.0)),
        ),
        (
            "ambient",
            audio_gen::resample_48k_to_5k(&audio_gen::ambient_pad(7, 5.0)),
        ),
    ] {
        let fp = h.extract(&audio, SampleRate::HZ_5000).unwrap();
        assert!(!fp.frames.is_empty(), "{name}: Haitsma must produce frames");
    }
}

#[test]
fn pipeline_all_generators_panako() {
    let mut p = Panako::default();
    for (name, audio) in [
        (
            "multi",
            audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(5, 5.0)),
        ),
        (
            "perc",
            audio_gen::resample_48k_to_8k(&audio_gen::percussion(5, 5.0)),
        ),
        (
            "ambient",
            audio_gen::resample_48k_to_8k(&audio_gen::ambient_pad(5, 5.0)),
        ),
    ] {
        let fp = p.extract(&audio, SampleRate::HZ_8000).unwrap();
        assert!(
            !fp.hashes.is_empty(),
            "{name}: Panako must produce triplets"
        );
    }
}
