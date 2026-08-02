//! End-to-end matching tests: real `extract` → `match_one`.
//!
//! These wire the classical extractors to the matching module so the
//! offset conventions and scoring are exercised on genuine landmark /
//! frame data (not hand-built synthetic fingerprints like the unit
//! tests). Signals are deterministic chirps so landmarks vary over time
//! — a stationary tone would repeat identical landmarks and blur the
//! alignment peak.

use audiofp::classical::{Haitsma, Wang};
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, WangMatchConfig, WangMatcher,
};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

const SECS: f32 = 8.0;

/// Two counter-sweeping chirps + light noise. `variant` shifts the sweep
/// bands so different variants are genuinely different recordings.
fn synth_chirp(variant: u32, sr: u32) -> Vec<f32> {
    let n = (sr as f32 * SECS) as usize;
    let t_total = SECS;
    let base = (variant % 7) as f32 * 110.0;
    let (f0a, f1a) = (300.0 + base, 1800.0 + base);
    let (f0b, f1b) = (1600.0 - base, 500.0 + base);
    let mut out = Vec::with_capacity(n);
    let mut x = variant.max(1).wrapping_mul(2_654_435_761);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.02;
        let t = i as f32 / sr as f32;
        let pa = 2.0 * core::f32::consts::PI * (f0a * t + (f1a - f0a) * t * t / (2.0 * t_total));
        let pb = 2.0 * core::f32::consts::PI * (f0b * t + (f1b - f0b) * t * t / (2.0 * t_total));
        out.push(0.5 * pa.sin() + 0.4 * pb.sin() + noise);
    }
    out
}

/// Stationary tones — spectrally disjoint from the chirps, so it never
/// aligns with a `synth_chirp` signal.
fn synth_tones(sr: u32) -> Vec<f32> {
    let n = (sr as f32 * SECS) as usize;
    let mut out = Vec::with_capacity(n);
    let mut x = 0x1234_5678u32;
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.02;
        let t = i as f32 / sr as f32;
        let s = 0.4 * (2.0 * core::f32::consts::PI * 611.0 * t).sin()
            + 0.3 * (2.0 * core::f32::consts::PI * 997.0 * t).sin()
            + 0.3 * (2.0 * core::f32::consts::PI * 2333.0 * t).sin()
            + noise;
        out.push(s);
    }
    out
}

#[test]
fn wang_self_match_end_to_end() {
    let mut w = Wang::default();
    let sig = synth_chirp(1, 8_000);
    let fp = w
        .extract(AudioBuffer::new(&sig, SampleRate::HZ_8000))
        .unwrap();
    assert!(!fp.hashes.is_empty(), "extraction must produce landmarks");

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&fp, &fp);
    assert!(res.is_match, "self-match must be positive: {res:?}");
    assert_eq!(res.offset.frames, 0, "self-match offset must be zero");
    assert!(res.score > 0.5, "self-match score too low: {}", res.score);
    assert_eq!(res.time_scale, 1.0);
}

#[test]
fn wang_offset_recovery_end_to_end() {
    let mut w = Wang::default();
    let sig = synth_chirp(2, 8_000);
    let reference = w
        .extract(AudioBuffer::new(&sig, SampleRate::HZ_8000))
        .unwrap();

    // Prepend exactly 2 s of silence = 125 Wang frames (8000*2 / 128 =
    // 125, integer) so the per-second peak buckets stay aligned. The
    // query then starts *before* the reference by 125 frames, and the
    // documented convention (δ = t_ref − t_query) gives offset −125.
    const K_FRAMES: i64 = 125;
    let mut shifted = vec![0.0_f32; 2 * 8_000];
    shifted.extend_from_slice(&sig);
    let query = w
        .extract(AudioBuffer::new(&shifted, SampleRate::HZ_8000))
        .unwrap();

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&query, &reference);
    assert!(res.is_match, "shifted self-match must be positive: {res:?}");
    assert_eq!(
        res.offset.frames, -K_FRAMES,
        "2 s of query lead-in → offset −125 frames"
    );
    // 125 frames @ 62.5 fps = 2000 ms.
    assert_eq!(res.offset.ms, -2_000);
}

#[test]
fn wang_unrelated_no_match_end_to_end() {
    let mut w = Wang::default();
    let a = w
        .extract(AudioBuffer::new(
            &synth_chirp(3, 8_000),
            SampleRate::HZ_8000,
        ))
        .unwrap();
    let b = w
        .extract(AudioBuffer::new(&synth_tones(8_000), SampleRate::HZ_8000))
        .unwrap();
    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&a, &b);
    assert!(
        !res.is_match,
        "spectrally disjoint signals must not match: {res:?}"
    );
}

#[test]
fn haitsma_self_match_end_to_end() {
    let mut h = Haitsma::default();
    let sig = synth_chirp(5, 5_000);
    let fp = h
        .extract(AudioBuffer::new(&sig, SampleRate::HZ_5000))
        .unwrap();
    assert!(!fp.frames.is_empty(), "extraction must produce frames");
    // 8 s @ 78.125 fps ≈ 625 frames → exercises the LUT path (> 512).
    assert!(fp.frames.len() > 512);

    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        min_overlap_frames: 64,
        ..Default::default()
    });
    let res = matcher.match_one(&fp, &fp);
    assert!(res.is_match, "Haitsma self-match must be positive: {res:?}");
    assert_eq!(res.offset.frames, 0, "self-match offset must be zero");
    assert!(
        (res.score - 1.0).abs() < 1e-6,
        "self-match BER must be 0, score={}",
        res.score
    );
}
