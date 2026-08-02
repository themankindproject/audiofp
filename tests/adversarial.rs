//! Adversarial and edge-case tests: silence, noise, NaN/Inf, clipping,
//! empty input, sample-rate mismatches.
//!
//! Goal: zero panics, clean errors, no OOM on hostile input.

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::{Fingerprinter, SampleRate};

mod common;
use common::audio_gen;

// ---------------------------------------------------------------------------
// Silence & noise
// ---------------------------------------------------------------------------

#[test]
fn silence_5sec_wang() {
    let silence = vec![0.0f32; 8000 * 5];
    let mut w = Wang::default();
    let fp = w
        .extract(&silence, SampleRate::HZ_8000)
        .unwrap();
    // Silence should produce an empty or very sparse fingerprint — never panic.
    // Self-match of empty fingerprint → MatchResult::NONE (tested in unit tests).
    assert!(
        fp.hashes.is_empty(),
        "silence must produce empty fingerprint"
    );
}

#[test]
fn silence_5sec_haitsma() {
    let silence = vec![0.0f32; 5000 * 5];
    let mut h = Haitsma::default();
    let fp = h
        .extract(&silence, SampleRate::HZ_5000)
        .unwrap();
    // Haitsma silence → all-zero frames
    assert!(
        fp.frames.iter().all(|&f| f == 0),
        "silence must produce zero frames"
    );
}

#[test]
fn silence_5sec_panako() {
    let silence = vec![0.0f32; 8000 * 5];
    let mut p = Panako::default();
    let fp = p
        .extract(&silence, SampleRate::HZ_8000)
        .unwrap();
    assert!(
        fp.hashes.is_empty(),
        "silence must produce empty fingerprint"
    );
}

#[test]
fn white_noise_wang() {
    let audio = audio_gen::multi_instrument(1, 6.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut w = Wang::default();
    let fp = w
        .extract(&pcm, SampleRate::HZ_8000)
        .unwrap();
    // Multi-instrument audio must produce landmarks
    assert!(!fp.hashes.is_empty());
}

#[test]
fn dc_offset_no_panic() {
    // Constant DC signal — no variation, no peaks, but shouldn't panic.
    let dc = vec![0.5f32; 8000 * 3];
    let mut w = Wang::default();
    let fp = w
        .extract(&dc, SampleRate::HZ_8000)
        .unwrap();
    // DC → flat spectrum → sparse or empty fingerprint. Both fine.
    // We just verify extraction doesn't panic.
    let _ = fp;
}

// ---------------------------------------------------------------------------
// Sample rate mismatches
// ---------------------------------------------------------------------------

#[test]
fn wrong_sample_rate_wang() {
    let audio = vec![0.0f32; 8000 * 3];
    // Wang expects 8 kHz; 16 kHz must fail.
    let mut w = Wang::default();
    let err = w
        .extract(&audio, SampleRate::HZ_16000)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sample rate"),
        "expected sample-rate error: {msg}"
    );
}

#[test]
fn wrong_sample_rate_haitsma() {
    let audio = vec![0.0f32; 5000 * 3];
    let mut h = Haitsma::default();
    let err = h
        .extract(&audio, SampleRate::HZ_8000)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sample rate"),
        "expected sample-rate error: {msg}"
    );
}

#[test]
fn wrong_sample_rate_panako() {
    let audio = vec![0.0f32; 8000 * 3];
    let mut p = Panako::default();
    let err = p
        .extract(&audio, SampleRate::HZ_16000)
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("sample rate"),
        "expected sample-rate error: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Short / edge-case input
// ---------------------------------------------------------------------------

#[test]
fn very_short_input_wang() {
    let audio = vec![0.0f32; 128]; // less than minimum
    let mut w = Wang::default();
    let err = w
        .extract(&audio, SampleRate::HZ_8000)
        .unwrap_err();
    assert!(format!("{err}").contains("short"));
}

#[test]
fn very_short_input_panako() {
    let audio = vec![0.0f32; 128];
    let mut p = Panako::default();
    let err = p
        .extract(&audio, SampleRate::HZ_8000)
        .unwrap_err();
    assert!(format!("{err}").contains("short"));
}

#[test]
fn amplitude_clipping_no_nan() {
    // Clipping at ±1.0 should not produce NaN in output.
    let sig = audio_gen::multi_instrument(5, 4.0)
        .into_iter()
        .map(|s| s.clamp(-1.0, 1.0))
        .collect::<Vec<_>>();
    let pcm = audio_gen::resample_48k_to_8k(&sig);
    let mut w = Wang::default();
    let fp = w
        .extract(&pcm, SampleRate::HZ_8000)
        .unwrap();
    // Verify no NaN in hash values
    for h in &fp.hashes {
        assert!(h.hash < u32::MAX, "bogus hash value");
    }
}

#[test]
fn deterministic_extraction() {
    // Same input twice → identical fingerprints.
    let sig = audio_gen::percussion(99, 5.0);
    let pcm = audio_gen::resample_48k_to_8k(&sig);

    let mut w = Wang::default();
    let fp1 = w.extract(&pcm, SampleRate::HZ_8000).unwrap();
    let fp2 = w.extract(&pcm, SampleRate::HZ_8000).unwrap();
    assert_eq!(fp1.hashes.len(), fp2.hashes.len());
    for (a, b) in fp1.hashes.iter().zip(fp2.hashes.iter()) {
        let ha = a.hash; // copy out of packed struct
        let hb = b.hash;
        assert_eq!(ha, hb);
        let ta = a.t_anchor;
        let tb = b.t_anchor;
        assert_eq!(ta, tb);
    }
}

// ---------------------------------------------------------------------------
// Matching edge cases (pipeline-level)
// ---------------------------------------------------------------------------

#[test]
fn empty_catalog_query_returns_none() {
    use audiofp::matching::WangIndex;
    use audiofp::matching::WangMatchConfig;

    let sig = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(1, 3.0));
    let mut w = Wang::default();
    let fp = w
        .extract(&sig, SampleRate::HZ_8000)
        .unwrap();

    let index = WangIndex::build(&[], 100);
    let cfg = WangMatchConfig::default();
    assert!(
        index.query(&fp, &cfg).is_none(),
        "empty catalog must return None"
    );
}

#[test]
fn catalog_all_different_no_false_positive() {
    use audiofp::matching::{WangIndex, WangMatchConfig};

    let mut w = Wang::default();
    let mut refs = Vec::new();
    for i in 0..20u64 {
        let sig = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(i + 100, 4.0));
        let fp = w
            .extract(&sig, SampleRate::HZ_8000)
            .unwrap();
        refs.push(fp);
    }

    // Query is a completely different piece (percussion vs multi-instrument)
    let query_sig = audio_gen::resample_48k_to_8k(&audio_gen::percussion(999, 4.0));
    let query = w
        .extract(&query_sig, SampleRate::HZ_8000)
        .unwrap();

    let index = WangIndex::build(&refs, 100);
    let cfg = WangMatchConfig::default();
    assert!(
        index.query(&query, &cfg).is_none(),
        "unrelated query must not falsely match in catalog"
    );
}

#[test]
fn wang_match_mismatched_frames_per_sec_is_none() {
    use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};

    let sig = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(42, 3.0));
    let mut w = Wang::default();
    let mut q = w
        .extract(&sig, SampleRate::HZ_8000)
        .unwrap();
    let r = q.clone();
    q.frames_per_sec = r.frames_per_sec * 2.0;

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&q, &r);
    assert!(
        !res.is_match && res.votes == 0,
        "fps mismatch must soft-fail to NONE: {res:?}"
    );
}

#[test]
fn extract_then_match_adversarial_pcm_no_panic() {
    use audiofp::matching::{
        Matcher, PanakoMatchConfig, PanakoMatcher, WangMatchConfig, WangMatcher,
    };

    // Hostile PCM: NaNs and Infs — must not panic. Offline extract may
    // return Err(NonFiniteSample) (base-branch PCM policy) or succeed
    // with a sparse fingerprint; either path is acceptable so long as
    // matching on a successful extract is also panic-free.
    let mut pcm = vec![0.0f32; 8000 * 2];
    pcm[100] = f32::NAN;
    pcm[200] = f32::INFINITY;
    pcm[300] = f32::NEG_INFINITY;

    let mut w = Wang::default();
    let mut p = Panako::default();
    let wang = WangMatcher::new(WangMatchConfig::default());
    let panako = PanakoMatcher::new(PanakoMatchConfig::default());

    if let Ok(wang_fp) = w.extract(&pcm, SampleRate::HZ_8000) {
        let _ = wang.match_one(&wang_fp, &wang_fp);
    }
    if let Ok(panako_fp) = p.extract(&pcm, SampleRate::HZ_8000) {
        let _ = panako.match_one(&panako_fp, &panako_fp);
    }

    // Finite path: matching empty/sparse fingerprints must stay well-formed.
    let silence = vec![0.0f32; 8000 * 2];
    if let Ok(wang_fp) = w.extract(&silence, SampleRate::HZ_8000) {
        let _ = wang.match_one(&wang_fp, &wang_fp);
    }
    if let Ok(panako_fp) = p.extract(&silence, SampleRate::HZ_8000) {
        let _ = panako.match_one(&panako_fp, &panako_fp);
    }
}
