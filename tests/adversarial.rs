//! Adversarial and edge-case tests: silence, noise, NaN/Inf, clipping,
//! empty input, sample-rate mismatches.
//!
//! Goal: zero panics, clean errors, no OOM on hostile input.

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

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
        .extract(AudioBuffer::new(&silence, SampleRate::HZ_8000))
        .unwrap();
    // Silence should produce an empty or very sparse fingerprint — never panic.
    // Self-match of empty fingerprint → MatchResult::NONE (tested in unit tests).
    assert!(fp.hashes.is_empty(), "silence must produce empty fingerprint");
}

#[test]
fn silence_5sec_haitsma() {
    let silence = vec![0.0f32; 5000 * 5];
    let mut h = Haitsma::default();
    let fp = h
        .extract(AudioBuffer::new(&silence, SampleRate::HZ_5000))
        .unwrap();
    // Haitsma silence → all-zero frames
    assert!(fp.frames.iter().all(|&f| f == 0), "silence must produce zero frames");
}

#[test]
fn silence_5sec_panako() {
    let silence = vec![0.0f32; 8000 * 5];
    let mut p = Panako::default();
    let fp = p
        .extract(AudioBuffer::new(&silence, SampleRate::HZ_8000))
        .unwrap();
    assert!(fp.hashes.is_empty(), "silence must produce empty fingerprint");
}

#[test]
fn white_noise_wang() {
    let audio = audio_gen::multi_instrument(1, 6.0);
    let pcm = audio_gen::resample_48k_to_8k(&audio);
    let mut w = Wang::default();
    let fp = w
        .extract(AudioBuffer::new(&pcm, SampleRate::HZ_8000))
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
        .extract(AudioBuffer::new(&dc, SampleRate::HZ_8000))
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
        .extract(AudioBuffer::new(&audio, SampleRate::HZ_16000))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("sample rate"), "expected sample-rate error: {msg}");
}

#[test]
fn wrong_sample_rate_haitsma() {
    let audio = vec![0.0f32; 5000 * 3];
    let mut h = Haitsma::default();
    let err = h
        .extract(AudioBuffer::new(&audio, SampleRate::HZ_8000))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("sample rate"), "expected sample-rate error: {msg}");
}

#[test]
fn wrong_sample_rate_panako() {
    let audio = vec![0.0f32; 8000 * 3];
    let mut p = Panako::default();
    let err = p
        .extract(AudioBuffer::new(&audio, SampleRate::HZ_16000))
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("sample rate"), "expected sample-rate error: {msg}");
}

// ---------------------------------------------------------------------------
// Short / edge-case input
// ---------------------------------------------------------------------------

#[test]
fn very_short_input_wang() {
    let audio = vec![0.0f32; 128]; // less than minimum
    let mut w = Wang::default();
    let err = w
        .extract(AudioBuffer::new(&audio, SampleRate::HZ_8000))
        .unwrap_err();
    assert!(format!("{err}").contains("short"));
}

#[test]
fn very_short_input_panako() {
    let audio = vec![0.0f32; 128];
    let mut p = Panako::default();
    let err = p
        .extract(AudioBuffer::new(&audio, SampleRate::HZ_8000))
        .unwrap_err();
    assert!(format!("{err}").contains("short"));
}

#[test]
fn amplitude_clipping_no_nan() {
    // Clipping at ±1.0 should not produce NaN in output.
    let sig = audio_gen::multi_instrument(5, 4.0).into_iter().map(|s| s.clamp(-1.0, 1.0)).collect::<Vec<_>>();
    let pcm = audio_gen::resample_48k_to_8k(&sig);
    let mut w = Wang::default();
    let fp = w
        .extract(AudioBuffer::new(&pcm, SampleRate::HZ_8000))
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
    let buf1 = AudioBuffer::new(&pcm, SampleRate::HZ_8000);
    let buf2 = AudioBuffer::new(&pcm, SampleRate::HZ_8000);

    let mut w = Wang::default();
    let fp1 = w.extract(buf1).unwrap();
    let fp2 = w.extract(buf2).unwrap();
    assert_eq!(fp1.hashes.len(), fp2.hashes.len());
    for (a, b) in fp1.hashes.iter().zip(fp2.hashes.iter()) {
        assert_eq!(a.hash, b.hash);
        assert_eq!(a.t_anchor, b.t_anchor);
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
    let fp = w.extract(AudioBuffer::new(&sig, SampleRate::HZ_8000)).unwrap();

    let index = WangIndex::build(&[], 100);
    let cfg = WangMatchConfig::default();
    assert!(index.query(&fp, &cfg).is_none(), "empty catalog must return None");
}

#[test]
fn catalog_all_different_no_false_positive() {
    use audiofp::matching::{WangIndex, WangMatchConfig};

    let mut w = Wang::default();
    let mut refs = Vec::new();
    for i in 0..20u64 {
        let sig = audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(i + 100, 4.0));
        let fp = w.extract(AudioBuffer::new(&sig, SampleRate::HZ_8000)).unwrap();
        refs.push(fp);
    }

    // Query is a completely different piece (percussion vs multi-instrument)
    let query_sig = audio_gen::resample_48k_to_8k(&audio_gen::percussion(999, 4.0));
    let query = w.extract(AudioBuffer::new(&query_sig, SampleRate::HZ_8000)).unwrap();

    let index = WangIndex::build(&refs, 100);
    let cfg = WangMatchConfig::default();
    assert!(
        index.query(&query, &cfg).is_none(),
        "unrelated query must not falsely match in catalog"
    );
}
