//! Integration tests for `.afp` file caching: the full
//! extract → cache → load → match pipeline on realistic audio, plus the
//! rayon parallel-extraction workflow from issue #119.
//!
//! Unit tests for the cache primitives live in `src/cache.rs`; this file
//! exercises the acceptance criteria on `audio_gen` fixtures.

#![cfg(feature = "std")]

use audiofp::cache::{CachedFingerprint, cache_to_file, load_all_cached, load_from_cache};
use audiofp::classical::Wang;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

mod common;

use common::audio_gen;

/// Unique temp dir per test, cleaned up on drop (same idiom as the
/// in-tree unit tests).
struct TempDir(std::path::PathBuf);
impl TempDir {
    fn new(tag: &str) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "audiofp_cache_it_{tag}_{}_{}_{n}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Synthesize `secs` of deterministic 8 kHz audio.
fn synth_8k(seed: u64, secs: f32) -> Vec<f32> {
    audio_gen::resample_48k_to_8k(&audio_gen::multi_instrument(seed, secs))
}

// =========================================================================
// Acceptance: extract → cache → load → compare/match == original
// =========================================================================

#[test]
fn extract_cache_load_match_roundtrip() {
    let pcm = synth_8k(7, 3.0);
    let fp = Wang::default().extract(&pcm, SampleRate::HZ_8000).unwrap();
    assert!(!fp.hashes.is_empty());

    let dir = TempDir::new("roundtrip");
    let path = dir.0.join("track.afp");
    cache_to_file(&fp, &path).unwrap();

    // Bit-exact: the cache layer adds nothing.
    let loaded: WangFingerprint = load_from_cache(&path).unwrap();
    assert_eq!(loaded.hashes, fp.hashes);
    assert_eq!(loaded.frames_per_sec, fp.frames_per_sec);

    // And the loaded fingerprint still matches the original.
    let matcher = WangMatcher::new(WangMatchConfig::default());
    let res = matcher.match_one(&loaded, &fp);
    assert!(
        res.is_match,
        "loaded fingerprint must match original: {res:?}"
    );
}

use audiofp::classical::WangFingerprint;

// =========================================================================
// Acceptance: rayon parallel extraction feeding the cache
// =========================================================================

#[cfg(feature = "rayon")]
#[test]
fn parallel_extract_then_serial_ingest() {
    use audiofp::fingerprint_batch_parallel;

    const N: u64 = 8;

    // Parallel extraction (CPU-bound) — one Wang fingerprint per track.
    let items: Vec<(u64, Vec<f32>, SampleRate)> = (1..=N)
        .map(|seed| (seed, synth_8k(seed, 2.0), SampleRate::HZ_8000))
        .collect();
    let results = fingerprint_batch_parallel(items, Wang::default);

    // Cache writes: each worker's output goes to its own file.
    let dir = TempDir::new("rayon");
    let mut originals = Vec::with_capacity(results.len());
    for (seed, res) in &results {
        let fp = res.as_ref().expect("extraction must succeed");
        cache_to_file(fp, &dir.0.join(format!("{seed}.afp"))).unwrap();
        originals.push((*seed, fp.clone()));
    }

    // Serial ingest of the whole directory.
    let cached = load_all_cached(&dir.0).unwrap();
    assert_eq!(cached.len(), N as usize);

    let matcher = WangMatcher::new(WangMatchConfig::default());
    // Path-sorted → seeds ascend with zero-padded names? No: "1.afp"… "8.afp"
    // sort lexicographically the same as numerically for 1..=8.
    for (i, (path, entry)) in cached.iter().enumerate() {
        let seed = (i + 1) as u64;
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            format!("{seed}.afp"),
            "entries must be path-sorted"
        );
        let CachedFingerprint::Wang(loaded) = entry else {
            panic!("expected a Wang fingerprint for {path:?}");
        };
        // Bit-exact per track…
        let original = &originals[i].1;
        assert_eq!(loaded.hashes, original.hashes);
        assert_eq!(loaded.frames_per_sec, original.frames_per_sec);
        // …and each cached track still self-matches.
        let res = matcher.match_one(loaded, original);
        assert!(res.is_match, "seed {seed}: {res:?}");
    }
}

// =========================================================================
// Directory scan without rayon (also keeps `load_all_cached` exercised in
// non-rayon feature combos).
// =========================================================================

#[test]
fn cache_dir_load_matches_single_file_load() {
    let dir = TempDir::new("dir_scan");
    for seed in [21_u64, 22, 23] {
        let pcm = synth_8k(seed, 2.0);
        let fp = Wang::default().extract(&pcm, SampleRate::HZ_8000).unwrap();
        cache_to_file(&fp, &dir.0.join(format!("{seed}.afp"))).unwrap();
    }

    let all = load_all_cached(&dir.0).unwrap();
    assert_eq!(all.len(), 3);
    for (path, entry) in &all {
        let single: WangFingerprint = load_from_cache(path).unwrap();
        let CachedFingerprint::Wang(from_dir) = entry else {
            panic!("expected Wang for {path:?}");
        };
        assert_eq!(from_dir.hashes, single.hashes);
        assert_eq!(from_dir.frames_per_sec, single.frames_per_sec);
    }
}

// =========================================================================
// Envelope survives the cache round-trip
// =========================================================================

#[test]
fn envelope_survives_cache() {
    let pcm = synth_8k(11, 2.0);
    let fp = Wang::default().extract(&pcm, SampleRate::HZ_8000).unwrap();

    let dir = TempDir::new("envelope");
    let path = dir.0.join("env.afp");
    cache_to_file(&fp, &path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let cached = CachedFingerprint::from_blob(&bytes).unwrap();
    let env = cached.envelope();
    assert_eq!(env.algorithm, "wang-v1");
    assert_eq!(env.sample_rate, 8_000);
    assert_eq!(env.hash_count, fp.hashes.len());
    assert_eq!(env.frames_per_sec, fp.frames_per_sec);
}
