//! Parallel fingerprint extraction → `.afp` cache files → serial ingest.
//!
//! ```bash
//! cargo run --example cache_workflow --features rayon -- /tmp/afp-cache
//! ```
//!
//! Synthesizes 8 deterministic tracks, fingerprints them in parallel with
//! rayon, writes `.afp` cache files, then ingests the whole directory
//! serially and verifies every cached track identifies against the
//! in-memory catalog. The workflow mirrors `olaf cache` / `olaf
//! store_cached`: extraction is CPU-bound and parallel; storage and
//! indexing are single-writer and serial.

use std::path::PathBuf;

use audiofp::cache::{CachedFingerprint, cache_to_file, load_all_cached};
use audiofp::classical::Wang;
use audiofp::fingerprint_batch_parallel;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Result, SampleRate};

fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| audiofp::AfpError::Config("usage: cache_workflow <cache-dir>".into()))?;
    std::fs::create_dir_all(&dir).map_err(|e| audiofp::AfpError::io_with_path(&dir, e))?;

    // ------------------------------------------------------------------
    // 1. Parallel extraction (CPU-bound) → .afp cache files.
    // ------------------------------------------------------------------
    const TRACKS: u64 = 8;
    let items: Vec<(u64, Vec<f32>, SampleRate)> = (1..=TRACKS)
        .map(|seed| (seed, synth_track(seed), SampleRate::HZ_8000))
        .collect();

    println!("Extracting {TRACKS} tracks in parallel...");
    let results = fingerprint_batch_parallel(items, Wang::default);

    for (seed, res) in &results {
        let fp = res.as_ref().map_err(|e| {
            audiofp::AfpError::Config(format!("extraction failed for seed {seed}: {e}"))
        })?;
        cache_to_file(fp, &dir.join(format!("{seed}.afp")))?;
    }
    println!("Wrote {TRACKS} .afp cache files to {}", dir.display());

    // ------------------------------------------------------------------
    // 2. Serial ingest + verification.
    // ------------------------------------------------------------------
    let cached = load_all_cached(&dir)?;
    println!("Ingested {} cached fingerprints", cached.len());

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let mut verified = 0;
    for (path, entry) in &cached {
        let CachedFingerprint::Wang(fp) = entry else {
            println!("  {} — not a Wang fingerprint, skipping", path.display());
            continue;
        };
        let res = matcher.match_one(fp, fp);
        if res.is_match {
            verified += 1;
        }
        println!(
            "  {} — {} hashes, self-match score {:.3}",
            path.display(),
            fp.hashes.len(),
            res.score
        );
    }
    println!("{verified}/{} cached tracks verified", cached.len());
    Ok(())
}

/// Deterministic synthetic "track": bass line + sustained chord + stepped
/// melody, all derived from the seed, so different seeds produce different
/// landmark configurations (~2 s of mono 8 kHz audio). Real applications
/// would decode files via `audiofp::io` instead.
fn synth_track(seed: u64) -> Vec<f32> {
    const SR: f32 = 8_000.0;
    const SECS: usize = 2;
    let n = SR as usize * SECS;

    let root = 110.0 + (seed % 6) as f32 * 30.0; // distinct root per seed
    let chord = [1.0, 1.25, 1.5, 2.0]; // spread partials across the band
    let melody_steps = [1.0, 1.5, 2.0, 1.75, 1.25, 2.25]; // per 1/3 s note
    let melody_note_secs = 1.0 / 3.0;

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f32 / SR;
        let phase = std::f32::consts::TAU * t;

        // Bass: root an octave down, one cycle per 0.5 s of amplitude.
        let bass = (phase * root / 2.0).sin() * 0.30 * (0.5 + 0.5 * (phase * 2.0).cos());

        // Chord: sustained partials with slow per-partial tremolo.
        let chord_val: f32 = chord
            .iter()
            .enumerate()
            .map(|(k, &ratio)| {
                let f = root * ratio;
                (phase * f).sin() * 0.15 * (0.6 + 0.4 * (phase * (0.7 + 0.13 * k as f32)).cos())
            })
            .sum();

        // Melody: one stepped note per third of a second.
        let step = melody_steps[(t / melody_note_secs) as usize % melody_steps.len()];
        let melody_f = root * 4.0 * step;
        let note_t = t % melody_note_secs;
        let melody =
            (phase * melody_f).sin() * 0.35 * (-note_t * 3.0).exp() * (note_t * 60.0).min(1.0);

        out.push(bass + chord_val + melody);
    }
    out
}
