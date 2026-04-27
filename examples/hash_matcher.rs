//! Time-aligned hash voter — the actual matching algorithm a Shazam-style
//! system runs on top of `audiofp`'s landmark output.
//!
//! ```bash
//! cargo run --example hash_matcher --release -- ref1.flac ref2.flac -- query.mp3
//! ```
//!
//! Everything before `--` is enrolled into an in-memory database; everything
//! after is queried against it. For each query the program prints the top
//! reference matches with a vote count and the best-matching time offset.
//!
//! ## Algorithm
//!
//! 1. Extract Wang landmark hashes from every reference and the query.
//! 2. For each `(query_hash, query_t_anchor)`, look up matching reference
//!    `(track_id, db_t_anchor)` entries in the index.
//! 3. For each (track_id, db_t_anchor) hit, compute
//!    `Δt = db_t_anchor - query_t_anchor` and count it in a per-track
//!    histogram.
//! 4. The peak of each track's `Δt` histogram is its score; the offset at
//!    that peak is where the query best aligns inside the reference.
//! 5. The best match is the track with the largest peak count.
//!
//! Same-offset votes are the matching primitive: random hash collisions
//! scatter across all `Δt` values and don't peak; a true match
//! concentrates votes at the alignment offset.

use std::collections::HashMap;

use audiofp::classical::{Wang, WangHash};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

const FRAMES_PER_SEC: f32 = 62.5;

/// In-memory inverted index from hash → list of `(track_id, t_anchor)` hits.
struct HashDatabase {
    index: HashMap<u32, Vec<(u32, u32)>>,
    track_names: Vec<String>,
}

impl HashDatabase {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
            track_names: Vec::new(),
        }
    }

    fn enroll(&mut self, name: String, hashes: &[WangHash]) -> u32 {
        let track_id = self.track_names.len() as u32;
        self.track_names.push(name);
        for h in hashes {
            self.index
                .entry(h.hash)
                .or_default()
                .push((track_id, h.t_anchor));
        }
        track_id
    }

    fn query(&self, query_hashes: &[WangHash]) -> Vec<MatchResult> {
        // Per-track Δt histogram: track_id → (Δt → vote count).
        let mut histograms: HashMap<u32, HashMap<i64, u32>> = HashMap::new();

        for q in query_hashes {
            if let Some(hits) = self.index.get(&q.hash) {
                for &(track_id, db_t) in hits {
                    let dt = db_t as i64 - q.t_anchor as i64;
                    *histograms
                        .entry(track_id)
                        .or_default()
                        .entry(dt)
                        .or_default() += 1;
                }
            }
        }

        let mut results: Vec<MatchResult> = histograms
            .iter()
            .map(|(&track_id, hist)| {
                // Peak = the Δt with the most votes for this track.
                let (best_dt, best_count) = hist
                    .iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(&dt, &c)| (dt, c))
                    .unwrap_or((0, 0));
                let total_hits: u32 = hist.values().sum();
                MatchResult {
                    track_name: self.track_names[track_id as usize].clone(),
                    score: best_count,
                    offset_frames: best_dt,
                    total_collisions: total_hits,
                }
            })
            .collect();
        results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        results
    }
}

#[derive(Debug)]
struct MatchResult {
    track_name: String,
    score: u32,
    offset_frames: i64,
    total_collisions: u32,
}

fn fingerprint(wang: &mut Wang, path: &str) -> Result<Vec<WangHash>, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    Ok(wang.extract(buf)?.hashes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sep = args
        .iter()
        .position(|a| a == "--")
        .ok_or("usage: hash_matcher <ref...> -- <query...>")?;
    let refs = &args[..sep];
    let queries = &args[sep + 1..];
    if refs.is_empty() || queries.is_empty() {
        return Err("need at least one reference and one query".into());
    }

    let mut wang = Wang::default();
    let mut db = HashDatabase::new();

    println!("Enrolling {} reference track(s)...", refs.len());
    for path in refs {
        let hashes = fingerprint(&mut wang, path)?;
        let id = db.enroll(path.clone(), &hashes);
        println!("  [{:>2}] {}  ({} hashes)", id, path, hashes.len());
    }

    for query_path in queries {
        println!("\n--- Querying: {} ---", query_path);
        let qh = fingerprint(&mut wang, query_path)?;
        println!("  {} query hashes", qh.len());

        let results = db.query(&qh);
        if results.is_empty() {
            println!("  no matches found in any reference");
            continue;
        }

        println!();
        println!("  Score   Hits  Offset    Track");
        println!("  -----  -----  --------  -----");
        for r in results.iter().take(5) {
            let offset_secs = r.offset_frames as f32 / FRAMES_PER_SEC;
            println!(
                "  {:>5}  {:>5}  {:>+7.2}s  {}",
                r.score, r.total_collisions, offset_secs, r.track_name,
            );
        }

        // Heuristic verdict: a peak that's at least 5× higher than the
        // mean per-bucket noise floor is almost certainly a real match.
        if let Some(best) = results.first() {
            let noise = if best.total_collisions > best.score {
                ((best.total_collisions - best.score) / 100).max(1)
            } else {
                1
            };
            if best.score > noise * 5 {
                println!("\n  → confident match: {}", best.track_name);
            } else {
                println!(
                    "\n  → no confident match (best score {} too close to noise)",
                    best.score
                );
            }
        }
    }

    Ok(())
}
