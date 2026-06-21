//! Time-aligned hash voter — the Shazam-style matching algorithm.
//!
//! [`Matcher`] maintains an in-memory inverted index from hash value to
//! `(track_id, t_anchor)` entries.  Query it with a set of hashes from a
//! recording to find the reference track that shares the most temporally
//! aligned collisions.
//!
//! # Algorithm
//!
//! 1. For each `(query_hash, query_t_anchor)` look up all matching
//!    `(track_id, db_t_anchor)` entries in the index.
//! 2. For each hit compute `Δt = db_t_anchor − query_t_anchor` and
//!    accumulate a vote in a per-track `(Δt → count)` histogram.
//! 3. The peak of each track's histogram is its **score**; the `Δt` at
//!    that peak is the **offset** into the reference.
//! 4. The best match is the track with the largest score.
//!
//! Same-offset votes concentrate for a real match; random hash
//! collisions scatter uniformly across `Δt` values.
//!
//! # Example
//!
//! ```
//! use audiofp::classical::{Wang, WangHash};
//! use audiofp::matcher::Matcher;
//! use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
//!
//! let samples = vec![0.0_f32; 8_000 * 4];
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
//! let mut wang = Wang::default();
//! let fp = wang.extract(buf).unwrap();
//!
//! let mut m = Matcher::new();
//! m.enroll("silence".into(), &fp.hashes);
//! let results = m.query(&fp.hashes);
//! assert!(results.is_empty() || results[0].score >= 0);
//! ```

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::classical::WangHash;

/// One match result returned by [`Matcher::query`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// Name passed to [`Matcher::enroll`] for this track.
    pub track_name: String,
    /// Number of same-offset votes at the peak `Δt`.
    pub score: u32,
    /// Time offset (in STFT frames) between the query and the
    /// reference: `db_t_anchor − query_t_anchor` at the histogram peak.
    pub offset_frames: i64,
    /// Total number of hash collisions with this track (across all
    /// `Δt` values, not just the peak).
    pub total_collisions: u32,
}

/// Shazam-style in-memory inverted-index matcher.
///
/// Build one `Matcher` per set of enrolled references; reuse it across
/// many queries.
pub struct Matcher {
    /// `hash → Vec<(track_id, t_anchor)>`  — the core inverted index.
    index: BTreeMap<u32, Vec<(u32, u32)>>,
    /// `track_id → name`, indexed by enrollment order.
    track_names: Vec<String>,
}

impl Default for Matcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Matcher {
    /// Create an empty matcher with no enrolled tracks.
    pub fn new() -> Self {
        Self {
            index: BTreeMap::new(),
            track_names: Vec::new(),
        }
    }

    /// Enroll a reference recording under the given `name`.
    ///
    /// Returns the assigned `track_id` (sequential from 0).
    pub fn enroll(&mut self, name: String, hashes: &[WangHash]) -> u32 {
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

    /// Query the enrolled references with `query_hashes`.
    ///
    /// Returns matches sorted by descending `score` (best match first).
    pub fn query(&self, query_hashes: &[WangHash]) -> Vec<MatchResult> {
        // Per-track Δt histogram: track_id → (Δt → vote count).
        let mut histograms: BTreeMap<u32, BTreeMap<i64, u32>> = BTreeMap::new();

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
                let (best_dt, best_count) = hist
                    .iter()
                    .max_by_key(|&(_, c)| c)
                    .map(|(&dt, &c)| (dt, c))
                    .unwrap_or((0, 0));
                let total_collisions: u32 = hist.values().sum();
                MatchResult {
                    track_name: self.track_names[track_id as usize].clone(),
                    score: best_count,
                    offset_frames: best_dt,
                    total_collisions,
                }
            })
            .collect();
        results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        results
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::classical::Wang;
    use crate::{AudioBuffer, Fingerprinter, SampleRate};

    fn synth(seed: u32, sr: usize, secs: usize) -> Vec<f32> {
        use core::f32::consts::PI;
        let n = sr * secs;
        let mut out = Vec::with_capacity(n);
        let mut x: u32 = seed.max(1);
        for k in 0..n {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
            let t = k as f32 / sr as f32;
            let s = 0.5 * libm::sinf(2.0 * PI * 880.0 * t)
                + 0.3 * libm::sinf(2.0 * PI * 1320.0 * t)
                + noise;
            out.push(s);
        }
        out
    }

    #[test]
    fn self_query_returns_perfect_match() {
        let samples = synth(0xCAFE, 8_000, 6);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let mut m = Matcher::new();
        m.enroll("ref".into(), &fp.hashes);
        let results = m.query(&fp.hashes);

        assert!(
            !results.is_empty(),
            "self-query must find at least one match"
        );
        let best = &results[0];
        assert_eq!(best.track_name, "ref");
        assert_eq!(best.offset_frames, 0, "self-query offset must be 0");
        assert!(
            best.score as usize >= fp.hashes.len() / 2,
            "self-query score {} must be >= half the hashes {}",
            best.score,
            fp.hashes.len(),
        );
    }

    #[test]
    fn two_tracks_select_the_correct_match() {
        // Two different seeds produce two different signals; querying
        // with signal B must prefer track B.
        let samples_a = synth(0xCAFE, 8_000, 6);
        let samples_b = synth(0xDEAD, 8_000, 6);

        let mut wang = Wang::default();
        let buf_a = AudioBuffer {
            samples: &samples_a,
            rate: SampleRate::HZ_8000,
        };
        let buf_b = AudioBuffer {
            samples: &samples_b,
            rate: SampleRate::HZ_8000,
        };
        let fp_a = wang.extract(buf_a).unwrap();
        let fp_b = wang.extract(buf_b).unwrap();

        let mut m = Matcher::new();
        m.enroll("track_a".into(), &fp_a.hashes);
        m.enroll("track_b".into(), &fp_b.hashes);

        let results = m.query(&fp_b.hashes);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].track_name, "track_b",
            "query with signal B must prefer track_b",
        );
    }

    #[test]
    fn empty_db_returns_empty_results() {
        let samples = synth(0xCAFE, 8_000, 3);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let m = Matcher::new();
        let results = m.query(&fp.hashes);
        assert!(results.is_empty());
    }

    #[test]
    fn silence_query_returns_no_matches() {
        let samples = synth(0xCAFE, 8_000, 6);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let mut m = Matcher::new();
        m.enroll("ref".into(), &fp.hashes);

        // Query with silence → no hashes → no matches.
        let silence = vec![0.0_f32; 8_000 * 3];
        let buf_s = AudioBuffer {
            samples: &silence,
            rate: SampleRate::HZ_8000,
        };
        let fp_s = wang.extract(buf_s).unwrap();
        let results = m.query(&fp_s.hashes);
        assert!(
            results.is_empty(),
            "silence must not match any enrolled track"
        );
    }
}
