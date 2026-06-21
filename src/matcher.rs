//! Time-aligned hash voter — the Shazam-style matching algorithm.
//!
//! [`Matcher`] maintains an in-memory inverted index from hash value to
//! `(track_id, t_anchor)` entries.  Query it with a set of hashes from a
//! recording to find the reference track that shares the most temporally
//! aligned collisions.
//!
//! Generic over any [`Hash32`] type, so the same matcher works with
//! [`WangHash`], [`PanakoHash`], or [`HaitsmaHash`] without changing
//! the call site.
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
//! # Performance characteristics
//!
//! - **Index**: `HashMap` on `std` targets (`O(1)` lookup), `BTreeMap`
//!   on `no_std + alloc` (`O(log n)` lookup).
//! - **Query**: collects all `(track_id, Δt)` votes into a single flat
//!   `Vec`, sorts once, then counts runs — *zero per-track map
//!   allocations* during query.  The previous design allocated a
//!   nested `BTreeMap<BTreeMap>` per query; this design allocates one
//!   `Vec`.
//! - **Track names**: stored as `Arc<str>` so cloning into
//!   [`MatchResult`] is a refcount bump, not a heap allocation.
//!
//! # Example
//!
//! ```
//! use audiofp::classical::{Wang, WangHash};
//! use audiofp::hash::Hash32;
//! use audiofp::matcher::Matcher;
//! use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
//!
//! let samples = vec![0.0_f32; 8_000 * 4];
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
//! let mut wang = Wang::default();
//! let fp = wang.extract(buf).unwrap();
//!
//! let mut m: Matcher<WangHash> = Matcher::new();
//! m.enroll("silence".into(), &fp.hashes);
//! let results = m.query(&fp.hashes);
//! assert!(results.is_empty() || results[0].score >= 0);
//! ```
//!
//! [`Hash32`]: crate::hash::Hash32
//! [`WangHash`]: crate::classical::WangHash
//! [`PanakoHash`]: crate::classical::PanakoHash
//! [`HaitsmaHash`]: crate::classical::HaitsmaHash

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::hash::Hash32;

// ---------------------------------------------------------------------------
// Conditional index map: HashMap on std, BTreeMap on no_std + alloc.
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
type IndexMap<K, V> = std::collections::HashMap<K, V>;
#[cfg(not(feature = "std"))]
type IndexMap<K, V> = alloc::collections::BTreeMap<K, V>;

/// One match result returned by [`Matcher::query`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    /// Name passed to [`Matcher::enroll`] for this track.
    pub track_name: Arc<str>,
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
/// Generic over any [`Hash32`] type.  Build one `Matcher` per set of
/// enrolled references; reuse it across many queries.
///
/// # Examples
///
/// ```
/// use audiofp::classical::WangHash;
/// use audiofp::matcher::Matcher;
///
/// let mut m = Matcher::<WangHash>::new();
/// assert_eq!(m.track_count(), 0);
/// ```
pub struct Matcher<T: Hash32> {
    /// `hash → Vec<(track_id, t_anchor)>`  — the core inverted index.
    index: IndexMap<u32, Vec<(u32, u32)>>,
    /// `track_id → name`, indexed by enrollment order.
    track_names: Vec<Arc<str>>,
    /// Total number of hash entries in the index (sum of all Vec
    /// lengths).  Maintained incrementally for O(1) `hash_count()`.
    total_hashes: usize,
    /// Marker so the compiler keeps `T` named even when all methods
    /// that reference it are monomorphised away.
    _marker: core::marker::PhantomData<T>,
}

impl<T: Hash32> Default for Matcher<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Hash32> Matcher<T> {
    /// Create an empty matcher with no enrolled tracks.
    pub fn new() -> Self {
        Self {
            index: IndexMap::new(),
            track_names: Vec::new(),
            total_hashes: 0,
            _marker: core::marker::PhantomData,
        }
    }

    /// Number of enrolled reference tracks.
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.track_names.len()
    }

    /// Total number of hash entries across all enrolled tracks.
    ///
    /// This is the sum of the `Vec` lengths in the inverted index — a
    /// rough proxy for index memory footprint.
    #[must_use]
    pub fn hash_count(&self) -> usize {
        self.total_hashes
    }

    /// Remove all enrolled tracks and reset the index.
    pub fn clear(&mut self) {
        self.index.clear();
        self.track_names.clear();
        self.total_hashes = 0;
    }

    /// Enroll a reference recording under the given `name`.
    ///
    /// Returns the assigned `track_id` (sequential from 0).
    pub fn enroll(&mut self, name: String, hashes: &[T]) -> u32 {
        let track_id = self.track_names.len() as u32;
        self.track_names.push(Arc::from(name));
        for h in hashes {
            self.index
                .entry(h.hash())
                .or_default()
                .push((track_id, h.t_anchor()));
        }
        self.total_hashes += hashes.len();
        track_id
    }

    /// Remove a track and all its hash entries from the index.
    ///
    /// Returns `true` if the track was found and removed, `false` if
    /// `track_id` is out of range.
    ///
    /// **Note:** this is `O(hash_count())` in the worst case because
    /// every `Vec` in the index must be scanned for entries belonging
    /// to the removed track.  For bulk removal prefer [`clear`].
    ///
    /// [`clear`]: Self::clear
    pub fn remove_track(&mut self, track_id: u32) -> bool {
        if (track_id as usize) >= self.track_names.len() {
            return false;
        }
        // Remove from the name table.
        self.track_names.remove(track_id as usize);
        // Decrement track_ids > track_id in every Vec, and remove
        // entries that belonged to the deleted track.
        for entries in self.index.values_mut() {
            entries.retain_mut(|(tid, _)| {
                if *tid == track_id {
                    false
                } else {
                    if *tid > track_id {
                        *tid -= 1;
                    }
                    true
                }
            });
        }
        // Recompute total_hashes (cheaper than tracking deltas through
        // the retain loop, and remove_track is not a hot path).
        self.total_hashes = self.index.values().map(Vec::len).sum();
        true
    }

    /// Query the enrolled references with `query_hashes`.
    ///
    /// Returns matches sorted by descending `score` (best match first).
    /// Tracks with zero collisions are omitted from the result.
    pub fn query(&self, query_hashes: &[T]) -> Vec<MatchResult> {
        // -----------------------------------------------------------------
        // Phase 1: collect all (track_id, Δt) votes into a single flat
        // Vec.  This is the key optimisation vs the previous design:
        // instead of allocating a BTreeMap<BTreeMap> per query (one
        // BTreeMap per track, one BTreeMap per Δt bucket), we collect
        // every vote into one Vec, sort once, and count runs.
        // -----------------------------------------------------------------
        let mut votes: Vec<(u32, i64)> = Vec::new();
        for q in query_hashes {
            if let Some(hits) = self.index.get(&q.hash()) {
                for &(track_id, db_t) in hits {
                    let dt = db_t as i64 - q.t_anchor() as i64;
                    votes.push((track_id, dt));
                }
            }
        }
        if votes.is_empty() {
            return Vec::new();
        }

        // -----------------------------------------------------------------
        // Phase 2: sort by (track_id, Δt) so that all votes for the same
        // (track, offset) are contiguous.  Then count runs to find the
        // peak Δt per track and the total collisions per track.
        // -----------------------------------------------------------------
        votes.sort_unstable_by_key(|&(tid, dt)| (tid, dt));

        let mut results: Vec<MatchResult> = Vec::new();
        let mut i = 0;
        while i < votes.len() {
            let (current_track, _) = votes[i];
            let track_start = i;

            // Find the end of this track's votes.
            while i < votes.len() && votes[i].0 == current_track {
                i += 1;
            }
            let track_end = i;

            // Within [track_start, track_end): count runs of identical
            // Δt to find the peak.
            let mut best_dt: i64 = 0;
            let mut best_count: u32 = 0;
            let mut total_collisions: u32 = 0;
            let mut j = track_start;
            while j < track_end {
                let current_dt = votes[j].1;
                let run_start = j;
                while j < track_end && votes[j].1 == current_dt {
                    j += 1;
                }
                let run_count = (j - run_start) as u32;
                total_collisions += run_count;
                if run_count > best_count {
                    best_count = run_count;
                    best_dt = current_dt;
                }
            }

            results.push(MatchResult {
                track_name: Arc::clone(&self.track_names[current_track as usize]),
                score: best_count,
                offset_frames: best_dt,
                total_collisions,
            });
        }

        // Sort results by descending score.
        results.sort_unstable_by(|a, b| b.score.cmp(&a.score));
        results
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec;

    use super::*;
    use crate::classical::{Haitsma, HaitsmaHash, Panako, PanakoHash, Wang, WangHash};
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

    // --- WangHash tests ---

    #[test]
    fn wang_self_query_returns_perfect_match() {
        let samples = synth(0xCAFE, 8_000, 6);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let mut m: Matcher<WangHash> = Matcher::new();
        m.enroll("ref".to_string(), &fp.hashes);
        let results = m.query(&fp.hashes);

        assert!(
            !results.is_empty(),
            "self-query must find at least one match"
        );
        let best = &results[0];
        assert_eq!(best.track_name.as_ref(), "ref");
        assert_eq!(best.offset_frames, 0, "self-query offset must be 0");
        assert!(
            best.score as usize >= fp.hashes.len() / 2,
            "self-query score {} must be >= half the hashes {}",
            best.score,
            fp.hashes.len(),
        );
    }

    #[test]
    fn wang_two_tracks_select_the_correct_match() {
        let samples_a = synth(0xCAFE, 8_000, 6);
        let samples_b = synth(0xDEAD, 8_000, 6);

        let mut wang = Wang::default();
        let fp_a = wang
            .extract(AudioBuffer {
                samples: &samples_a,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        let fp_b = wang
            .extract(AudioBuffer {
                samples: &samples_b,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut m: Matcher<WangHash> = Matcher::new();
        m.enroll("track_a".to_string(), &fp_a.hashes);
        m.enroll("track_b".to_string(), &fp_b.hashes);

        let results = m.query(&fp_b.hashes);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].track_name.as_ref(),
            "track_b",
            "query with signal B must prefer track_b",
        );
    }

    // --- PanakoHash tests (exercises genericity) ---

    #[test]
    fn panako_self_query_returns_perfect_match() {
        let samples = synth(0xCAFE, 8_000, 6);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut panako = Panako::default();
        let fp = panako.extract(buf).unwrap();

        let mut m: Matcher<PanakoHash> = Matcher::new();
        m.enroll("ref".to_string(), &fp.hashes);
        let results = m.query(&fp.hashes);

        assert!(!results.is_empty());
        let best = &results[0];
        assert_eq!(best.track_name.as_ref(), "ref");
        assert_eq!(best.offset_frames, 0);
    }

    // --- HaitsmaHash tests (exercises genericity + hash_pairs) ---

    #[test]
    fn haitsma_self_query_returns_perfect_match() {
        let samples = synth(0xCAFE, 5_000, 6);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_5000,
        };
        let mut h = Haitsma::default();
        let fp = h.extract(buf).unwrap();
        let pairs = fp.hash_pairs();

        let mut m: Matcher<HaitsmaHash> = Matcher::new();
        m.enroll("ref".to_string(), &pairs);
        let results = m.query(&pairs);

        assert!(!results.is_empty());
        let best = &results[0];
        assert_eq!(best.track_name.as_ref(), "ref");
        assert_eq!(best.offset_frames, 0);
    }

    // --- DB management tests ---

    #[test]
    fn track_count_and_hash_count_track_enrollment() {
        let samples = synth(0xCAFE, 8_000, 4);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let mut m: Matcher<WangHash> = Matcher::new();
        assert_eq!(m.track_count(), 0);
        assert_eq!(m.hash_count(), 0);

        m.enroll("a".to_string(), &fp.hashes);
        assert_eq!(m.track_count(), 1);
        assert_eq!(m.hash_count(), fp.hashes.len());

        m.enroll("b".to_string(), &fp.hashes);
        assert_eq!(m.track_count(), 2);
        assert_eq!(m.hash_count(), fp.hashes.len() * 2);
    }

    #[test]
    fn clear_resets_the_index() {
        let samples = synth(0xCAFE, 8_000, 3);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let mut wang = Wang::default();
        let fp = wang.extract(buf).unwrap();

        let mut m: Matcher<WangHash> = Matcher::new();
        m.enroll("a".to_string(), &fp.hashes);
        assert_eq!(m.track_count(), 1);
        m.clear();
        assert_eq!(m.track_count(), 0);
        assert_eq!(m.hash_count(), 0);
        assert!(m.query(&fp.hashes).is_empty());
    }

    #[test]
    fn remove_track_evicts_entries() {
        let samples_a = synth(0xCAFE, 8_000, 4);
        let samples_b = synth(0xBEEF, 8_000, 4);
        let mut wang = Wang::default();
        let fp_a = wang
            .extract(AudioBuffer {
                samples: &samples_a,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        let fp_b = wang
            .extract(AudioBuffer {
                samples: &samples_b,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut m: Matcher<WangHash> = Matcher::new();
        m.enroll("a".to_string(), &fp_a.hashes);
        m.enroll("b".to_string(), &fp_b.hashes);
        assert_eq!(m.track_count(), 2);

        // Remove track 0 ("a").
        assert!(m.remove_track(0));
        assert_eq!(m.track_count(), 1);

        // Track "b" is now track_id 0 (re-indexed).
        let results = m.query(&fp_b.hashes);
        assert!(!results.is_empty());
        assert_eq!(results[0].track_name.as_ref(), "b");

        // Querying with "a"'s hashes should not find a match (or find
        // only noise collisions with "b").
        let results_a = m.query(&fp_a.hashes);
        // "a" was removed; any remaining matches are spurious
        // collisions with "b" and should have a low score.
        if let Some(r) = results_a.first() {
            assert_ne!(r.track_name.as_ref(), "a");
        }

        // Removing an out-of-range track_id returns false.
        assert!(!m.remove_track(99));
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

        let m: Matcher<WangHash> = Matcher::new();
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

        let mut m: Matcher<WangHash> = Matcher::new();
        m.enroll("ref".to_string(), &fp.hashes);

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

    #[test]
    fn match_result_is_clone_and_eq() {
        let r1 = MatchResult {
            track_name: Arc::from("test"),
            score: 10,
            offset_frames: 0,
            total_collisions: 20,
        };
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }
}
