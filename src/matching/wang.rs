//! Wang (Shazam-style) matcher — offset-histogram voter.
//!
//! The canonical Shazam alignment: matching landmark hashes must agree
//! on a single **constant time offset**. Random collisions scatter
//! across offsets while a true match spikes at one.
//!
//! # Algorithm
//!
//! 1. **Index the reference** — `HashMap<hash, Vec<t_anchor>>`, dropping
//!    hashes whose posting list exceeds `max_postings_per_hash`.
//! 2. **Vote** — for each query landmark, for each ref anchor with the
//!    same hash, compute `δ = t_ref − t_query` and bump a dense
//!    histogram bin.
//! 3. **Consolidate** — box-convolve with `±offset_tolerance_frames` so
//!    framing-jitter votes coalesce into a single peak.
//! 4. **Prominence** — `peak ÷ (mean of non-peak bins + 1)` — the
//!    discriminator between a true spike and flat random collisions.
//! 5. **Score** — normalised to `[0, 1]` by dividing peak votes by the
//!    number of query hashes that landed in the winning offset span.
//!
//! # Performance
//!
//! - Time: `O(R + Q + range)` — sub-millisecond for song-length inputs.
//! - Memory: dense histogram ≈ 4 bytes/frame → ~60 KB for 4 min @ 62.5 fps.
//! - Index: sorted flat arrays with binary search
//!   ([`SortedPostings`](super::maps::SortedPostings)).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::classical::WangFingerprint;
use crate::matching::maps::SortedPostings;
use crate::matching::{
    MatchResult, Matcher, TimeOffset, clamp_score, compute_prominence, frames_per_sec_compatible,
};

/// Configuration for [`WangMatcher`].
#[derive(Clone, Debug)]
pub struct WangMatchConfig {
    /// Consolidate votes within ±N frames of the peak (framing jitter). Default 1.
    pub offset_tolerance_frames: u32,
    /// Absolute floor on peak vote count. Default 5.
    pub min_votes: u32,
    /// Decision threshold on the normalised score in `[0, 1]`. Default 0.15.
    pub min_score: f32,
    /// Peak ÷ background floor (false-positive guard). Default 5.0.
    pub min_prominence: f32,
    /// Skip hashes whose reference posting list exceeds this. Default 100.
    pub max_postings_per_hash: u32,
}

impl Default for WangMatchConfig {
    fn default() -> Self {
        Self {
            offset_tolerance_frames: 1,
            min_votes: 5,
            min_score: 0.15,
            min_prominence: 5.0,
            max_postings_per_hash: 100,
        }
    }
}

/// Prebuilt single-reference index for [`WangMatcher`].
///
/// [`WangMatcher::match_one`] rebuilds the reference's inverted index on
/// **every call** (`SortedPostings::build` is O(R log R) per match —
/// audit C1). When the same reference is matched repeatedly (batch 1:1,
/// query loops against a fixed catalog, streaming identification), build
/// the [`WangRefIndex`] once and call
/// [`WangMatcher::match_one_prebuilt`]; the per-query cost then drops to
/// the pure O(Q log U + range) voting pass, and the index applies the
/// same stop-hash filter the 1:1 path would.
///
/// The two entry points are guaranteed to agree: `match_one` is defined
/// as build-then-`match_one_prebuilt`, so results are identical by
/// construction.
pub struct WangRefIndex {
    postings: SortedPostings,
    r_max: i64,
    frames_per_sec: f32,
}

impl WangRefIndex {
    /// Build the index for `reference` using the stop-hash policy from
    /// `cfg` (`max_postings_per_hash`). Returns `None` when the reference
    /// has no hashes or every hash is filtered out — the same conditions
    /// under which `match_one` returns [`MatchResult::NONE`].
    #[must_use]
    pub fn build(reference: &WangFingerprint, cfg: &WangMatchConfig) -> Option<Self> {
        if reference.hashes.is_empty() {
            return None;
        }
        let r_hashes: alloc::vec::Vec<(u32, u32)> = reference
            .hashes
            .iter()
            .map(|h| (h.hash, h.t_anchor))
            .collect();
        let postings = SortedPostings::build(&r_hashes, cfg.max_postings_per_hash)?;
        let r_max = r_hashes.iter().map(|&(_, t)| t as i64).max().unwrap_or(0);
        Some(Self {
            postings,
            r_max,
            frames_per_sec: reference.frames_per_sec,
        })
    }
}

/// Offline 1:1 Wang matcher (Shazam-style offset-histogram voter).
pub struct WangMatcher {
    cfg: WangMatchConfig,
}

impl Matcher for WangMatcher {
    type Fingerprint = WangFingerprint;
    type Config = WangMatchConfig;

    fn new(cfg: Self::Config) -> Self {
        Self { cfg }
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn match_one(&self, query: &Self::Fingerprint, reference: &Self::Fingerprint) -> MatchResult {
        // Soft-fail on fps mismatch in all builds (audit 67-5).
        if !frames_per_sec_compatible(query.frames_per_sec, reference.frames_per_sec) {
            return MatchResult::NONE;
        }

        if query.hashes.is_empty() || reference.hashes.is_empty() {
            return MatchResult::NONE;
        }

        // audit C1: the prebuilt index path and the 1:1 path must agree;
        // `match_one` is exactly build-then-`match_one_prebuilt`.
        let index = match WangRefIndex::build(reference, &self.cfg) {
            Some(index) => index,
            None => return MatchResult::NONE,
        };
        self.match_one_prebuilt(query, &index)
    }
}

impl WangMatcher {
    /// Match `query` against a reference whose index was built once
    /// (`WangRefIndex::build`), skipping the per-call O(R log R) index
    /// rebuild (audit C1).
    ///
    /// Produces exactly the [`Matcher::match_one`] result for the same
    /// query/reference pair.
    #[must_use]
    pub fn match_one_prebuilt(
        &self,
        query: &WangFingerprint,
        reference: &WangRefIndex,
    ) -> MatchResult {
        // Soft-fail on fps mismatch in all builds (audit 67-5).
        if !frames_per_sec_compatible(query.frames_per_sec, reference.frames_per_sec) {
            return MatchResult::NONE;
        }

        if query.hashes.is_empty() {
            return MatchResult::NONE;
        }

        let cfg = &self.cfg;
        let index = &reference.postings;

        // --- 2. Vote into dense offset histogram ---
        // δ = t_ref − t_query ∈ [−q_max, r_max]
        //
        // `query.hashes` is iterated directly: the previous
        // `Vec<(u32, u32)>` projection allocated and memcpy'd the whole
        // query on every call for no benefit — `WangHash` already stores
        // exactly `(hash, t_anchor)`.
        let q_max = query
            .hashes
            .iter()
            .map(|h| h.t_anchor as i64)
            .max()
            .unwrap_or(0);
        let r_max = reference.r_max;

        let dmin: i64 = -q_max;
        let dmax: i64 = r_max;
        // Range arithmetic in u64: on 32-bit targets a span beyond
        // 4 Gi bins used to truncate through `as usize` and silently
        // fold distant offsets onto the same bins.
        let range_u64 = (dmax - dmin + 1) as u64;

        // Cap the histogram so a pathological query/reference cannot OOM;
        // votes beyond the cap are silently dropped.
        const MAX_HIST_BINS: u64 = 10_000_000;
        let capped_u64 = range_u64.min(MAX_HIST_BINS);
        let capped = capped_u64 as usize;
        let mut hist: Vec<u32> = vec![0u32; capped];

        for h in &query.hashes {
            let q_t = h.t_anchor;
            for &tr in index.get(h.hash) {
                let d = tr as i64 - q_t as i64;
                let idx = (d - dmin) as u64;
                if idx < capped_u64 {
                    let bucket = &mut hist[idx as usize];
                    // wrapping: votes are capped by MAX_HIST_BINS; overflow
                    // only on pathological input and is harmless here.
                    *bucket = bucket.wrapping_add(1);
                }
            }
        }

        // --- 3. Consolidate ±tolerance with a sliding window ---
        // Equivalent to a ±tol box filter, but O(1) per bin with no
        // transient O(range)-byte u64 prefix array (the previous
        // prefix-sum approach peaked at ~3× the histogram's memory).
        //
        // The running `window` sum is `u64` because it adds up to
        // `2·tol + 1` `u32` bins; narrowing back to `u32` saturates so a
        // crafted fingerprint whose bin sum overflows cannot wrap a huge
        // peak down to a small value (which would silently drop a match).
        let tol = cfg.offset_tolerance_frames as usize;
        let consolidated: Vec<u32> = if tol > 0 {
            let mut out = vec![0u32; capped];
            let mut window: u64 = 0;
            // Window for bin 0: hist[0 ..= min(tol, capped-1)].
            let init_hi = tol.min(capped - 1);
            for &v in &hist[..=init_hi] {
                window += v as u64;
            }
            out[0] = u32::try_from(window).unwrap_or(u32::MAX);
            for i in 1..capped {
                let enter = i + tol;
                if enter < capped {
                    window += hist[enter] as u64;
                }
                // The bin leaving the window on the left.
                if i > tol {
                    window -= hist[i - tol - 1] as u64;
                }
                out[i] = u32::try_from(window).unwrap_or(u32::MAX);
            }
            out
        } else {
            // Zero tolerance: histogram IS the consolidated view — avoid clone.
            core::mem::take(&mut hist)
        };

        // --- 4. Find peak (use consolidated for robustness, but pick
        //     the plateau centre to avoid jitter bias) ---
        let peak_val = *consolidated.iter().max().unwrap_or(&0);
        if peak_val < cfg.min_votes {
            return MatchResult::NONE;
        }

        let plateau_start = consolidated
            .iter()
            .position(|&v| v == peak_val)
            .unwrap_or(0);
        let plateau_end = consolidated
            .iter()
            .rposition(|&v| v == peak_val)
            .unwrap_or(0);
        let peak_idx = (plateau_start + plateau_end) / 2;

        // --- 5. Prominence (on consolidated histogram) ---
        let prominence = compute_prominence(&consolidated, peak_idx);
        if prominence < cfg.min_prominence {
            return MatchResult::NONE;
        }

        // --- 6. Score ---
        // Counts how many distinct query hashes contribute at least one
        // vote to the winning offset. `SortedPostings::get` returns the
        // group sorted ascending, so the "is there a posting whose δ lands
        // within ±tol of δ*" test is a binary search for the first anchor
        // at or past the window's lower edge — O(log P) instead of the
        // previous O(P) linear scan over every posting.
        let tol_i64 = cfg.offset_tolerance_frames as i64;
        let delta_star = peak_idx as i64 + dmin;
        let mut contrib_count: u32 = 0;
        for h in &query.hashes {
            let postings = index.get(h.hash);
            // t_ref must satisfy |(t_ref − q_t) − δ*| ≤ tol, i.e.
            // t_ref ∈ [q_t + δ* − tol, q_t + δ* + tol].
            let centre = h.t_anchor as i64 + delta_star;
            let lo = centre - tol_i64;
            let hi = centre + tol_i64;
            let start = postings.partition_point(|&tr| (tr as i64) < lo);
            if start < postings.len() && (postings[start] as i64) <= hi {
                contrib_count += 1;
            }
        }

        let denom = query.hashes.len().max(1) as f32;
        let score = clamp_score(contrib_count as f32 / denom);

        if score < cfg.min_score {
            return MatchResult::NONE;
        }

        let offset = TimeOffset::from_frames(delta_star, reference.frames_per_sec);

        MatchResult {
            is_match: true,
            score,
            votes: peak_val,
            prominence,
            offset,
            time_scale: 1.0,
        }
    }
}

impl Default for WangMatcher {
    fn default() -> Self {
        Self::new(WangMatchConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::WangHash;

    /// Build a synthetic Wang fingerprint with known anchor positions.
    fn make_fp(anchors: &[u32], hash_offset: u32) -> WangFingerprint {
        WangFingerprint {
            hashes: anchors
                .iter()
                .enumerate()
                .map(|(i, &t)| WangHash {
                    hash: (i as u32).wrapping_add(hash_offset),
                    t_anchor: t,
                })
                .collect(),
            frames_per_sec: 62.5,
        }
    }

    #[test]
    fn config_defaults() {
        let c = WangMatchConfig::default();
        assert_eq!(c.offset_tolerance_frames, 1);
        assert_eq!(c.min_votes, 5);
        assert!((c.min_score - 0.15).abs() < 1e-6);
        assert_eq!(c.max_postings_per_hash, 100);
    }

    #[test]
    fn empty_query_returns_none() {
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let q = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let r = make_fp(&[10, 20, 30], 0);
        assert_eq!(matcher.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn empty_reference_returns_none() {
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let q = make_fp(&[10, 20, 30], 0);
        let r = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        assert_eq!(matcher.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn self_match_score_near_one() {
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let fp = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let result = matcher.match_one(&fp, &fp);
        assert!(result.is_match, "self-match must be positive");
        assert!(
            result.score > 0.8,
            "self-match score too low: {}",
            result.score
        );
        assert_eq!(result.offset.frames, 0, "self-match offset must be zero");
        assert_eq!(result.time_scale, 1.0);
    }

    #[test]
    fn offset_recovery_query_starts_after_reference() {
        // The query is the segment of the reference that starts 50 frames
        // in: a shared landmark sits at reference frame t_query + 50, so
        // δ = t_ref − t_query = +50 (query starts *after* the reference).
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let ref_fp = make_fp(&[150, 250, 350, 450, 550, 650, 750, 850], 0);
        let query_fp = make_fp(&[100, 200, 300, 400, 500, 600, 700, 800], 0);
        let result = matcher.match_one(&query_fp, &ref_fp);
        assert!(result.is_match, "offset match expected");
        assert_eq!(
            result.offset.frames, 50,
            "query starts +50 frames into the reference"
        );
        assert!(result.score > 0.5, "score too low: {}", result.score);
    }

    #[test]
    fn offset_recovery_query_starts_before_reference() {
        // The query has 50 frames of extra lead-in, so its landmarks sit at
        // higher local frame indices than the reference's: δ = t_ref −
        // t_query = −50 (query starts *before* the reference).
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let ref_fp = make_fp(&[100, 200, 300, 400, 500, 600, 700, 800], 0);
        let query_fp = make_fp(&[150, 250, 350, 450, 550, 650, 750, 850], 0);
        let result = matcher.match_one(&query_fp, &ref_fp);
        assert!(result.is_match, "negative offset match expected");
        assert_eq!(
            result.offset.frames, -50,
            "query starts 50 frames before the reference"
        );
    }

    #[test]
    fn unrelated_signals_no_match() {
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let ref_fp = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        // Different hashes → no collisions
        let query_fp = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 1000);
        let result = matcher.match_one(&query_fp, &ref_fp);
        assert!(!result.is_match, "unrelated signals must not match");
        assert_eq!(result.votes, 0);
    }

    #[test]
    fn low_votes_below_threshold_no_match() {
        // Only 3 hashes total, min_votes=5 → no match
        let matcher = WangMatcher::new(WangMatchConfig {
            min_votes: 5,
            ..Default::default()
        });
        let fp = make_fp(&[10, 20, 30], 0);
        let result = matcher.match_one(&fp, &fp);
        assert!(!result.is_match);
    }

    #[test]
    fn determinism() {
        let matcher = WangMatcher::new(WangMatchConfig::default());
        let a = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let b = make_fp(&[15, 25, 35, 45, 55, 65, 75, 85], 0);
        let r1 = matcher.match_one(&a, &b);
        let r2 = matcher.match_one(&a, &b);
        assert_eq!(r1, r2, "match_one must be deterministic");
    }

    #[test]
    fn stop_hashes_filtered_out() {
        let matcher = WangMatcher::new(WangMatchConfig {
            max_postings_per_hash: 1,
            min_votes: 3,
            min_prominence: 2.0,
            ..Default::default()
        });
        let fp = make_fp(&[10, 20, 30, 40, 50], 0);
        let result = matcher.match_one(&fp, &fp);
        assert!(result.is_match, "non-stop hashes should still match");
    }

    #[test]
    fn prominence_guards_against_random_collisions() {
        // Two signals with different hash values but same anchor pattern —
        // this should produce low prominence (flat histogram).
        let matcher = WangMatcher::new(WangMatchConfig {
            min_prominence: 5.0,
            ..Default::default()
        });
        let ref_fp = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        // Hashes with duplicates → some collisions but spread across offsets
        let query_fp = WangFingerprint {
            hashes: (0..50)
                .map(|i| WangHash {
                    hash: (i / 5),
                    t_anchor: i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let result = matcher.match_one(&query_fp, &ref_fp);
        // Should fail due to low prominence (spread out, not a sharp spike)
        assert!(!result.is_match);
    }

    #[test]
    fn consolidation_coalesces_jitter_votes() {
        // Hashes that are offset by ±1 frame should coalesce into one peak
        let cfg = WangMatchConfig {
            offset_tolerance_frames: 2,
            min_votes: 3,
            min_score: 0.1,
            min_prominence: 2.0,
            ..Default::default()
        };
        let matcher = WangMatcher::new(cfg);

        // Reference: 100, 200, 300
        let ref_fp = make_fp(&[100, 200, 300], 0);
        // Query: 150 (+50), 249 (+49), 351 (+51) — jittered around +50
        let query_fp = WangFingerprint {
            hashes: alloc::vec![
                WangHash {
                    hash: 0,
                    t_anchor: 150,
                },
                WangHash {
                    hash: 1,
                    t_anchor: 249
                },
                WangHash {
                    hash: 2,
                    t_anchor: 351
                },
            ],
            frames_per_sec: 62.5,
        };
        let result = matcher.match_one(&query_fp, &ref_fp);
        assert!(
            result.is_match,
            "jittered votes should consolidate: {:?}",
            result
        );
    }

    // ── WangRefIndex (audit C1) ──
    //
    // `match_one_prebuilt` must agree with `match_one` for every query /
    // reference pair and configuration, and reusing one built index
    // across queries must stay deterministic.

    fn parity_cases() -> alloc::vec::Vec<(
        WangFingerprint,
        WangFingerprint,
        WangMatchConfig,
        &'static str,
    )> {
        alloc::vec![
            (
                make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0),
                make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0),
                WangMatchConfig::default(),
                "self match",
            ),
            (
                make_fp(&[100, 200, 300, 400, 500, 600, 700, 800], 0),
                make_fp(&[150, 250, 350, 450, 550, 650, 750, 850], 0),
                WangMatchConfig::default(),
                "positive offset",
            ),
            (
                make_fp(&[150, 250, 350, 450, 550, 650, 750, 850], 0),
                make_fp(&[100, 200, 300, 400, 500, 600, 700, 800], 0),
                WangMatchConfig::default(),
                "negative offset",
            ),
            (
                make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 1000),
                make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0),
                WangMatchConfig::default(),
                "unrelated hashes",
            ),
        ]
    }

    #[test]
    fn prebuilt_matches_one_to_one_for_all_cases() {
        for (query, reference, cfg, name) in parity_cases() {
            let matcher = WangMatcher::new(cfg.clone());
            let direct = matcher.match_one(&query, &reference);
            let index = WangRefIndex::build(&reference, &cfg)
                .expect("reference with hashes must build an index");
            let prebuilt = matcher.match_one_prebuilt(&query, &index);
            assert_eq!(
                direct, prebuilt,
                "match_one_prebuilt diverged from match_one for case: {name}"
            );
        }
    }

    #[test]
    fn prebuilt_reuse_across_queries_is_deterministic() {
        let reference = make_fp(&[100, 200, 300, 400, 500, 600, 700, 800], 0);
        let cfg = WangMatchConfig::default();
        let matcher = WangMatcher::new(cfg.clone());
        let index = WangRefIndex::build(&reference, &cfg).unwrap();

        for query in [10u32, 60, 110, 160, 210] {
            let q = make_fp(&[query, query + 100, query + 200, query + 300], 0);
            // Same query → same result, no matter how many times the
            // index has been reused.
            let a = matcher.match_one_prebuilt(&q, &index);
            let b = matcher.match_one_prebuilt(&q, &index);
            assert_eq!(a, b, "prebuilt matching must be deterministic");
            let c = matcher.match_one(&q, &reference);
            assert_eq!(a, c, "prebuilt must equal the 1:1 path");
        }
    }

    #[test]
    fn prebuilt_handles_empty_and_fps_mismatch() {
        let cfg = WangMatchConfig::default();
        let matcher = WangMatcher::new(cfg.clone());
        let empty = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        assert!(WangRefIndex::build(&empty, &cfg).is_none());

        let reference = make_fp(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let index = WangRefIndex::build(&reference, &cfg)
            .expect("reference with hashes must build an index");

        // fps mismatch → NONE even with a prebuilt index.
        let mismatched = WangFingerprint {
            hashes: reference.hashes.clone(),
            frames_per_sec: 31.25,
        };
        assert_eq!(
            matcher.match_one_prebuilt(&mismatched, &index),
            MatchResult::NONE
        );
        // Empty query → NONE.
        assert_eq!(
            matcher.match_one_prebuilt(&empty, &index),
            MatchResult::NONE
        );
    }
}
