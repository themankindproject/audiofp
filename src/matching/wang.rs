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
//! - Index: [`HashMap`](std::collections::HashMap) under `std` (default);
//!   `BTreeMap` fallback when built without `std`.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::classical::WangFingerprint;
use crate::matching::maps::HashMap;
use crate::matching::{MatchResult, Matcher, TimeOffset, clamp_score, compute_prominence};

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
        debug_assert_eq!(
            query.frames_per_sec, reference.frames_per_sec,
            "query and reference must use the same frame rate"
        );

        if query.hashes.is_empty() || reference.hashes.is_empty() {
            return MatchResult::NONE;
        }

        let cfg = &self.cfg;

        // --- 1. Index the reference ---
        let mut index: HashMap<u32, Vec<u32>> =
            super::maps::hashmap_with_capacity(reference.hashes.len());
        for h in &reference.hashes {
            index.entry(h.hash).or_default().push(h.t_anchor);
        }
        // Remove stop-hashes (appear in too many positions)
        index.retain(|_, v| (v.len() as u32) <= cfg.max_postings_per_hash);

        if index.is_empty() {
            return MatchResult::NONE;
        }

        // --- 2. Vote into dense offset histogram ---
        //
        // Offset semantics (matches `TimeOffset` docs and the Haitsma
        // matcher): δ = t_reference − t_query. Self-match → δ = 0. A
        // positive δ means the query aligns later into the reference,
        // i.e. the query starts *after* the reference.
        let q_max = query
            .hashes
            .iter()
            .map(|h| h.t_anchor as i64)
            .max()
            .unwrap_or(0);
        let r_max = reference
            .hashes
            .iter()
            .map(|h| h.t_anchor as i64)
            .max()
            .unwrap_or(0);

        // δ = t_ref − t_query ∈ [−q_max, r_max]
        let dmin: i64 = -q_max;
        let dmax: i64 = r_max;
        let range = (dmax - dmin + 1) as usize;

        // Cap histogram size to avoid huge allocations on pathological inputs.
        const MAX_HIST_BINS: usize = 10_000_000;
        let capped = range.min(MAX_HIST_BINS);
        let mut hist: Vec<u32> = vec![0u32; capped];

        for h in &query.hashes {
            if let Some(list) = index.get(&h.hash) {
                for &tr in list {
                    let d = tr as i64 - h.t_anchor as i64;
                    let idx = (d - dmin) as usize;
                    if idx < capped {
                        hist[idx] += 1;
                    }
                }
            }
        }

        // --- 3. Consolidate ±tolerance via prefix-sum box filter ---
        let tol = cfg.offset_tolerance_frames as usize;
        let consolidated: Vec<u32> = if tol > 0 {
            // Build prefix sums for O(1) range queries
            let mut prefix: Vec<u64> = vec![0u64; capped + 1];
            for i in 0..capped {
                prefix[i + 1] = prefix[i] + hist[i] as u64;
            }
            let mut out = vec![0u32; capped];
            for (i, item) in out.iter_mut().enumerate() {
                let left = i.saturating_sub(tol);
                let right = (i + tol + 1).min(capped);
                *item = (prefix[right] - prefix[left]) as u32;
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
        let tol_i64 = cfg.offset_tolerance_frames as i64;
        let delta_star = peak_idx as i64 + dmin;
        let mut contrib_count: u32 = 0;
        for h in &query.hashes {
            if let Some(list) = index.get(&h.hash) {
                for &tr in list {
                    let d = tr as i64 - h.t_anchor as i64;
                    if (d - delta_star).abs() <= tol_i64 {
                        contrib_count += 1;
                        break;
                    }
                }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
                    t_anchor: 150
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
}
