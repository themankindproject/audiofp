//! Panako matcher — tempo-invariant 2-D Hough voter.
//!
//! Panako's β makes hashes survive ±5 % time-stretch, but under scale `s`
//! the alignment is a **line** `t_ref ≈ s·t_query + b`, not a constant
//! offset. So voting happens in 2-D `(scale, offset)` space.
//!
//! # Algorithm
//!
//! 1. **Index the reference** — `HashMap<hash, Vec<(t_a, t_b, t_c)>>`,
//!    dropping hashes whose posting list exceeds `max_postings_per_hash`.
//! 2. **Match & vote** — for each query triple, for each ref triple with
//!    the same hash: compute local scale `s = ref_span / query_span` and
//!    predicted offset `b = t_a_ref − s·t_a_query`. Vote into a sparse
//!    2-D accumulator `(scale_bin, offset_bin)` if `s ∈ [scale_min, scale_max]`.
//! 3. **Peak** — find the accumulator bin with the most votes. This is
//!    the coarse `(s*, b*, votes)`.
//! 4. **RANSAC refine** (if `ransac_refine`): gather inlier `(t_q, t_r)`
//!    pairs near the coarse peak, fit an affine line via iterative
//!    sampling, keep the best inlier set.
//! 5. **Score** — normalised vote count + prominence + decision.
//!
//! # Performance
//!
//! - Time: `O(R + matched_pairs + scale_bins·offset_bins)`. Heavier
//!   than Wang (2-D grid) but still fast for song-length inputs.
//! - Memory: sparse accumulator, transient per-match.
//!
//! See [#100](https://github.com/themankindproject/audiofp/issues/100)
//! for tracking.

extern crate alloc;

use alloc::vec::Vec;

use crate::classical::PanakoFingerprint;
use crate::matching::maps::HashMap;
use crate::matching::{MatchResult, Matcher, TimeOffset, clamp_score, compute_prominence};

/// Configuration for [`PanakoMatcher`].
#[derive(Clone, Debug)]
pub struct PanakoMatchConfig {
    /// Minimum time-scale to search. Default 0.80.
    pub scale_min: f32,
    /// Maximum time-scale to search. Default 1.25.
    pub scale_max: f32,
    /// Number of scale bins. Default 24 (~2% resolution).
    pub scale_bins: u32,
    /// Consolidate votes within ±N frames. Default 1.
    pub offset_tolerance_frames: u32,
    /// Absolute floor on peak vote count. Default 5.
    pub min_votes: u32,
    /// Decision threshold on the normalised score in `[0, 1]`. Default 0.15.
    pub min_score: f32,
    /// Peak ÷ background floor. Default 5.0.
    pub min_prominence: f32,
    /// Skip hashes whose reference posting list exceeds this. Default 100.
    pub max_postings_per_hash: u32,
    /// Refine the coarse 2-D Hough peak with RANSAC line-fitting. Default true.
    pub ransac_refine: bool,
}

impl Default for PanakoMatchConfig {
    fn default() -> Self {
        Self {
            scale_min: 0.80,
            scale_max: 1.25,
            scale_bins: 24,
            offset_tolerance_frames: 1,
            min_votes: 5,
            min_score: 0.15,
            min_prominence: 5.0,
            max_postings_per_hash: 100,
            ransac_refine: true,
        }
    }
}

/// Offline 1:1 Panako matcher (2-D Hough + optional RANSAC).
///
/// The only matcher that produces a meaningful [`MatchResult::time_scale`];
/// all others report 1.0 (constant tempo).
pub struct PanakoMatcher {
    cfg: PanakoMatchConfig,
}

impl Matcher for PanakoMatcher {
    type Fingerprint = PanakoFingerprint;
    type Config = PanakoMatchConfig;

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

        // --- 1. Index reference hashes ---
        // Keyed by hash; each posting stores the full triplet timestamps.
        let mut index: HashMap<u32, Vec<(u32, u32, u32)>> = HashMap::new();
        for h in &reference.hashes {
            index
                .entry(h.hash)
                .or_default()
                .push((h.t_anchor, h.t_b, h.t_c));
        }
        index.retain(|_, v| (v.len() as u32) <= cfg.max_postings_per_hash);

        if index.is_empty() {
            return MatchResult::NONE;
        }

        // --- 2. Match → (scale, offset) vote pairs ---
        //
        // Each matched (query triple, ref triple) with equal hash yields
        // one candidate. The local scale `s` comes from comparing the
        // temporal spans of the two triples:  s = ref_span / query_span.
        // The predicted alignment is  b = t_a_ref − s·t_a_query.
        //
        // We also store the raw (t_query, t_ref) anchor pair so RANSAC
        // can re-fit from the inliers without recomputing everything.
        let scale_min = cfg.scale_min as f64;
        let scale_max = cfg.scale_max as f64;
        let scale_per_bin = (scale_max - scale_min) / cfg.scale_bins as f64;
        let eps_scale = scale_per_bin * 0.5;

        // Accumulator: (scale_bin, offset_floor) → vote count
        let mut acc: HashMap<(u32, i64), u32> = HashMap::new();
        // RANSAC candidate store: for each query anchor, list of
        // (ref_anchor, ref_b, ref_c, local_scale).
        let tol = cfg.offset_tolerance_frames as i64;

        for h in &query.hashes {
            if let Some(list) = index.get(&h.hash) {
                for &(tr_a, _tr_b, tr_c) in list {
                    let q_span = (h.t_c - h.t_anchor).max(1) as f64;
                    let r_span = (tr_c - tr_a) as f64;
                    let s = r_span / q_span;

                    if s < scale_min - eps_scale || s > scale_max + eps_scale {
                        continue;
                    }

                    let b = tr_a as f64 - s * h.t_anchor as f64;

                    // Compute bin indices
                    let s_bin = ((s - scale_min) / scale_per_bin)
                        .clamp(0.0, (cfg.scale_bins - 1) as f64)
                        as u32;
                    // Offset floor for the accumulator bin (±tol consolidation
                    // happens around the peak, so use the raw floored value).
                    let off_key = (b / (tol.max(1)) as f64).round() as i64;

                    *acc.entry((s_bin, off_key)).or_insert(0) += 1;
                }
            }
        }

        if acc.is_empty() {
            return MatchResult::NONE;
        }

        // --- 3. Find peak in 2-D accumulator ---
        // Consolidate each candidate bin by summing votes in its ±tol
        // neighbourhood (scale: ±1 bin, offset: ±offset_tolerance).
        let tol_i64 = tol;
        let acc_vec: Vec<((u32, i64), u32)> = acc.iter().map(|(&k, &v)| (k, v)).collect();

        let mut peak_votes = 0u32;
        let mut peak_s_bin: u32 = 0;
        let mut peak_off_key: i64 = 0;

        for &((s_bin, off_key), _) in &acc_vec {
            let mut neigh_votes = 0u32;
            for &((ns, no), v) in &acc_vec {
                let ds = ns.abs_diff(s_bin);
                let doff = (no - off_key).abs();
                if ds <= 1 && doff <= tol_i64 {
                    neigh_votes += v;
                }
            }
            if neigh_votes > peak_votes {
                peak_votes = neigh_votes;
                peak_s_bin = s_bin;
                peak_off_key = off_key;
            }
        }

        if peak_votes < cfg.min_votes {
            return MatchResult::NONE;
        }

        // --- 4. Prominence ---
        let acc_flat: Vec<u32> = acc.values().copied().collect();
        let peak_linear_idx = acc_vec
            .iter()
            .position(|&(k, _)| k == (peak_s_bin, peak_off_key))
            .unwrap_or(0);
        let prominence = compute_prominence(&acc_flat, peak_linear_idx);
        if prominence < cfg.min_prominence {
            return MatchResult::NONE;
        }

        // Coarse scale / offset from the peak bin centre.
        let coarse_s = scale_min + (peak_s_bin as f64 + 0.5) * scale_per_bin;
        let coarse_b = peak_off_key as f64 * (tol.max(1)) as f64;

        // --- 5. RANSAC refinement ---
        let (final_scale, final_offset, ransac_votes) = if cfg.ransac_refine {
            ransac_refine(
                query,
                reference,
                &index,
                coarse_s as f32,
                coarse_b,
                tol_i64,
                cfg,
            )
        } else {
            (coarse_s as f32, coarse_b, peak_votes)
        };

        let votes = if cfg.ransac_refine {
            ransac_votes.max(peak_votes)
        } else {
            peak_votes
        };

        // --- 6. Score ---
        let denom = query.hashes.len().max(1) as f32;
        let score = clamp_score(votes as f32 / denom);

        if score < cfg.min_score || votes < cfg.min_votes {
            return MatchResult::NONE;
        }

        let offset = TimeOffset::from_frames(final_offset.round() as i64, reference.frames_per_sec);

        MatchResult {
            is_match: true,
            score,
            votes,
            prominence,
            offset,
            time_scale: final_scale.clamp(0.5, 2.0),
        }
    }
}

// ---------------------------------------------------------------------------
// RANSAC line fitting
// ---------------------------------------------------------------------------

/// Iterative RANSAC: sample 2 pairs, fit `t_ref = s·t_query + b`,
/// count inliers, keep the best.
///
/// Returns `(scale, offset, inlier_count)`.
fn ransac_refine(
    query: &PanakoFingerprint,
    _reference: &PanakoFingerprint,
    index: &HashMap<u32, Vec<(u32, u32, u32)>>,
    coarse_s: f32,
    coarse_b: f64,
    tol_frames: i64,
    cfg: &PanakoMatchConfig,
) -> (f32, f64, u32) {
    // Gather candidate anchor pairs: for each query hash, grab all
    // matching ref entries as (t_q, t_r) pairs.
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for h in &query.hashes {
        if let Some(list) = index.get(&h.hash) {
            for &(tr_a, _, _) in list {
                pairs.push((h.t_anchor as f64, tr_a as f64));
            }
        }
    }

    if pairs.len() < 2 {
        return (coarse_s, coarse_b, 0);
    }

    // Scale-filtered inlier tolerance in offset-space.
    let inlier_tol = (tol_frames.max(2)) as f64;

    let n_iter = 128usize.min(pairs.len() * 4);
    // Use a deterministic-but-random-looking seed derived from the data.
    let seed = pairs.len().wrapping_mul(2_654_435_761) as u64;
    let mut rng = XorShift64(seed.max(1));

    let mut best_inliers = 0u32;
    let mut best_s = coarse_s;
    let mut best_b = coarse_b;

    for _ in 0..n_iter {
        // Pick two distinct random pairs.
        let i = (rng.next() % pairs.len() as u64) as usize;
        let j = (rng.next() % pairs.len() as u64) as usize;
        if i == j {
            continue;
        }

        let (tq1, tr1) = pairs[i];
        let (tq2, tr2) = pairs[j];

        let dq = tq2 - tq1;
        if dq.abs() < 1.0 {
            continue;
        }
        let s = ((tr2 - tr1) / dq) as f32;
        if s < cfg.scale_min || s > cfg.scale_max {
            continue;
        }
        let b = tr1 - s as f64 * tq1;

        // Count inliers.
        let mut inliers = 0u32;
        for &(tq, tr) in &pairs {
            let predicted = s as f64 * tq + b;
            if (tr - predicted).abs() <= inlier_tol {
                inliers += 1;
            }
        }

        if inliers > best_inliers {
            best_inliers = inliers;
            best_s = s;
            best_b = b;
        }
    }

    (best_s, best_b, best_inliers)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG for RANSAC (no_std compatible)
// ---------------------------------------------------------------------------

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::PanakoHash;

    /// Build a synthetic Panako fingerprint.
    fn make_fp(triples: &[(u32, u32, u32)], hash_offset: u32) -> PanakoFingerprint {
        PanakoFingerprint {
            hashes: triples
                .iter()
                .enumerate()
                .map(|(i, &(ta, tb, tc))| PanakoHash {
                    hash: (i as u32).wrapping_add(hash_offset),
                    t_anchor: ta,
                    t_b: tb,
                    t_c: tc,
                })
                .collect(),
            frames_per_sec: 62.5,
        }
    }

    #[test]
    fn config_defaults() {
        let c = PanakoMatchConfig::default();
        assert!((c.scale_min - 0.80).abs() < 1e-6);
        assert!((c.scale_max - 1.25).abs() < 1e-6);
        assert_eq!(c.scale_bins, 24);
        assert!(c.ransac_refine);
    }

    #[test]
    fn empty_query_returns_none() {
        let m = PanakoMatcher::new(PanakoMatchConfig::default());
        let q = make_fp(&[], 0);
        let r = make_fp(&[(10, 20, 30)], 0);
        assert_eq!(m.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn empty_reference_returns_none() {
        let m = PanakoMatcher::new(PanakoMatchConfig::default());
        let q = make_fp(&[(10, 20, 30)], 0);
        let r = make_fp(&[], 0);
        assert_eq!(m.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn self_match() {
        let m = PanakoMatcher::new(PanakoMatchConfig::default());
        let fp = make_fp(
            &[
                (10, 15, 25),
                (50, 55, 65),
                (90, 95, 105),
                (130, 135, 145),
                (170, 175, 185),
                (210, 215, 225),
                (250, 255, 265),
                (290, 295, 305),
            ],
            0,
        );
        let res = m.match_one(&fp, &fp);
        assert!(res.is_match, "self-match must be positive: {res:?}");
        assert_eq!(res.offset.frames, 0, "self-match offset must be zero");
        assert!(
            (res.time_scale - 1.0).abs() < 0.1,
            "self-match scale must be ~1.0: {}",
            res.time_scale
        );
    }

    #[test]
    fn constant_tempo_offset_recovery() {
        // Query is shifted +50 frames relative to reference; same tempo.
        let m = PanakoMatcher::new(PanakoMatchConfig {
            ransac_refine: false,
            ..Default::default()
        });
        let r = make_fp(
            &[
                (150, 155, 165),
                (190, 195, 205),
                (230, 235, 245),
                (270, 275, 285),
                (310, 315, 325),
                (350, 355, 365),
                (390, 395, 405),
                (430, 435, 445),
            ],
            0,
        );
        let q = make_fp(
            &[
                (100, 105, 115),
                (140, 145, 155),
                (180, 185, 195),
                (220, 225, 235),
                (260, 265, 275),
                (300, 305, 315),
                (340, 345, 355),
                (380, 385, 395),
            ],
            0,
        );
        let res = m.match_one(&q, &r);
        assert!(res.is_match, "shifted match expected: {res:?}");
        assert_eq!(
            res.offset.frames, 50,
            "offset must be +50, got {}",
            res.offset.frames
        );
    }

    #[test]
    fn tempo_stretch_still_matches() {
        // Simulate 10% speed-up: reference timestamps are 10% smaller.
        // Scale = ref_span/query_span = (0.9*span)/(span) = 0.9.
        // But scale < 0.80 with default config, so widen the range.
        let m = PanakoMatcher::new(PanakoMatchConfig {
            scale_min: 0.85,
            scale_max: 1.20,
            ransac_refine: false,
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        });

        let normal = make_fp(
            &[
                (100, 120, 150),
                (200, 220, 250),
                (300, 320, 350),
                (400, 420, 450),
                (500, 520, 550),
                (600, 620, 650),
                (700, 720, 750),
                (800, 820, 850),
            ],
            0,
        );
        // ~10% faster query (timestamps compressed).
        let fast = make_fp(
            &[
                (90, 108, 135),
                (180, 198, 225),
                (270, 288, 315),
                (360, 378, 405),
                (450, 468, 495),
                (540, 558, 585),
                (630, 648, 675),
                (720, 738, 765),
            ],
            0,
        );
        let res = m.match_one(&fast, &normal);
        // With our synthetic data and relaxed thresholds, a tempo
        // stretch should be detectable.
        assert!(
            res.is_match || res.votes > 0,
            "tempo-stretched variant should produce votes or match: {res:?}"
        );
    }

    #[test]
    fn unrelated_signals_no_match() {
        let m = PanakoMatcher::new(PanakoMatchConfig::default());
        let r = make_fp(
            &[
                (10, 20, 30),
                (50, 60, 70),
                (90, 100, 110),
                (130, 140, 150),
                (170, 180, 190),
            ],
            0,
        );
        let q = make_fp(
            &[
                (10, 20, 30),
                (50, 60, 70),
                (90, 100, 110),
                (130, 140, 150),
                (170, 180, 190),
            ],
            10_000,
        );
        let res = m.match_one(&q, &r);
        assert!(!res.is_match, "unrelated signals must not match: {res:?}");
    }

    #[test]
    fn determinism() {
        let m = PanakoMatcher::new(PanakoMatchConfig {
            ransac_refine: true,
            ..Default::default()
        });
        let a = make_fp(
            &[
                (10, 20, 30),
                (50, 60, 70),
                (90, 100, 110),
                (130, 140, 150),
                (170, 180, 190),
                (210, 220, 230),
                (250, 260, 270),
                (290, 300, 310),
            ],
            1,
        );
        let b = make_fp(
            &[
                (15, 25, 35),
                (55, 65, 75),
                (95, 105, 115),
                (135, 145, 155),
                (175, 185, 195),
                (215, 225, 235),
                (255, 265, 275),
                (295, 305, 315),
            ],
            1,
        );
        let r1 = m.match_one(&a, &b);
        let r2 = m.match_one(&a, &b);
        assert_eq!(r1, r2, "match_one must be deterministic");
    }

    #[test]
    fn low_votes_below_threshold_no_match() {
        let m = PanakoMatcher::new(PanakoMatchConfig {
            min_votes: 100,
            ransac_refine: false,
            ..Default::default()
        });
        let fp = make_fp(&[(10, 20, 30), (50, 60, 70)], 0);
        let res = m.match_one(&fp, &fp);
        assert!(!res.is_match);
    }

    #[test]
    fn ransac_refine_enabled_does_not_crash() {
        let m = PanakoMatcher::new(PanakoMatchConfig {
            ransac_refine: true,
            ..Default::default()
        });
        let fp = make_fp(
            &[
                (10, 20, 30),
                (50, 60, 70),
                (90, 100, 110),
                (130, 140, 150),
                (170, 180, 190),
                (210, 220, 230),
                (250, 260, 270),
                (290, 300, 310),
            ],
            0,
        );
        let res = m.match_one(&fp, &fp);
        assert!(res.is_match, "RANSAC self-match must be positive: {res:?}");
        assert_eq!(res.offset.frames, 0);
    }
}
