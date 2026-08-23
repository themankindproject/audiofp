//! Panako matcher — tempo-invariant 2-D Hough voter.
//!
//! Panako's β makes hashes survive ±5 % time-stretch, but under scale `s`
//! the alignment is a **line** `t_ref ≈ s·t_query + b`, not a constant
//! offset. So voting happens in 2-D `(scale, offset)` space.
//!
//! # Algorithm
//!
//! 1. **Index the reference** — `HashMap<hash, Vec<(t_anchor, t_b, t_c)>>`,
//!    dropping hashes whose posting list exceeds `max_postings_per_hash`.
//! 2. **Match & vote** — for each query triple, for each ref triple with
//!    the same hash: compute local scale `s = ref_span / query_span` and
//!    predicted offset `b = t_anchor_ref − s·t_anchor_query`. Vote into a
//!    sparse 2-D accumulator `(scale_bin, offset_bin)` if
//!    `s ∈ [scale_min, scale_max]`.
//! 3. **Peak** — find the accumulator bin with the most votes. This is
//!    the coarse `(s*, b*, votes)`. Peak finding consolidates each bin's
//!    ±(1 scale-bin, ±tol offset-bin) neighbourhood — `O(bins²)`, fine
//!    for the typical few-hundred-bin accumulators of song-length inputs.
//! 4. **RANSAC refine** (if `ransac_refine`): from the `(t_q, t_r)`
//!    anchor pairs collected during voting, iteratively sample 2 pairs,
//!    fit `t_ref = s·t_query + b`, keep the largest inlier set.
//! 5. **Score** — normalised vote count + prominence + decision.
//!
//! # Performance
//!
//! - Time: `O(R + matched_pairs + bins²)`. Heavier than Wang (2-D grid
//!   + RANSAC) but still fast for song-length inputs.
//! - Memory: sparse accumulator, transient per-match.
//!
//! See [#100](https://github.com/themankindproject/audiofp/issues/100)
//! for tracking.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::classical::PanakoFingerprint;
use crate::matching::maps::{HashMap, hashmap_new};
use crate::matching::{
    MatchResult, Matcher, TimeOffset, clamp_score, compute_prominence, frames_per_sec_compatible,
};

/// Configuration for [`PanakoMatcher`].
///
/// # Scale search vs reported `time_scale`
///
/// `scale_min` / `scale_max` / `scale_bins` describe the **internal**
/// Hough grid for `s = ref_span / query_span`. The public
/// [`MatchResult::time_scale`] is the reciprocal `1/s` (query/reference
/// duration), clamped to `[0.5, 2.0]` on output.
///
/// Default search is `s ∈ [0.80, 1.25]` (~±25 %). A true 2× speed-up
/// (`s ≈ 2`) is outside that grid and will not be recovered — widen
/// `scale_min`/`scale_max` if you need larger tempo ratios. The wider
/// output clamp cannot invent votes the accumulator never saw.
#[derive(Clone, Debug)]
pub struct PanakoMatchConfig {
    /// Minimum **internal** time-scale `s = ref/query` to search. Default 0.80.
    pub scale_min: f32,
    /// Maximum **internal** time-scale `s = ref/query` to search. Default 1.25.
    pub scale_max: f32,
    /// Number of scale bins. Default 24 (~2% resolution). Must be > 0.
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
/// all others report 1.0 (constant tempo). Scale estimates outside the
/// wider `[0.5, 2.0]` range are clamped (see `match_one`).
pub struct PanakoMatcher {
    cfg: PanakoMatchConfig,
}

/// Prebuilt reference index for [`PanakoMatcher::match_one_prebuilt`].
///
/// Holds the stop-hash-filtered inverted map
/// `hash → Vec<(t_anchor, t_b, t_c)>` plus the reference's
/// `frames_per_sec`. Build once with [`PanakoRefIndex::build`], then
/// match many queries against the same reference without the per-call
/// O(R) HashMap construction.
///
/// The two entry points are guaranteed to agree: `match_one` is defined
/// as build-then-`match_one_prebuilt`, so results are identical by
/// construction.
pub struct PanakoRefIndex {
    index: HashMap<u32, Vec<(u32, u32, u32)>>,
    frames_per_sec: f32,
    hash_count: usize,
}

impl PanakoRefIndex {
    /// Build the index for `reference` using the stop-hash policy from
    /// `cfg` (`max_postings_per_hash`). Returns `None` when the reference
    /// has no hashes or every hash is filtered out.
    #[must_use]
    pub fn build(reference: &PanakoFingerprint, cfg: &PanakoMatchConfig) -> Option<Self> {
        if reference.hashes.is_empty() {
            return None;
        }
        let mut index: HashMap<u32, Vec<(u32, u32, u32)>> = hashmap_new();
        for h in &reference.hashes {
            index
                .entry(h.hash)
                .or_default()
                .push((h.t_anchor, h.t_b, h.t_c));
        }
        index.retain(|_, v| (v.len() as u32) <= cfg.max_postings_per_hash);
        if index.is_empty() {
            return None;
        }
        Some(Self {
            index,
            frames_per_sec: reference.frames_per_sec,
            hash_count: reference.hashes.len(),
        })
    }

    /// Number of hashes in the (unfiltered) reference fingerprint.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hash_count
    }

    /// `true` when the reference fingerprint was empty / non-matchable.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hash_count == 0
    }
}

impl Matcher for PanakoMatcher {
    type Fingerprint = PanakoFingerprint;
    type Config = PanakoMatchConfig;

    fn new(cfg: Self::Config) -> Self {
        Self {
            cfg: normalize_config(cfg),
        }
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn match_one(&self, query: &Self::Fingerprint, reference: &Self::Fingerprint) -> MatchResult {
        // Soft-fail on fps mismatch in all builds (audit 67-5). A silent
        // conversion with the reference rate would produce wrong `offset.ms`.
        if !frames_per_sec_compatible(query.frames_per_sec, reference.frames_per_sec) {
            return MatchResult::NONE;
        }

        if query.hashes.is_empty() || reference.hashes.is_empty() {
            return MatchResult::NONE;
        }

        // Delegate to build-then-prebuilt so both paths agree by construction.
        let index = match PanakoRefIndex::build(reference, &self.cfg) {
            Some(index) => index,
            None => return MatchResult::NONE,
        };
        self.match_one_prebuilt(query, &index)
    }
}

impl PanakoMatcher {
    /// Match `query` against a reference whose index was built once
    /// ([`PanakoRefIndex::build`]), skipping the per-call O(R) HashMap
    /// construction.
    ///
    /// Produces exactly the [`Matcher::match_one`] result for the same
    /// query/reference pair.
    #[must_use]
    pub fn match_one_prebuilt(
        &self,
        query: &PanakoFingerprint,
        reference: &PanakoRefIndex,
    ) -> MatchResult {
        // Soft-fail on fps mismatch.
        if !frames_per_sec_compatible(query.frames_per_sec, reference.frames_per_sec) {
            return MatchResult::NONE;
        }

        if query.hashes.is_empty() || reference.is_empty() {
            return MatchResult::NONE;
        }

        let cfg = &self.cfg;
        let index = &reference.index;

        // --- 2. Match → (scale, offset) vote pairs ---
        //
        // Each matched (query triple, ref triple) with equal hash yields
        // one candidate. The local scale `s` comes from comparing the
        // temporal spans of the two triples:  s = ref_span / query_span.
        // The predicted alignment is  b = t_a_ref − s·t_a_query.
        //
        // We also collect the raw (t_query, t_ref) anchor pairs so the
        // optional RANSAC pass can re-fit without rescanning the index.
        let scale_min = cfg.scale_min as f64;
        let scale_max = cfg.scale_max as f64;
        let scale_per_bin = (scale_max - scale_min) / cfg.scale_bins as f64;
        // ±half-bin slack so a scale at the grid edge still votes into the
        // boundary bin.
        let eps_scale = scale_per_bin * 0.5;

        // Accumulator: (scale_bin, offset_bin) → vote count
        let mut acc: HashMap<(u32, i64), u32> = hashmap_new();
        // Candidate anchor pairs for RANSAC: (t_query_anchor, t_ref_anchor).
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        let tol = cfg.offset_tolerance_frames as i64;

        for h in &query.hashes {
            let hash = h.hash;
            let q_ta = h.t_anchor;
            let q_tc = h.t_c;
            if let Some(list) = index.get(&hash) {
                for &(tr_a, _tr_b, tr_c) in list {
                    let q_span = (q_tc - q_ta).max(1) as f64;
                    let r_span = (tr_c - tr_a) as f64;
                    let s = r_span / q_span;

                    if s < scale_min - eps_scale || s > scale_max + eps_scale {
                        continue;
                    }

                    let b = tr_a as f64 - s * q_ta as f64;

                    let s_bin = ((s - scale_min) / scale_per_bin)
                        .clamp(0.0, (cfg.scale_bins - 1) as f64)
                        as u32;
                    // Offset bin at 1-frame granularity. Binning at
                    // `tol`-frame granularity would combine the ±tol-bin
                    // consolidation into an effective ±tol²-frame window
                    // (and quantize the reported offset to `tol` frames);
                    // keeping 1-frame keys makes consolidation ±tol frames
                    // exactly, matching `WangMatcher`'s semantics.
                    let off_key = b.round() as i64;

                    *acc.entry((s_bin, off_key)).or_insert(0) += 1;
                    pairs.push((q_ta as f64, tr_a as f64));
                }
            }
        }

        if acc.is_empty() {
            return MatchResult::NONE;
        }

        // --- 3. Find peak in 2-D accumulator ---
        // Consolidate each candidate bin by summing votes in its ±tol
        // neighbourhood (scale: ±1 bin, offset: ±offset_tolerance).
        // Snapshot the accumulator ONCE — every subsequent step (peak
        // finding, prominence, bin lookup) reads from this single
        // ordering, so there is no possibility of positional mismatch.
        //
        // `consolidated[i]` holds the neighbourhood-summed vote count for
        // the same bin as `acc_vec[i]`. Prominence is then computed on
        // `consolidated` (not raw bin values), matching the
        // neighbourhood-aware peak selection and the Wang gold standard
        // (audit B5).
        let tol_i64 = tol;
        let mut acc_vec: Vec<((u32, i64), u32)> = acc.iter().map(|(&k, &v)| (k, v)).collect();
        // Sort by (s_bin, off_key) so we can use a sliding window on
        // the offset dimension within each scale-bin neighbourhood.
        acc_vec.sort_unstable_by_key(|&((s, o), _)| (s, o));

        let mut consolidated: Vec<u32> = vec![0u32; acc_vec.len()];
        let mut peak_votes = 0u32;
        let mut peak_linear_idx = 0usize;

        for (i, &((s_bin, off_key), _)) in acc_vec.iter().enumerate() {
            let mut neigh_votes = 0u32;
            // Since acc_vec is sorted by (s_bin, off_key), bins with
            // ds <= 1 are clustered. Scan forward/backward from i to
            // find neighbours within the scale ± 1 and offset ± tol
            // window. This is O(W) per element where W is the
            // neighbourhood size (typically small), giving O(B·W) total
            // instead of O(B²).
            //
            // Scan backward from i.
            let mut j = i;
            loop {
                if j == 0 {
                    break;
                }
                j -= 1;
                let ((ns, no), v) = acc_vec[j];
                if s_bin.saturating_sub(ns) > 1 {
                    break;
                }
                if ns.abs_diff(s_bin) <= 1 && (no - off_key).abs() <= tol_i64 {
                    neigh_votes += v;
                }
            }
            // Centre element.
            neigh_votes += acc_vec[i].1;
            // Scan forward from i.
            for &((ns, no), v) in &acc_vec[(i + 1)..] {
                if ns.saturating_sub(s_bin) > 1 {
                    break;
                }
                if ns.abs_diff(s_bin) <= 1 && (no - off_key).abs() <= tol_i64 {
                    neigh_votes += v;
                }
            }
            consolidated[i] = neigh_votes;
            if neigh_votes > peak_votes {
                peak_votes = neigh_votes;
                peak_linear_idx = i;
            }
        }

        if peak_votes < cfg.min_votes {
            return MatchResult::NONE;
        }

        // --- 4. Prominence ---
        // Computed on the consolidated histogram so a peak whose true mass
        // is spread across neighbouring bins (framing / scale jitter) is
        // not understated relative to its background. `peak_linear_idx`
        // indexes both `acc_vec` and `consolidated` consistently.
        let (peak_s_bin, peak_off_key) = acc_vec[peak_linear_idx].0;
        let prominence = compute_prominence(&consolidated, peak_linear_idx);
        if prominence < cfg.min_prominence {
            return MatchResult::NONE;
        }

        // Coarse scale from the peak bin centre; offset straight from
        // the 1-frame-granularity peak key.
        let coarse_s = scale_min + (peak_s_bin as f64 + 0.5) * scale_per_bin;
        let coarse_b = peak_off_key as f64;

        // --- 5. RANSAC refinement ---
        // Reuses the (t_q, t_r) pairs gathered during voting — no second
        // scan over the index. When RANSAC finds no inliers (degenerate
        // input), fall back to the coarse Hough result.
        let (final_scale, final_offset, votes) = if cfg.ransac_refine {
            let (s, b, n) = ransac_refine(&pairs, coarse_s as f32, coarse_b, tol_i64, cfg);
            if n > 0 {
                (s, b, n)
            } else {
                (coarse_s as f32, coarse_b, peak_votes)
            }
        } else {
            (coarse_s as f32, coarse_b, peak_votes)
        };

        // --- 6. Score ---
        let denom = query.hashes.len().max(1) as f32;
        let score = clamp_score(votes as f32 / denom);

        if score < cfg.min_score || votes < cfg.min_votes {
            return MatchResult::NONE;
        }

        let offset = TimeOffset::from_frames(final_offset.round() as i64, reference.frames_per_sec);

        // The model `t_ref = s·t_query + b` yields s = ref/query frame
        // rate. The public contract on `MatchResult::time_scale` is
        // query/reference duration, so report the reciprocal. Clamped
        // to a wider [0.5, 2.0] range than the search grid so genuine
        // large tempo changes aren't silently saturated.
        let time_scale = if final_scale.abs() > 1e-6 {
            (1.0 / final_scale).clamp(0.5, 2.0)
        } else {
            1.0
        };

        MatchResult {
            is_match: true,
            score,
            votes,
            prominence,
            offset,
            time_scale,
        }
    }
}

/// Normalize a [`PanakoMatchConfig`] so the Hough grid is always
/// well-formed.
///
/// A degenerate scale grid (`scale_bins == 0`, non-finite bounds, or
/// `scale_max ≤ scale_min`) used to be caught only by a `debug_assert`
/// — in release builds it silently produced `scale_per_bin` = 0 / ∞ /
/// NaN, collapsing every vote into saturated bins and yielding garbage
/// results. Normalization replaces those fields with the defaults so
/// every build behaves identically; well-formed configs pass through
/// untouched.
pub(crate) fn normalize_config(mut cfg: PanakoMatchConfig) -> PanakoMatchConfig {
    const DEFAULT_SCALE_MIN: f32 = 0.80;
    const DEFAULT_SCALE_MAX: f32 = 1.25;
    const DEFAULT_SCALE_BINS: u32 = 24;

    if cfg.scale_bins == 0 {
        cfg.scale_bins = DEFAULT_SCALE_BINS;
    }
    if !(cfg.scale_min.is_finite() && cfg.scale_max.is_finite() && cfg.scale_max > cfg.scale_min) {
        cfg.scale_min = DEFAULT_SCALE_MIN;
        cfg.scale_max = DEFAULT_SCALE_MAX;
    }
    cfg
}

/// Iterative RANSAC over the `(t_query, t_ref)` anchor pairs collected
/// during Hough voting. Samples 2 pairs, fits `t_ref = s·t_query + b`,
/// counts inliers within `tol_frames`, and keeps the best fit.
///
/// Returns `(scale, offset, inlier_count)`. When fewer than 2 pairs are
/// available or no valid model is found, returns `(coarse_s, coarse_b, 0)`
/// Count inliers for a candidate line `t_ref = s * t_query + b`.
/// Written as a tight loop without per-element branching so LLVM can
/// auto-vectorize the abs-diff + compare into SIMD instructions.
#[inline]
fn count_inliers_wide(pairs: &[(f64, f64)], s: f64, b: f64, tol: f64) -> u32 {
    let mut count: u32 = 0;
    for &(tq, tr) in pairs {
        let predicted = s * tq + b;
        // Branchless: cast bool to u32 — allows LLVM to vectorize.
        count += ((tr - predicted).abs() <= tol) as u32;
    }
    count
}

/// so the caller can fall back to the Hough peak geometry without mixing
/// the RANSAC score with the Hough vote count (audit B4).
///
/// The pairs are reused from voting — no second scan over the index.
fn ransac_refine(
    pairs: &[(f64, f64)],
    coarse_s: f32,
    coarse_b: f64,
    tol_frames: i64,
    cfg: &PanakoMatchConfig,
) -> (f32, f64, u32) {
    if pairs.len() < 2 {
        return (coarse_s, coarse_b, 0);
    }

    // Inlier tolerance in offset-space (frames). `tol_frames` is the same
    // ±N consolidation window used by the Hough peak, floored at 2 so a
    // single-frame jitter still admits real inliers.
    let inlier_tol = (tol_frames.max(2)) as f64;

    // Budget: at least 4 samples per pair, capped at 128 — more iterations
    // cost time, fewer hurt the inlier estimate on noisy data.
    let n_iter = 128usize.min(pairs.len() * 4);
    // Deterministic seed derived from the data so repeated runs on the
    // same input produce identical results (no_std has no system RNG).
    let seed = pairs.len().wrapping_mul(2_654_435_761) as u64;
    let mut rng = XorShift64(seed.max(1));

    let mut best_inliers = 0u32;
    let mut best_s = coarse_s;
    let mut best_b = coarse_b;

    for _ in 0..n_iter {
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
        if !s.is_finite() || s < cfg.scale_min || s > cfg.scale_max {
            continue;
        }
        let b = tr1 - s as f64 * tq1;

        let inliers = count_inliers_wide(pairs, s as f64, b, inlier_tol);

        if inliers > best_inliers {
            best_inliers = inliers;
            best_s = s;
            best_b = b;
        }
    }

    (best_s, best_b, best_inliers)
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

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
        // Simulate ~10% query speed-up: query timestamps are compressed.
        // Internal s = ref_span / query_span ≈ 1/0.9 ≈ 1.111.
        // Public time_scale = 1/s ≈ 0.90 (query/reference duration).
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
        // ~10% faster query (timestamps compressed by 0.9).
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
        assert!(res.is_match, "tempo-stretched variant must match: {res:?}");
        assert!(
            (res.time_scale - 0.9).abs() < 0.08,
            "expected public time_scale ~0.90 (query/ref), got {}",
            res.time_scale
        );
        assert!(res.score.is_finite() && res.prominence.is_finite());
    }

    #[test]
    fn negative_offset_recovery() {
        // Query starts before reference → negative δ.
        let m = PanakoMatcher::new(PanakoMatchConfig {
            ransac_refine: false,
            ..Default::default()
        });
        let r = make_fp(
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
        let q = make_fp(
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
        let res = m.match_one(&q, &r);
        assert!(res.is_match, "negative-offset match expected: {res:?}");
        assert_eq!(
            res.offset.frames, -50,
            "offset must be -50, got {}",
            res.offset.frames
        );
    }

    #[test]
    fn prominence_clear_spike_exceeds_flat() {
        // Self-match on a sharp alignment must have higher prominence than
        // a near-empty unrelated pair (regression for B1/B5 — prominence
        // from the consolidated peak, not a HashMap-order-dependent bin).
        let m = PanakoMatcher::new(PanakoMatchConfig {
            ransac_refine: false,
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        });
        let spike = make_fp(
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
        let flat_q = make_fp(&[(10, 20, 30), (50, 60, 70)], 9_000);
        let flat_r = make_fp(&[(10, 20, 30), (50, 60, 70)], 0);

        let spike_res = m.match_one(&spike, &spike);
        let flat_res = m.match_one(&flat_q, &flat_r);
        assert!(spike_res.is_match, "self-match must succeed: {spike_res:?}");
        assert!(
            spike_res.prominence > flat_res.prominence,
            "clear spike prominence {} must exceed flat {}",
            spike_res.prominence,
            flat_res.prominence
        );
        assert!(spike_res.prominence.is_finite());
    }

    #[test]
    fn mismatched_frames_per_sec_returns_none() {
        let m = PanakoMatcher::new(PanakoMatchConfig::default());
        let mut a = make_fp(
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
        let b = a.clone();
        a.frames_per_sec = 31.25;
        let res = m.match_one(&a, &b);
        assert_eq!(res, MatchResult::NONE);
    }

    #[test]
    fn ransac_self_match_time_scale_near_one() {
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
        assert!(
            (res.time_scale - 1.0).abs() < 0.05,
            "RANSAC self-match time_scale must be ~1.0, got {}",
            res.time_scale
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

    // ── 1-frame offset-bin granularity (±tol consolidation window) ──
    //
    // Pre-fix, offsets were binned at `tol`-frame granularity and the
    // ±tol-bin consolidation window spanned ±tol² frames, quantizing
    // the reported offset to multiples of `tol`. With RANSAC disabled,
    // a query shifted by exactly 7 frames (plus distant decoys outside
    // the window) must report offset 7, not a quantized 8 (or 0).

    #[test]
    fn hough_offset_is_frame_precise_with_tolerance_4() {
        use crate::classical::PanakoHash;

        // Reference triplets: spans of exactly 10 frames, distinct hashes.
        let mut hashes = alloc::vec::Vec::new();
        for i in 0..10_u32 {
            hashes.push(PanakoHash {
                hash: 1_000 + i,
                t_anchor: 100 + i * 10,
                t_b: 105 + i * 10,
                t_c: 110 + i * 10,
            });
        }
        // Two decoy triplets far outside the ±4 window of the peak.
        hashes.push(PanakoHash {
            hash: 2_000,
            t_anchor: 500,
            t_b: 505,
            t_c: 510,
        });
        hashes.push(PanakoHash {
            hash: 2_001,
            t_anchor: 510,
            t_b: 515,
            t_c: 520,
        });
        let reference = PanakoFingerprint {
            hashes,
            frames_per_sec: 62.5,
        };

        // Query: the same ten triplets shifted by exactly 7 frames
        // (spans preserved → s = 1.0), plus the two decoys at δ=0.
        let mut q_hashes = alloc::vec::Vec::new();
        for i in 0..10_u32 {
            let t = 100 + i * 10 - 7;
            q_hashes.push(PanakoHash {
                hash: 1_000 + i,
                t_anchor: t,
                t_b: t + 5,
                t_c: t + 10,
            });
        }
        q_hashes.push(PanakoHash {
            hash: 2_000,
            t_anchor: 500,
            t_b: 505,
            t_c: 510,
        });
        q_hashes.push(PanakoHash {
            hash: 2_001,
            t_anchor: 510,
            t_b: 515,
            t_c: 520,
        });
        let query = PanakoFingerprint {
            hashes: q_hashes,
            frames_per_sec: 62.5,
        };

        // min_prominence 2: the sparse accumulator (peak bin + one
        // decoy bin) caps prominence at ~3, below the default 5 — the
        // assertion under test is the offset, not the threshold.
        let m = crate::matching::PanakoMatcher::new(crate::matching::PanakoMatchConfig {
            ransac_refine: false,
            offset_tolerance_frames: 4,
            min_votes: 5,
            min_prominence: 2.0,
            ..Default::default()
        });
        let res = m.match_one(&query, &reference);
        assert!(res.is_match, "query must match: {res:?}");
        assert_eq!(
            res.offset.frames, 7,
            "coarse Hough offset must be frame-precise, not quantized to tol"
        );
    }

    // ── Config normalization: degenerate scale grids are repaired at
    //    construction instead of silently corrupting Hough binning. ──

    #[test]
    fn degenerate_scale_config_is_normalized_at_construction() {
        use crate::matching::{PanakoMatchConfig, PanakoMatcher};

        let m = PanakoMatcher::new(PanakoMatchConfig {
            scale_bins: 0,
            ..Default::default()
        });
        assert_eq!(m.config().scale_bins, 24);

        let m = PanakoMatcher::new(PanakoMatchConfig {
            scale_min: 2.0,
            scale_max: 1.0, // inverted range
            ..Default::default()
        });
        assert_eq!(m.config().scale_min, 0.80);
        assert_eq!(m.config().scale_max, 1.25);

        let m = PanakoMatcher::new(PanakoMatchConfig {
            scale_min: f32::NAN,
            ..Default::default()
        });
        assert!(m.config().scale_min.is_finite());
        assert!(m.config().scale_max > m.config().scale_min);

        // And a normalized degenerate config still matches correctly.
        let fp = PanakoFingerprint {
            hashes: (0..6_u32)
                .map(|i| crate::classical::PanakoHash {
                    hash: 100 + i,
                    t_anchor: 10 + i * 10,
                    t_b: 15 + i * 10,
                    t_c: 20 + i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let m = PanakoMatcher::new(PanakoMatchConfig {
            scale_bins: 0,
            min_votes: 3,
            ..Default::default()
        });
        let res = m.match_one(&fp, &fp);
        assert!(
            res.is_match,
            "self-match must survive normalization: {res:?}"
        );
    }

    #[test]
    fn prebuilt_matches_one_to_one_parity() {
        // Same technique as Wang's parity tests: the prebuilt path
        // must produce exactly the same result as the plain path.
        let fp = PanakoFingerprint {
            hashes: (0..20_u32)
                .map(|i| crate::classical::PanakoHash {
                    hash: 500 + i,
                    t_anchor: 100 + i * 10,
                    t_b: 105 + i * 10,
                    t_c: 110 + i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let cfg = PanakoMatchConfig::default();
        let matcher = PanakoMatcher::new(cfg.clone());

        let plain = matcher.match_one(&fp, &fp);
        let index = PanakoRefIndex::build(&fp, &cfg).expect("fp has hashes");
        let prebuilt = matcher.match_one_prebuilt(&fp, &index);

        assert_eq!(plain, prebuilt);
        assert!(plain.is_match);
        assert!(!index.is_empty());
        assert_eq!(index.len(), 20);
    }

    #[test]
    fn prebuilt_empty_reference_returns_none() {
        let empty = PanakoFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let cfg = PanakoMatchConfig::default();
        assert!(PanakoRefIndex::build(&empty, &cfg).is_none());
    }

    #[test]
    fn prebuilt_fps_mismatch_returns_none() {
        let fp = PanakoFingerprint {
            hashes: (0..10_u32)
                .map(|i| crate::classical::PanakoHash {
                    hash: 500 + i,
                    t_anchor: 100 + i * 10,
                    t_b: 105 + i * 10,
                    t_c: 110 + i * 10,
                })
                .collect(),
            frames_per_sec: 62.5,
        };
        let cfg = PanakoMatchConfig::default();
        let index = PanakoRefIndex::build(&fp, &cfg).unwrap();

        let query = PanakoFingerprint {
            hashes: fp.hashes.clone(),
            frames_per_sec: 31.25, // different rate
        };
        let matcher = PanakoMatcher::new(cfg);
        assert_eq!(matcher.match_one_prebuilt(&query, &index), MatchResult::NONE);
    }
}
