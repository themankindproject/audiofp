//! 1:N matching functions and optional transient in-memory indexes.
//!
//! All functions operate on `&[Fingerprint]` slices. The optional
//! `WangIndex`/`HaitsmaIndex`/`PanakoIndex` are matching
//! *accelerators* that hold a combined inverted index in RAM for the
//! lifetime of the struct. They are never serialised, never persisted,
//! and carry no file handles.
//!
//! # Parallel / large 1:N
//!
//! [`match_best`] and [`match_ranked`] are sequential pairwise loops.
//! Matching itself has **no** `rayon` path — the `rayon` feature only
//! parallelises batch *fingerprinting* (`fingerprint_batch_parallel`).
//! For large catalogs, prefer the in-memory index types
//! ([`WangIndex`], [`HaitsmaIndex`], [`PanakoIndex`]) which amortise one
//! inverted-index build across many queries.

use crate::matching::{MatchResult, Matcher};

use alloc::vec;
use alloc::vec::Vec;

use crate::matching::maps::HashMap;

/// Soft cap on votes recorded per reference in [`WangIndex::query`].
///
/// Mirrors [`WangMatcher`](super::WangMatcher)'s dense-histogram
/// `MAX_HIST_BINS` so a pathological query/catalog cannot grow unbounded
/// `Vec`s under hash flooding (audit 67-6).
const MAX_VOTES_PER_REF: usize = 10_000_000;

/// Find the single best-matching reference.
///
/// Returns `Some((index, result))` if any reference clears its
/// decision threshold, or `None` if no reference matched.
///
/// Scans references once and keeps the best `is_match` result (by
/// [`match_result_compare_desc`](crate::matching::match_result_compare_desc)).
/// Stops early when a perfect score (`score >= 1.0`) is found. For the
/// full ranking of every reference use [`match_ranked`]; for large
/// catalogs prefer the index types.
#[must_use]
pub fn match_best<M: Matcher>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Option<(usize, MatchResult)> {
    let mut best: Option<(usize, MatchResult)> = None;
    for (i, r) in refs.iter().enumerate() {
        let result = matcher.match_one(query, r);
        if !result.is_match {
            continue;
        }
        let better = match &best {
            None => true,
            Some((_, b)) => {
                crate::matching::match_result_compare_desc(&result, b) == core::cmp::Ordering::Less
            }
        };
        if better {
            // Perfect hit: no later reference can outrank score 1.0.
            if result.score >= 1.0 {
                return Some((i, result));
            }
            best = Some((i, result));
        }
    }
    best
}

/// Score every reference and return all results sorted by descending
/// score (ties broken by descending prominence).
///
/// Empty reference list → empty result. Unmatched references appear
/// with `is_match == false` — the caller can inspect their scores.
///
/// For large 1:N catalogs, use the in-memory index types
/// ([`WangIndex`], [`HaitsmaIndex`], [`PanakoIndex`]) which amortise
/// the inverted-index build across queries. Pairwise [`match_ranked`]
/// always evaluates every reference.
#[must_use]
pub fn match_ranked<M: Matcher>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Vec<(usize, MatchResult)> {
    let mut results: Vec<(usize, MatchResult)> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, matcher.match_one(query, r)))
        .collect();
    results.sort_by(|a, b| crate::matching::match_result_compare_desc(&a.1, &b.1));
    results
}

// -----------------------------------------------------------------------
// WangIndex — transient in-memory inverted index for 1:N Wang matching
// -----------------------------------------------------------------------

/// An in-memory inverted index over several Wang fingerprints.
///
/// This is a matching **accelerator** — it combines many references into
/// one `hash → Vec<(ref_id, t_anchor)>` map so the per-query cost of
/// 1:N lookups is paid once. It is dropped with its owning scope and is
/// never serialised.
pub struct WangIndex {
    /// Inverted index: hash → list of (reference id, t_anchor).
    map: HashMap<u32, alloc::vec::Vec<(usize, u32)>>,
    /// Frame rates are stored per-reference for offset conversion.
    fps: alloc::vec::Vec<f32>,
}

impl WangIndex {
    /// Build an index from a slice of Wang fingerprints.
    ///
    /// Only hashes appearing in ≤ `max_postings_per_hash` references are
    /// kept (TF-IDF-style stop-hash removal applied globally).
    pub fn build(refs: &[crate::classical::WangFingerprint], max_postings_per_hash: u32) -> Self {
        let mut map: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for h in &fp.hashes {
                map.entry(h.hash).or_default().push((ref_id, h.t_anchor));
            }
        }

        // Remove stop-hashes (appear in too many references)
        map.retain(|_, v| (v.len() as u32) <= max_postings_per_hash);

        Self { map, fps }
    }

    /// Query the index, returning the single best-matching reference.
    ///
    /// Votes are tallied **per reference** (offset `δ = t_ref − t_query`,
    /// matching [`WangMatcher`](super::WangMatcher)), then each candidate
    /// reference is scored independently and the best match that clears
    /// the thresholds is returned as `(ref_id, result)`.
    ///
    /// This is an accelerator: it shares one inverted-index lookup across
    /// all references. Its prominence/score are computed on the sparse
    /// per-reference tally, so they can differ marginally from a direct
    /// [`WangMatcher::match_one`](super::WangMatcher::match_one) call
    /// (which uses a dense, box-consolidated histogram). For exact 1:1
    /// scores use [`match_ranked`].
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::WangFingerprint,
        cfg: &crate::matching::WangMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use crate::matching::{
            TimeOffset, clamp_score, compute_prominence, match_result_compare_desc,
        };

        if query.hashes.is_empty() || self.map.is_empty() {
            return None;
        }

        // Per-reference votes: ref_id → list of (offset δ, query-hash index).
        // Capped at MAX_VOTES_PER_REF so hash flooding cannot OOM (audit 67-6).
        let mut per_ref: HashMap<usize, Vec<(i64, u32)>> = HashMap::new();
        for (qi, h) in query.hashes.iter().enumerate() {
            if let Some(list) = self.map.get(&h.hash) {
                for &(ref_id, tr) in list {
                    let d = tr as i64 - h.t_anchor as i64;
                    let entry = per_ref.entry(ref_id).or_default();
                    if entry.len() < MAX_VOTES_PER_REF {
                        entry.push((d, qi as u32));
                    }
                }
            }
        }

        let tol = cfg.offset_tolerance_frames as i64;
        let q_len = query.hashes.len().max(1) as f32;
        let mut best: Option<(usize, MatchResult)> = None;

        for (&ref_id, votes) in &per_ref {
            // Raw per-offset bin counts (sparse), sorted by offset so
            // all downstream steps (peak search, prominence, plateau
            // selection) are independent of HashMap iteration order
            // (audit 67-1).
            let mut bins: HashMap<i64, u32> = HashMap::new();
            for &(d, _) in votes {
                *bins.entry(d).or_insert(0) += 1;
            }
            let mut bin_vec: Vec<(i64, u32)> = bins.iter().map(|(&d, &c)| (d, c)).collect();
            bin_vec.sort_unstable_by_key(|&(d, _)| d);

            // Consolidated peak with a parallel consolidated-values
            // vector (audit 67-2): peak selection sums ±tol neighbours,
            // so prominence must be computed on the same consolidated
            // values, not on raw bin counts. This matches WangMatcher's
            // prefix-sum consolidation on the dense histogram.
            let mut consolidated: Vec<u32> = vec![0u32; bin_vec.len()];
            let mut peak_votes = 0u32;
            let mut peak_linear_idx = 0usize;

            for (i, &(d0, _)) in bin_vec.iter().enumerate() {
                let mut s = 0u32;
                for &(d, c) in &bin_vec {
                    if (d - d0).abs() <= tol {
                        s += c;
                    }
                }
                consolidated[i] = s;
                if s > peak_votes {
                    peak_votes = s;
                    peak_linear_idx = i;
                }
            }

            // Plateau-centre tie-break: if multiple offsets share the
            // same consolidated peak, pick the median offset of the
            // plateau so the result is deterministic (audit 67-1).
            let peak_off = {
                let plateau: Vec<i64> = bin_vec
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| consolidated[i] == peak_votes)
                    .map(|(_, &(d, _))| d)
                    .collect();
                let mid = plateau.len() / 2;
                plateau
                    .get(mid)
                    .copied()
                    .unwrap_or_else(|| bin_vec.get(peak_linear_idx).map(|&(d, _)| d).unwrap_or(0))
            };

            if peak_votes < cfg.min_votes {
                continue;
            }

            // Prominence on the consolidated histogram (parity with
            // WangMatcher — audit 67-2).
            let prominence = compute_prominence(&consolidated, peak_linear_idx);
            if prominence < cfg.min_prominence {
                continue;
            }

            // Contrib count.
            let mut contrib_bits: HashMap<u32, ()> = HashMap::new();
            for &(d, qi) in votes {
                if (d - peak_off).abs() <= tol {
                    contrib_bits.insert(qi, ());
                }
            }
            let score = clamp_score(contrib_bits.len() as f32 / q_len);
            if score < cfg.min_score {
                continue;
            }

            let fps = self.fps.get(ref_id).copied().unwrap_or(62.5);
            let result = MatchResult {
                is_match: true,
                score,
                votes: peak_votes,
                prominence,
                offset: TimeOffset::from_frames(peak_off, fps),
                time_scale: 1.0,
            };

            let better = match &best {
                None => true,
                Some((_, b)) => match_result_compare_desc(&result, b) == core::cmp::Ordering::Less,
            };
            if better {
                best = Some((ref_id, result));
            }
        }

        best
    }

    /// Return the number of unique hashes in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// -----------------------------------------------------------------------
// HaitsmaIndex — transient in-memory 1:N Haitsma matching accelerator
// -----------------------------------------------------------------------

/// An in-memory inverted index over several Haitsma fingerprints.
///
/// Uses the same sub-fingerprint LUT strategy as
/// [`HaitsmaMatcher`](super::HaitsmaMatcher): build `u32 → Vec<(ref_id,
/// frame_pos)>` per frame, probe each query frame to discover candidate
/// alignments, then verify the best per-reference BER.
pub struct HaitsmaIndex {
    /// Inverted index: 32-bit sub-fingerprint → list of (ref_id, frame_pos).
    lut: HashMap<u32, Vec<(usize, u32)>>,
    /// Per-reference frame slices for BER verification.
    frames: Vec<Vec<u32>>,
    /// Per-reference frame rates for offset conversion.
    fps: Vec<f32>,
}

impl HaitsmaIndex {
    /// Build from a slice of Haitsma fingerprints.
    ///
    /// `max_postings_per_hash` caps the size of each sub-fingerprint's
    /// posting list. Hashes that appear in more than this many positions
    /// (silence / DC / highly repetitive content) are dropped entirely —
    /// the same TF-IDF-style stop-hash pruning used by [`WangIndex`] and
    /// [`PanakoIndex`]. This keeps query-time memory and work bounded on
    /// pathological catalogs (audit B7 / A1).
    pub fn build(
        refs: &[crate::classical::HaitsmaFingerprint],
        max_postings_per_hash: u32,
    ) -> Self {
        let mut lut: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
        let frames: Vec<Vec<u32>> = refs.iter().map(|r| r.frames.clone()).collect();
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for (pos, &frame) in fp.frames.iter().enumerate() {
                lut.entry(frame).or_default().push((ref_id, pos as u32));
            }
        }

        // Stop-hash prune: drop sub-fingerprints whose posting list exceeds
        // the cap. Without this, silence/DC frames produce enormous lists
        // that blow up memory and query time.
        lut.retain(|_, v| (v.len() as u32) <= max_postings_per_hash);

        Self { lut, frames, fps }
    }

    /// Query the index, returning the best-matching `(ref_id, result)`.
    ///
    /// For each query frame, probes the LUT to gather candidate
    /// `(ref_id, delta)` pairs. Each candidate reference is then
    /// verified with the exact-BER path at the best candidate offset.
    ///
    /// Only exact sub-fingerprint matches are probed (no bit-flips in
    /// the index path — use [`match_ranked`] with explicit
    /// `probe_bit_flips` when recall under codec distortion matters).
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::HaitsmaFingerprint,
        cfg: &crate::matching::HaitsmaMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use super::haitsma::{hamming_at_offset, overlap_at};
        use crate::matching::match_result_compare_desc;

        let q_frames = &query.frames;
        let q_len = q_frames.len();
        if q_len == 0 || self.lut.is_empty() {
            return None;
        }

        let min_overlap = cfg.min_overlap_frames as usize;

        // 1. Gather candidate (ref_id, delta) pairs from LUT probes.
        //    Track how many query frames hit each candidate so we can
        //    pick the best one (heuristic for the BER path).
        let mut candidates: HashMap<(usize, i64), u32> = HashMap::new();

        for (q_pos, &q_frame) in q_frames.iter().enumerate() {
            if let Some(list) = self.lut.get(&q_frame) {
                for &(ref_id, r_pos) in list {
                    let delta = r_pos as i64 - q_pos as i64;
                    let overlap = overlap_at(q_len, self.frames[ref_id].len(), delta);
                    if overlap >= min_overlap {
                        *candidates.entry((ref_id, delta)).or_insert(0) += 1;
                    }
                }
            }
        }

        if candidates.is_empty() {
            return None;
        }

        // 2. For each candidate reference, take the best δ and run exact BER.
        let mut best: Option<(usize, MatchResult)> = None;

        // Group candidates by ref_id and pick the top δ per reference.
        // On tied hit counts, prefer the δ with the smallest absolute
        // value (closest to zero alignment) so the winner is independent
        // of HashMap iteration order (audit B8).
        let mut per_ref: HashMap<usize, (i64, u32)> = HashMap::new();
        for (&(ref_id, delta), &hits) in &candidates {
            let entry = per_ref.entry(ref_id).or_insert((delta, hits));
            let better = hits > entry.1 || (hits == entry.1 && delta.abs() < entry.0.abs());
            if better {
                entry.0 = delta;
                entry.1 = hits;
            }
        }

        for (ref_id, (delta, _hits)) in per_ref {
            let r_frames = &self.frames[ref_id];
            let overlap = overlap_at(q_len, r_frames.len(), delta);

            if overlap < min_overlap {
                continue;
            }

            let exact_hamming = hamming_at_offset(q_frames, r_frames, delta, overlap, u64::MAX);

            let total_bits = (overlap * 32) as u64;
            let ber = if total_bits > 0 {
                exact_hamming as f32 / total_bits as f32
            } else {
                1.0
            };

            let score = crate::matching::clamp_score(1.0 - ber);
            let is_match = ber <= cfg.max_ber && (overlap as u32) >= cfg.min_overlap_frames;

            // Prominence: approximate — compare the winning BER against
            // what you'd expect from random alignment (~0.5).
            let prominence = if ber > 1e-6 { 0.5 / ber } else { 100.0 };

            let offset = crate::matching::TimeOffset::from_frames(delta, self.fps[ref_id]);

            let result = MatchResult {
                is_match,
                score,
                votes: overlap as u32,
                prominence,
                offset,
                time_scale: 1.0,
            };

            if !is_match {
                continue;
            }

            let better = match &best {
                None => true,
                Some((_, b)) => match_result_compare_desc(&result, b) == core::cmp::Ordering::Less,
            };
            if better {
                best = Some((ref_id, result));
            }
        }

        best
    }

    /// Return the number of unique sub-fingerprints in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lut.len()
    }

    /// Return `true` if the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lut.is_empty()
    }
}

// -----------------------------------------------------------------------
// PanakoIndex — transient in-memory 1:N Panako matching accelerator
// -----------------------------------------------------------------------

/// An in-memory inverted index over several Panako fingerprints.
///
/// Uses the same 2-D Hough accumulator strategy as
/// [`PanakoMatcher`](super::PanakoMatcher) but across the combined
/// reference set for efficient 1:N lookups.
pub struct PanakoIndex {
    /// Inverted index: hash → list of (ref_id, t_a, t_b, t_c).
    map: HashMap<u32, Vec<(usize, u32, u32, u32)>>,
    /// Per-reference frame rates for offset conversion.
    fps: Vec<f32>,
}

impl PanakoIndex {
    /// Build from a slice of Panako fingerprints.
    pub fn build(refs: &[crate::classical::PanakoFingerprint], max_postings_per_hash: u32) -> Self {
        let mut map: HashMap<u32, Vec<(usize, u32, u32, u32)>> = HashMap::new();
        let fps: Vec<f32> = refs.iter().map(|r| r.frames_per_sec).collect();

        for (ref_id, fp) in refs.iter().enumerate() {
            for h in &fp.hashes {
                map.entry(h.hash)
                    .or_default()
                    .push((ref_id, h.t_anchor, h.t_b, h.t_c));
            }
        }

        map.retain(|_, v| (v.len() as u32) <= max_postings_per_hash);

        Self { map, fps }
    }

    /// Query the index, returning the best-matching `(ref_id, result)`.
    ///
    /// Uses a per-reference 2-D Hough accumulator (scale × offset)
    /// across the shared inverted index. Vote tallies are per-reference;
    /// the best reference that clears thresholds wins.
    #[must_use]
    pub fn query(
        &self,
        query: &crate::classical::PanakoFingerprint,
        cfg: &crate::matching::PanakoMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        use super::panako::validate_config;
        use crate::matching::{
            TimeOffset, clamp_score, compute_prominence, match_result_compare_desc,
        };

        validate_config(cfg);

        if query.hashes.is_empty() || self.map.is_empty() {
            return None;
        }

        let scale_min = cfg.scale_min as f64;
        let scale_max = cfg.scale_max as f64;
        let scale_per_bin = (scale_max - scale_min) / cfg.scale_bins as f64;
        let eps_scale = scale_per_bin * 0.5;
        let tol = cfg.offset_tolerance_frames as i64;
        let q_len = query.hashes.len().max(1) as f32;

        // Build per-reference sparse accumulator across ALL query hashes.
        let mut acc: HashMap<usize, HashMap<(u32, i64), u32>> = HashMap::new();

        for h in &query.hashes {
            if let Some(list) = self.map.get(&h.hash) {
                for &(ref_id, tr_a, _tr_b, tr_c) in list {
                    let q_span = (h.t_c - h.t_anchor).max(1) as f64;
                    let r_span = (tr_c - tr_a) as f64;
                    let s = r_span / q_span;

                    if s < scale_min - eps_scale || s > scale_max + eps_scale {
                        continue;
                    }

                    let b = tr_a as f64 - s * h.t_anchor as f64;
                    let s_bin = ((s - scale_min) / scale_per_bin)
                        .clamp(0.0, (cfg.scale_bins - 1) as f64)
                        as u32;
                    let off_key = (b / (tol.max(1)) as f64).round() as i64;

                    *acc.entry(ref_id)
                        .or_default()
                        .entry((s_bin, off_key))
                        .or_insert(0) += 1;
                }
            }
        }

        let mut best: Option<(usize, MatchResult)> = None;

        for (&ref_id, bins) in &acc {
            let bin_vec: Vec<((u32, i64), u32)> = bins.iter().map(|(&k, &v)| (k, v)).collect();

            // Find peak bin via neighbourhood consolidation. Build a
            // consolidated-values vector aligned with `bin_vec` so
            // prominence is computed on the same neighbourhood-summed
            // values that selected the peak (parity with PanakoMatcher
            // and WangMatcher — audit B5).
            let mut consolidated: Vec<u32> = vec![0u32; bin_vec.len()];
            let mut peak_votes = 0u32;
            let mut peak_s_bin: u32 = 0;
            let mut peak_off_key: i64 = 0;

            for (i, &((s_bin, off_key), _)) in bin_vec.iter().enumerate() {
                let mut neigh = 0u32;
                for &((ns, no), v) in &bin_vec {
                    let ds = ns.abs_diff(s_bin);
                    if ds <= 1 && (no - off_key).abs() <= tol {
                        neigh += v;
                    }
                }
                consolidated[i] = neigh;
                if neigh > peak_votes {
                    peak_votes = neigh;
                    peak_s_bin = s_bin;
                    peak_off_key = off_key;
                }
            }

            if peak_votes < cfg.min_votes {
                continue;
            }

            // Prominence on the consolidated histogram. Find the index of
            // the chosen peak in `bin_vec` so we pass the right offset to
            // `compute_prominence`.
            let peak_linear_idx = bin_vec
                .iter()
                .position(|&((s, o), _)| s == peak_s_bin && o == peak_off_key)
                .unwrap_or(0);
            let prominence = compute_prominence(&consolidated, peak_linear_idx);
            if prominence < cfg.min_prominence {
                continue;
            }

            let score = clamp_score(peak_votes as f32 / q_len);
            if score < cfg.min_score {
                continue;
            }

            let coarse_s = scale_min + (peak_s_bin as f64 + 0.5) * scale_per_bin;
            let coarse_b = peak_off_key as f64 * (tol.max(1)) as f64;

            // Public contract: `time_scale = query_duration /
            // reference_duration` = `1 / s` where `s = ref_span /
            // query_span` is the internal Hough scale (audit B3).
            let time_scale = if coarse_s.abs() > 1e-6 {
                (1.0 / coarse_s).clamp(0.5, 2.0) as f32
            } else {
                1.0
            };

            let fps = self.fps.get(ref_id).copied().unwrap_or(62.5);
            let result = MatchResult {
                is_match: true,
                score,
                votes: peak_votes,
                prominence,
                offset: TimeOffset::from_frames(coarse_b.round() as i64, fps),
                time_scale,
            };

            let better = match &best {
                None => true,
                Some((_, b)) => match_result_compare_desc(&result, b) == core::cmp::Ordering::Less,
            };
            if better {
                best = Some((ref_id, result));
            }
        }

        best
    }

    /// Return the number of unique hashes in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Return `true` if the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, WangFingerprint};
    use crate::matching::{HaitsmaMatchConfig, PanakoMatchConfig, WangMatchConfig, WangMatcher};

    // --- match_best / match_ranked ---

    #[test]
    fn match_ranked_empty_refs() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let results = match_ranked(&m, &fp, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn match_ranked_returns_all() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let refs = alloc::vec![fp.clone(), fp.clone(), fp.clone()];
        let results = match_ranked(&m, &fp, &refs);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn match_best_returns_is_match() {
        let cfg = WangMatchConfig {
            min_votes: 0,
            min_score: -1.0,
            min_prominence: -1.0,
            ..Default::default()
        };
        let m = WangMatcher::new(cfg);
        let fp = WangFingerprint {
            hashes: alloc::vec![],
            frames_per_sec: 62.5,
        };
        let refs = alloc::vec![fp.clone()];
        assert!(match_best(&m, &fp, &refs).is_none());
    }

    #[test]
    fn match_ranked_is_sorted_by_score() {
        let cfg = WangMatchConfig::default();
        let m = WangMatcher::new(cfg);

        // Build 50 synthetic refs + query.
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs: Vec<WangFingerprint> = (0..50u32)
            .map(|i| mk(&[10, 20, 30, 40, 50, 60, 70, 80], i + 1))
            .collect();

        let results = match_ranked(&m, &query, &refs);
        assert_eq!(results.len(), 50);
        // Results must be sorted descending by score.
        for w in results.windows(2) {
            let (_, a) = &w[0];
            let (_, b) = &w[1];
            assert!(
                a.score >= b.score || (a.score - b.score).abs() < 1e-6,
                "not sorted: {} (score {}) before {} (score {})",
                w[0].0,
                a.score,
                w[1].0,
                b.score
            );
        }
    }

    /// Build a synthetic Wang fingerprint helper.
    fn mk(anchors: &[u32], hash_offset: u32) -> WangFingerprint {
        use crate::classical::WangHash;
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

    // --- WangIndex ---

    #[test]
    fn wang_index_returns_correct_ref_id() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let r2 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 2_000);
        let refs = alloc::vec![r0, r1, r2.clone()];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        let (id, res) = index
            .query(&r2, &cfg)
            .expect("query identical to reference 2 must match");
        assert_eq!(id, 2, "must identify reference 2");
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
    }

    #[test]
    fn wang_index_recovers_offset_and_ref_id() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[60, 70, 80, 90, 100, 110, 120, 130], 1_000);
        let refs = alloc::vec![r0, r1];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let (id, res) = index.query(&query, &cfg).expect("must match reference 1");
        assert_eq!(id, 1);
        assert_eq!(res.offset.frames, 50);
    }

    #[test]
    fn wang_index_no_match_for_unrelated_query() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs = alloc::vec![r0];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 9_000);
        assert!(index.query(&query, &cfg).is_none());
    }

    #[test]
    fn wang_index_len_and_empty() {
        let refs = alloc::vec![mk(&[10, 20, 30], 0)];
        let index = WangIndex::build(&refs, 100);
        assert!(!index.is_empty());
        assert_eq!(index.len(), 3); // 3 unique hashes (by anchor idx)
    }

    // --- HaitsmaIndex ---

    fn mk_haitsma_fp(frames: &[u32], fps: f32) -> HaitsmaFingerprint {
        HaitsmaFingerprint {
            frames: frames.to_vec(),
            frames_per_sec: fps,
        }
    }

    #[test]
    fn haitsma_index_self_match() {
        let ref_frames: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(0x01010101))
            .collect();
        let fp = mk_haitsma_fp(&ref_frames, 78.125);
        let refs = alloc::vec![fp.clone()];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };

        let (id, res) = index.query(&fp, &cfg).expect("self-match must succeed");
        assert_eq!(id, 0);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
        assert!((res.score - 1.0).abs() < 0.001, "score={}", res.score);
    }

    #[test]
    fn haitsma_index_identifies_correct_ref() {
        let r0: Vec<u32> = (0..600).map(|i| (i as u32).wrapping_mul(7919)).collect();
        let r1: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let r2: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(1_030_301))
            .collect();

        let refs = alloc::vec![
            mk_haitsma_fp(&r0, 78.125),
            mk_haitsma_fp(&r1, 78.125),
            mk_haitsma_fp(&r2, 78.125),
        ];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        };

        let (id, _res) = index.query(&refs[1], &cfg).expect("must find reference 1");
        assert_eq!(id, 1);
    }

    #[test]
    fn haitsma_index_no_match_unrelated() {
        // Use large, well-separated constants so the two sets share
        // no coincidental 32-bit frame values.
        let r0: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(7919).wrapping_add(0xA000_0000))
            .collect();
        let refs = alloc::vec![mk_haitsma_fp(&r0, 78.125)];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig::default();

        let unrelated: Vec<u32> = (0..600)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(0xDEAD_BEEF)
                    .wrapping_add(0x5000_0000)
            })
            .collect();
        let q = mk_haitsma_fp(&unrelated, 78.125);
        assert!(
            index.query(&q, &cfg).is_none(),
            "unrelated query must not match"
        );
    }

    #[test]
    fn haitsma_index_offset_recovery() {
        let ref_frames: Vec<u32> = (0..800).map(|i| (i as u32).wrapping_mul(7919)).collect();
        // Query = reference frames 100..500
        let query_frames: Vec<u32> = ref_frames[100..500].to_vec();
        let refs = alloc::vec![mk_haitsma_fp(&ref_frames, 78.125)];
        let index = HaitsmaIndex::build(&refs, 100);
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 200,
            ..Default::default()
        };

        let (_id, res) = index
            .query(&mk_haitsma_fp(&query_frames, 78.125), &cfg)
            .expect("must find offset sub-sequence");
        assert_eq!(
            res.offset.frames, 100,
            "offset must be +100, got {}",
            res.offset.frames
        );
    }

    #[test]
    fn haitsma_index_empty_refs_returns_none() {
        let index = HaitsmaIndex::build(&[], 100);
        let cfg = HaitsmaMatchConfig::default();
        let q = mk_haitsma_fp(&[1, 2, 3, 4, 5], 78.125);
        assert!(index.query(&q, &cfg).is_none());
        assert!(index.is_empty());
    }

    #[test]
    fn haitsma_index_stop_hash_prune() {
        // max_postings_per_hash = 1: a hash appearing in two refs is dropped.
        let shared = 0xABCD_EF01u32;
        let r0 = mk_haitsma_fp(&[shared, 0x1111_1111, 0x2222_2222], 78.125);
        let r1 = mk_haitsma_fp(&[shared, 0x3333_3333, 0x4444_4444], 78.125);
        let index = HaitsmaIndex::build(&[r0, r1], 1);
        // Only unique frames remain; the shared hash must be pruned.
        assert!(
            index.len() < 5,
            "stop-hash prune should drop the shared posting"
        );
    }

    // --- PanakoIndex ---

    fn mk_panako_fp(triples: &[(u32, u32, u32)], hash_offset: u32) -> PanakoFingerprint {
        use crate::classical::PanakoHash;
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
    fn panako_index_self_match() {
        let fp = mk_panako_fp(
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
        let refs = alloc::vec![fp.clone()];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig::default();

        let (id, res) = index.query(&fp, &cfg).expect("self-match must succeed");
        assert_eq!(id, 0);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0);
        assert!(
            (res.time_scale - 1.0).abs() < 0.1,
            "self-match time_scale must be ~1.0, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_recovers_offset() {
        let r = mk_panako_fp(
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
        let q = mk_panako_fp(
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
        let refs = alloc::vec![r];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            ransac_refine: false,
            ..Default::default()
        };
        let (id, res) = index.query(&q, &cfg).expect("offset match must succeed");
        assert_eq!(id, 0);
        assert_eq!(
            res.offset.frames, 50,
            "offset must be +50, got {}",
            res.offset.frames
        );
        assert!(
            (res.time_scale - 1.0).abs() < 0.1,
            "constant-tempo time_scale must be ~1.0, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_recovers_scale() {
        // Query timestamps compressed by 0.9 → internal s ≈ 1.111,
        // public time_scale ≈ 0.90.
        let normal = mk_panako_fp(
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
        let fast = mk_panako_fp(
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
        let refs = alloc::vec![normal];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            scale_min: 0.85,
            scale_max: 1.20,
            ransac_refine: false,
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };
        let (id, res) = index
            .query(&fast, &cfg)
            .expect("tempo-stretched query must match");
        assert_eq!(id, 0);
        assert!(
            (res.time_scale - 0.9).abs() < 0.08,
            "expected time_scale ~0.90, got {}",
            res.time_scale
        );
    }

    #[test]
    fn panako_index_identifies_correct_ref() {
        // Use enough triples to satisfy min_votes=5.
        let triples: Vec<(u32, u32, u32)> = (0..10u32)
            .map(|i| (i * 40 + 10, i * 40 + 20, i * 40 + 30))
            .collect();
        let r0 = mk_panako_fp(&triples, 0);
        let r1 = mk_panako_fp(&triples, 1_000);
        let refs = alloc::vec![r0.clone(), r1.clone()];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig {
            min_votes: 3,
            min_prominence: 1.0,
            min_score: 0.05,
            ..Default::default()
        };

        let (id, _res) = index.query(&r1, &cfg).expect("must find reference 1");
        assert_eq!(id, 1);
    }

    #[test]
    fn panako_index_no_match_unrelated() {
        let r0 = mk_panako_fp(&[(10, 20, 30), (50, 60, 70)], 0);
        let refs = alloc::vec![r0];
        let index = PanakoIndex::build(&refs, 100);
        let cfg = PanakoMatchConfig::default();
        let q = mk_panako_fp(&[(10, 20, 30), (50, 60, 70)], 9_000);
        assert!(index.query(&q, &cfg).is_none());
    }
}
