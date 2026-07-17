//! 1:N matching functions and optional transient in-memory indexes.
//!
//! All functions operate on `&[Fingerprint]` slices. The optional
//! `WangIndex`/`PanakoIndex` are matching *accelerators* that hold a
//! combined inverted index in RAM for the lifetime of the struct.
//! They are never serialised, never persisted, and carry no file
//! handles.

use crate::matching::{MatchResult, Matcher};

use alloc::vec::Vec;

use crate::matching::maps::HashMap;

/// Find the single best-matching reference.
///
/// Returns `Some((index, result))` if any reference clears its
/// decision threshold, or `None` if no reference matched.
///
/// Callers can pair this with `match_ranked` to see the full ranking
/// even when the best reference barely failed.
#[must_use]
pub fn match_best<M: Matcher>(
    matcher: &M,
    query: &M::Fingerprint,
    refs: &[M::Fingerprint],
) -> Option<(usize, MatchResult)> {
    match_ranked(matcher, query, refs)
        .into_iter()
        .find(|(_, r)| r.is_match)
}

/// Score every reference and return all results sorted by descending
/// score (ties broken by descending prominence).
///
/// Empty reference list → empty result. Unmatched references appear
/// with `is_match == false` — the caller can inspect their scores.
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
        use crate::matching::{TimeOffset, clamp_score, match_result_compare_desc};

        if query.hashes.is_empty() || self.map.is_empty() {
            return None;
        }

        // Per-reference votes: ref_id → list of (offset δ, query-hash index).
        // Tracking the query-hash index lets us count *distinct* query
        // landmarks that align (contrib), mirroring `WangMatcher`.
        let mut per_ref: HashMap<usize, Vec<(i64, u32)>> = HashMap::new();
        for (qi, h) in query.hashes.iter().enumerate() {
            if let Some(list) = self.map.get(&h.hash) {
                for &(ref_id, tr) in list {
                    let d = tr as i64 - h.t_anchor as i64;
                    per_ref.entry(ref_id).or_default().push((d, qi as u32));
                }
            }
        }

        let tol = cfg.offset_tolerance_frames as i64;
        let q_len = query.hashes.len().max(1) as f32;
        let mut best: Option<(usize, MatchResult)> = None;

        for (&ref_id, votes) in &per_ref {
            // Raw per-offset bin counts (sparse).
            let mut bins: HashMap<i64, u32> = HashMap::new();
            for &(d, _) in votes {
                *bins.entry(d).or_insert(0) += 1;
            }
            let bin_vec: Vec<(i64, u32)> = bins.iter().map(|(&d, &c)| (d, c)).collect();

            // Consolidated peak: for each candidate offset sum bins within ±tol.
            let mut peak_votes = 0u32;
            let mut peak_off = 0i64;
            for &(d0, _) in &bin_vec {
                let mut s = 0u32;
                for &(d, c) in &bin_vec {
                    if (d - d0).abs() <= tol {
                        s += c;
                    }
                }
                if s > peak_votes {
                    peak_votes = s;
                    peak_off = d0;
                }
            }
            if peak_votes < cfg.min_votes {
                continue;
            }

            // Prominence: peak ÷ (mean of the raw non-peak bins + 1).
            let peak_bin = bins.get(&peak_off).copied().unwrap_or(0);
            let total: u64 = bin_vec.iter().map(|&(_, c)| c as u64).sum();
            let n_bins = bin_vec.len();
            let mean_rest = if n_bins > 1 {
                (total.saturating_sub(peak_bin as u64)) as f32 / (n_bins - 1) as f32
            } else {
                0.0
            };
            let prominence = peak_votes as f32 / (mean_rest + 1.0);
            if prominence < cfg.min_prominence {
                continue;
            }

            // Contrib: distinct query-hash indices voting within ±tol of peak.
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
}

/// An in-memory inverted index over several Panako fingerprints.
///
/// **Stub (Phase 3 / [#100](https://github.com/themankindproject/audiofp/issues/100)).**
/// `build` accepts fingerprints but stores nothing; `query` always
/// returns `None`. Tempo-invariant 1:N matching is not available yet —
/// use [`WangIndex`] for constant-tempo catalogs, or wait for the
/// Panako matcher implementation.
///
/// See [`WangIndex`] for the intended design once this is filled in.
pub struct PanakoIndex {
    _private: (),
}

impl PanakoIndex {
    /// Build an index from a slice of Panako fingerprints (**no-op stub**).
    pub fn build(
        _refs: &[crate::classical::PanakoFingerprint],
        _max_postings_per_hash: u32,
    ) -> Self {
        Self { _private: () }
    }

    /// Query the index (**always `None` until Phase 3**).
    #[must_use]
    pub fn query(
        &self,
        _query: &crate::classical::PanakoFingerprint,
        _cfg: &crate::matching::PanakoMatchConfig,
    ) -> Option<(usize, MatchResult)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::WangFingerprint;
    use crate::matching::{WangMatchConfig, WangMatcher};

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
        // Empty fingerprints → NONE, so is_match = false
        assert!(match_best(&m, &fp, &refs).is_none());
    }

    /// Build a synthetic Wang fingerprint with distinct hashes per anchor.
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

    #[test]
    fn wang_index_returns_correct_ref_id() {
        // Three references with disjoint hash spaces so a query only
        // collides with its true source.
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let r1 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let r2 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 2_000);
        let refs = alloc::vec![r0, r1, r2.clone()];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        let (id, res) = index
            .query(&r2, &cfg)
            .expect("query identical to reference 2 must match");
        assert_eq!(id, 2, "must identify reference 2, not always 0");
        assert!(res.is_match);
        assert_eq!(res.offset.frames, 0, "self-match offset must be zero");
    }

    #[test]
    fn wang_index_recovers_offset_and_ref_id() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        // Reference 1 sits 50 frames later; a query drawn from it should
        // report ref_id 1 with a +50 offset (query starts after ref).
        let r1 = mk(&[60, 70, 80, 90, 100, 110, 120, 130], 1_000);
        let refs = alloc::vec![r0, r1];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();

        // Query = r1 shifted 50 frames earlier (same hashes as r1).
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 1_000);
        let (id, res) = index.query(&query, &cfg).expect("must match reference 1");
        assert_eq!(id, 1, "must identify reference 1");
        assert_eq!(res.offset.frames, 50, "query starts +50 into reference 1");
    }

    #[test]
    fn wang_index_no_match_for_unrelated_query() {
        let r0 = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 0);
        let refs = alloc::vec![r0];
        let index = WangIndex::build(&refs, 100);
        let cfg = WangMatchConfig::default();
        // Disjoint hash space → no collisions.
        let query = mk(&[10, 20, 30, 40, 50, 60, 70, 80], 9_000);
        assert!(index.query(&query, &cfg).is_none());
    }
}
