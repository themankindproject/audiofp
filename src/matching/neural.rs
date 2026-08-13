//! Neural matcher — cosine similarity over embedding vectors.
//!
//! Gated behind `#[cfg(feature = "neural")]` because it depends on
//! [`NeuralFingerprint`](crate::neural::NeuralFingerprint).
//!
//! # Strategy
//!
//! Embeddings are compared with **cosine similarity** (a dot product
//! when the vectors are already L2-normalised, which
//! [`NeuralEmbedderConfig`](crate::neural::NeuralEmbedderConfig)
//! produces by default). Three aggregation modes decide how two
//! *sequences* of window embeddings are reduced to one score:
//!
//! | Mode | Idea | Offset | Cost |
//! |---|---|---|---|
//! | [`Aggregation::Global`] | Mean-pool each side to one vector, single cosine | none (0) | `O((Nq+Nr)·D)` |
//! | [`Aggregation::SlidingMax`] | Slide the shorter sequence across the longer, take the best full-overlap mean cosine | recovered | `O(Nq·Nr·D)` |
//! | [`Aggregation::Dtw`] | Dynamic time warping over cosine distance (tempo-flexible) | none (0) | `O(Nq·Nr·D)` |
//!
//! `min_cosine` is compared against the **raw** best cosine
//! (model-dependent — 0.80 is a nominal default). `score` is that
//! cosine clamped to `[0, 1]`. Offset sign follows the crate-wide
//! convention: positive means the query aligns later into the
//! reference (see [`TimeOffset`](crate::matching::TimeOffset)).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::matching::{MatchResult, Matcher, TimeOffset, clamp_score, frames_per_sec_compatible};
use crate::neural::{NeuralEmbedding, NeuralFingerprint};

/// Aggregation strategy for comparing two embedding sequences.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Aggregation {
    /// Mean-pool both sequences to one vector each, single cosine.
    Global,
    /// Slide the query over the reference, take max mean cosine.
    SlidingMax,
    /// Dynamic time warping for tempo-flexible matching.
    Dtw,
}

/// Configuration for [`NeuralMatcher`].
#[derive(Clone, Debug)]
pub struct NeuralMatchConfig {
    /// Minimum cosine similarity for a positive match. MODEL-DEPENDENT.
    pub min_cosine: f32,
    /// Aggregation strategy. Default [`Aggregation::SlidingMax`].
    pub aggregation: Aggregation,
    /// If `true`, embeddings are assumed L2-normalised (skip re-norm). Default true.
    pub assume_normalized: bool,
}

impl Default for NeuralMatchConfig {
    fn default() -> Self {
        Self {
            min_cosine: 0.80,
            aggregation: Aggregation::SlidingMax,
            assume_normalized: true,
        }
    }
}

/// Offline 1:1 neural matcher (cosine similarity).
pub struct NeuralMatcher {
    cfg: NeuralMatchConfig,
}

impl Matcher for NeuralMatcher {
    type Fingerprint = NeuralFingerprint;
    type Config = NeuralMatchConfig;

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

        let q = &query.embeddings;
        let r = &reference.embeddings;
        if q.is_empty() || r.is_empty() {
            return MatchResult::NONE;
        }

        // Guard: embedding dims must agree and be consistent across all
        // vectors (mismatch → NONE, not an error).
        let dim = query.embedding_dim;
        if dim == 0
            || reference.embedding_dim != dim
            || q.iter().any(|e| e.vector.len() != dim)
            || r.iter().any(|e| e.vector.len() != dim)
        {
            return MatchResult::NONE;
        }

        let assume_norm = self.cfg.assume_normalized;
        let fps = reference.frames_per_sec;

        match self.cfg.aggregation {
            Aggregation::Global => self.match_global(q, r, dim, assume_norm),
            Aggregation::SlidingMax => self.match_sliding(q, r, assume_norm, fps),
            Aggregation::Dtw => self.match_dtw(q, r, assume_norm),
        }
    }
}

impl NeuralMatcher {
    /// Mean-pool both sequences to a single (renormalised) centroid and
    /// take one cosine. Pooling always denormalises, so the centroids
    /// are L2-normalised here regardless of `assume_norm`.
    fn match_global(
        &self,
        q: &[NeuralEmbedding],
        r: &[NeuralEmbedding],
        dim: usize,
        _assume_norm: bool,
    ) -> MatchResult {
        let qc = normalized_centroid(q, dim);
        let rc = normalized_centroid(r, dim);
        // Centroids are L2-normalised → cosine is the dot product.
        let cos = dot(&qc, &rc);
        self.build(
            cos,
            /*prominence=*/ 1.0,
            TimeOffset::ZERO,
            q.len().min(r.len()) as u32,
        )
    }

    /// Slide the shorter sequence across the longer one, comparing only
    /// full-overlap windows, and keep the best mean cosine.
    fn match_sliding(
        &self,
        q: &[NeuralEmbedding],
        r: &[NeuralEmbedding],
        assume_norm: bool,
        fps: f32,
    ) -> MatchResult {
        let query_is_short = q.len() <= r.len();
        let (short, long) = if query_is_short { (q, r) } else { (r, q) };
        let m = short.len();
        let n = long.len();

        let mut best_cos = f32::NEG_INFINITY;
        let mut best_j = 0usize;
        let mut sum_means = 0.0_f32;
        let mut count = 0u32;

        for j in 0..=(n - m) {
            let mut acc = 0.0_f32;
            for i in 0..m {
                acc += cosine(&short[i].vector, &long[j + i].vector, assume_norm);
            }
            let mean = acc / m as f32;
            sum_means += mean;
            count += 1;
            if mean > best_cos {
                best_cos = mean;
                best_j = j;
            }
        }

        // δ = t_ref − t_query in embedding-window units. If the query is
        // the shorter side it aligns at reference window `best_j`
        // (positive). Otherwise the reference is the shorter side and the
        // query started earlier (negative).
        let delta = if query_is_short {
            best_j as i64
        } else {
            -(best_j as i64)
        };

        // Prominence via a +1 shift so cosine's [-1, 1] maps to [0, 2]
        // (positive, ~1 when the histogram is flat).
        let prominence = if count > 1 {
            let mean_others = (sum_means - best_cos) / (count - 1) as f32;
            (best_cos + 1.0) / (mean_others + 1.0)
        } else {
            1.0
        };

        self.build(
            best_cos,
            prominence,
            TimeOffset::from_frames(delta, fps),
            m as u32,
        )
    }

    /// Dynamic time warping over cosine distance (`1 − cos`). Tempo-
    /// flexible: allows non-linear alignment of the two sequences.
    fn match_dtw(
        &self,
        q: &[NeuralEmbedding],
        r: &[NeuralEmbedding],
        assume_norm: bool,
    ) -> MatchResult {
        let m = q.len();
        let n = r.len();

        // Rolling two-row DP. cost = 1 − cosine ∈ [0, 2].
        let mut prev = vec![f32::INFINITY; n + 1];
        let mut curr = vec![f32::INFINITY; n + 1];
        prev[0] = 0.0;

        for i in 1..=m {
            curr[0] = f32::INFINITY;
            for j in 1..=n {
                let dist = 1.0 - cosine(&q[i - 1].vector, &r[j - 1].vector, assume_norm);
                let best_prev = prev[j].min(curr[j - 1]).min(prev[j - 1]);
                curr[j] = dist + best_prev;
            }
            core::mem::swap(&mut prev, &mut curr);
        }

        let total_cost = prev[n];
        // Normalise by the shortest possible warp path (`max(m, n)`) so
        // the mean step cost is comparable to `1 − cos`; recover an
        // effective cosine. (The longest warp path would be
        // `m + n − 1`; using the shortest keeps the normaliser tight
        // and the score in a sensible range — audit 67-4.)
        let path_len = m.max(n) as f32;
        let mean_dist = if path_len > 0.0 {
            total_cost / path_len
        } else {
            2.0
        };
        let equiv_cos = 1.0 - mean_dist;

        // DTW is the one mode with a tempo model: report the
        // query/reference window-count ratio.
        let time_scale = if n > 0 { m as f32 / n as f32 } else { 1.0 };
        let mut result = self.build(equiv_cos, 1.0, TimeOffset::ZERO, m.min(n) as u32);
        result.time_scale = time_scale;
        result
    }

    /// Assemble a [`MatchResult`] from a cosine score.
    ///
    /// Both `cos` and `prominence` are sanitised to finite values so the
    /// public API can never leak Inf/NaN (audit 67-3). `prominence` is
    /// floored at 0 — negative prominence has no useful meaning and a
    /// degenerate `mean_others == -1` could otherwise produce Inf.
    fn build(&self, cos: f32, prominence: f32, offset: TimeOffset, votes: u32) -> MatchResult {
        let cos = if cos.is_finite() { cos } else { 0.0 };
        let prominence = if prominence.is_finite() && prominence >= 0.0 {
            prominence
        } else {
            0.0
        };
        MatchResult {
            is_match: cos >= self.cfg.min_cosine,
            score: clamp_score(cos),
            votes,
            prominence,
            offset,
            time_scale: 1.0,
        }
    }
}

/// Dot product of two equal-length vectors, vectorised 8-wide.
///
/// The hot inner loop of every aggregation mode (SlidingMax runs it
/// `Nq·Nr` times, DTW `Nq·Nr` times), so this is the single most
/// valuable SIMD site in the neural matcher.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for i in 0..chunks {
        let off = i * 8;
        let va = f32x8::new(
            a[off..off + 8]
                .try_into()
                .expect("a chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        let vb = f32x8::new(
            b[off..off + 8]
                .try_into()
                .expect("b chunk is exactly 8 elements: loop iterates n/8 complete chunks"),
        );
        acc = va.mul_add(vb, acc);
    }

    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += a[i] * b[i];
    }
    sum
}

/// Cosine similarity. When `assume_norm` is `true` the inputs are taken
/// to be unit vectors and the dot product is returned directly.
#[inline]
fn cosine(a: &[f32], b: &[f32], assume_norm: bool) -> f32 {
    let d = dot(a, b);
    if assume_norm {
        return d;
    }
    let na = crate::neural::embedder::sumsq_wide(a).sqrt();
    let nb = crate::neural::embedder::sumsq_wide(b).sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        d / (na * nb)
    }
}

/// Mean of all embedding vectors, L2-normalised. Returns a zero vector
/// if the mean has (near-)zero norm.
fn normalized_centroid(embs: &[NeuralEmbedding], dim: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; dim];
    for e in embs {
        for (acc, &v) in c.iter_mut().zip(e.vector.iter()) {
            *acc += v;
        }
    }
    let inv_n = 1.0 / embs.len() as f32;
    for v in c.iter_mut() {
        *v *= inv_n;
    }
    let norm = dot(&c, &c).sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for v in c.iter_mut() {
            *v *= inv;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimestampMs;

    /// Build a normalised embedding from a raw vector.
    fn emb(v: &[f32], t_ms: u64) -> NeuralEmbedding {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        NeuralEmbedding {
            vector: v.iter().map(|x| x / norm).collect(),
            t_start: TimestampMs(t_ms),
        }
    }

    fn fp(embs: Vec<NeuralEmbedding>, dim: usize) -> NeuralFingerprint {
        NeuralFingerprint {
            embeddings: embs,
            embedding_dim: dim,
            frames_per_sec: 1.0,
        }
    }

    #[test]
    fn config_defaults() {
        let c = NeuralMatchConfig::default();
        assert!((c.min_cosine - 0.80).abs() < 1e-6);
        assert_eq!(c.aggregation, Aggregation::SlidingMax);
        assert!(c.assume_normalized);
    }

    #[test]
    fn empty_returns_none() {
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let empty = fp(vec![], 3);
        let one = fp(vec![emb(&[1.0, 0.0, 0.0], 0)], 3);
        assert_eq!(m.match_one(&empty, &one), MatchResult::NONE);
        assert_eq!(m.match_one(&one, &empty), MatchResult::NONE);
    }

    #[test]
    fn dim_mismatch_returns_none() {
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let a = fp(vec![emb(&[1.0, 0.0, 0.0], 0)], 3);
        let b = fp(vec![emb(&[1.0, 0.0], 0)], 2);
        assert_eq!(m.match_one(&a, &b), MatchResult::NONE);
    }

    #[test]
    fn self_match_cosine_one_global() {
        let m = NeuralMatcher::new(NeuralMatchConfig {
            aggregation: Aggregation::Global,
            ..Default::default()
        });
        let f = fp(
            vec![
                emb(&[1.0, 2.0, 3.0], 0),
                emb(&[0.5, 0.1, 0.9], 1000),
                emb(&[0.2, 0.8, 0.4], 2000),
            ],
            3,
        );
        let res = m.match_one(&f, &f);
        assert!(res.is_match, "self-match must be positive");
        assert!(res.score > 0.999, "self-match score ~1: {}", res.score);
    }

    #[test]
    fn self_match_sliding() {
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let f = fp(
            vec![
                emb(&[1.0, 0.0, 0.0], 0),
                emb(&[0.0, 1.0, 0.0], 1000),
                emb(&[0.0, 0.0, 1.0], 2000),
            ],
            3,
        );
        let res = m.match_one(&f, &f);
        assert!(res.is_match);
        assert!(res.score > 0.999, "score {}", res.score);
        assert_eq!(res.offset.frames, 0);
    }

    #[test]
    fn orthogonal_no_match() {
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let a = fp(
            vec![emb(&[1.0, 0.0, 0.0], 0), emb(&[1.0, 0.0, 0.0], 1000)],
            3,
        );
        let b = fp(
            vec![emb(&[0.0, 1.0, 0.0], 0), emb(&[0.0, 1.0, 0.0], 1000)],
            3,
        );
        let res = m.match_one(&a, &b);
        assert!(!res.is_match, "orthogonal embeddings must not match");
        assert!(
            res.score < 0.5,
            "orthogonal score should be low: {}",
            res.score
        );
    }

    #[test]
    fn sliding_offset_recovery_positive() {
        // Query is a 2-window subsequence starting at reference window 2.
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let a = emb(&[1.0, 0.0, 0.0, 0.0], 0);
        let b = emb(&[0.0, 1.0, 0.0, 0.0], 0);
        let c = emb(&[0.0, 0.0, 1.0, 0.0], 0);
        let d = emb(&[0.0, 0.0, 0.0, 1.0], 0);
        let reference = fp(vec![a.clone(), b.clone(), c.clone(), d.clone()], 4);
        let query = fp(vec![c.clone(), d.clone()], 4);
        let res = m.match_one(&query, &reference);
        assert!(res.is_match, "subsequence must match");
        assert_eq!(res.offset.frames, 2, "query starts at reference window 2");
    }

    #[test]
    fn sliding_offset_recovery_negative() {
        // The reference is the shorter side: query starts before it.
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let a = emb(&[1.0, 0.0, 0.0, 0.0], 0);
        let b = emb(&[0.0, 1.0, 0.0, 0.0], 0);
        let c = emb(&[0.0, 0.0, 1.0, 0.0], 0);
        let d = emb(&[0.0, 0.0, 0.0, 1.0], 0);
        let query = fp(vec![a.clone(), b.clone(), c.clone(), d.clone()], 4);
        let reference = fp(vec![c.clone(), d.clone()], 4);
        let res = m.match_one(&query, &reference);
        assert!(res.is_match);
        assert_eq!(res.offset.frames, -2, "reference aligns at query window 2");
    }

    #[test]
    fn dtw_self_match() {
        let m = NeuralMatcher::new(NeuralMatchConfig {
            aggregation: Aggregation::Dtw,
            ..Default::default()
        });
        let f = fp(
            vec![
                emb(&[1.0, 0.0, 0.0], 0),
                emb(&[0.0, 1.0, 0.0], 1000),
                emb(&[0.0, 0.0, 1.0], 2000),
            ],
            3,
        );
        let res = m.match_one(&f, &f);
        assert!(
            res.is_match,
            "DTW self-match must be positive: {}",
            res.score
        );
        assert!(res.score > 0.999, "DTW self-match score ~1: {}", res.score);
    }

    #[test]
    fn dtw_tolerates_time_stretch() {
        // Reference has each window; query repeats windows (a slow-down).
        let m = NeuralMatcher::new(NeuralMatchConfig {
            aggregation: Aggregation::Dtw,
            min_cosine: 0.9,
            ..Default::default()
        });
        let a = emb(&[1.0, 0.0, 0.0], 0);
        let b = emb(&[0.0, 1.0, 0.0], 0);
        let reference = fp(vec![a.clone(), b.clone()], 3);
        // Query = a, a, b (a held twice) — DTW should still align well.
        let query = fp(vec![a.clone(), a.clone(), b.clone()], 3);
        let res = m.match_one(&query, &reference);
        assert!(
            res.is_match,
            "DTW should tolerate the repeat: {}",
            res.score
        );
    }

    #[test]
    fn not_normalized_config_still_works() {
        // Raw (un-normalised) vectors with assume_normalized = false.
        let m = NeuralMatcher::new(NeuralMatchConfig {
            assume_normalized: false,
            aggregation: Aggregation::SlidingMax,
            ..Default::default()
        });
        let raw = |v: &[f32]| NeuralEmbedding {
            vector: v.to_vec(),
            t_start: TimestampMs(0),
        };
        // Same direction, different magnitude → cosine 1.0.
        let a = fp(vec![raw(&[2.0, 0.0, 0.0]), raw(&[0.0, 3.0, 0.0])], 3);
        let b = fp(vec![raw(&[5.0, 0.0, 0.0]), raw(&[0.0, 7.0, 0.0])], 3);
        let res = m.match_one(&a, &b);
        assert!(
            res.is_match,
            "co-directional vectors must match: {}",
            res.score
        );
        assert!(res.score > 0.999);
    }

    #[test]
    fn determinism() {
        let m = NeuralMatcher::new(NeuralMatchConfig::default());
        let a = fp(
            vec![emb(&[0.3, 0.7, 0.1], 0), emb(&[0.9, 0.2, 0.4], 1000)],
            3,
        );
        let b = fp(
            vec![emb(&[0.5, 0.5, 0.2], 0), emb(&[0.1, 0.8, 0.6], 1000)],
            3,
        );
        assert_eq!(m.match_one(&a, &b), m.match_one(&a, &b));
    }
}
