//! In-memory fingerprint matching and identification.
//!
//! This module provides a [`Matcher`] trait and one algorithm per
//! fingerprinter. All matching is **purely in memory** — there is no
//! persistence, no serialisation format, no database adapter, and no
//! on-disk index. Everything operates on in-memory fingerprints and
//! returns an in-memory [`MatchResult`].
//!
//! # Architecture
//!
//! | Matcher | Fingerprint | Strategy |
//! |---|---|---|
//! | [`WangMatcher`] | [`WangFingerprint`](crate::classical::WangFingerprint) | Offset-histogram voter (Shazam-style) |
//! | [`PanakoMatcher`] | [`PanakoFingerprint`](crate::classical::PanakoFingerprint) | 2-D Hough + optional RANSAC (tempo-invariant) |
//! | [`HaitsmaMatcher`] | [`HaitsmaFingerprint`](crate::classical::HaitsmaFingerprint) | BER sliding + sub-fingerprint LUT |
//! | [`NeuralMatcher`] | [`NeuralFingerprint`](crate::neural::NeuralFingerprint) | Cosine similarity (requires the `neural` feature) |
//!
//! # Quick example
//!
//! ```
//! extern crate alloc;
//! use audiofp::classical::{Wang, WangFingerprint};
//! use audiofp::matching::{WangMatcher, WangMatchConfig, Matcher, MatchResult};
//! use audiofp::{Fingerprinter, SampleRate};
//!
//! let samples: alloc::vec::Vec<f32> = alloc::vec![0.0_f32; 8_000 * 4];
//! let mut wang = Wang::default();

//! let q = wang.extract(&samples, SampleRate::HZ_8000).unwrap();
//! let r = q.clone();
//!
//! let matcher = WangMatcher::new(WangMatchConfig::default());
//! let m = matcher.match_one(&q, &r);
//! if m.is_match {
//!     println!("same recording (score {:.2}, offset {} ms)", m.score, m.offset.ms);
//! }
//! ```

extern crate alloc;

use core::cmp::Ordering;

mod wang;
pub use wang::{WangMatchConfig, WangMatcher};

mod panako;
pub use panako::{PanakoMatchConfig, PanakoMatcher};

mod haitsma;
pub use haitsma::{HaitsmaMatchConfig, HaitsmaMatcher};

#[cfg(feature = "neural")]
mod neural;
#[cfg(feature = "neural")]
pub use neural::{Aggregation, NeuralMatchConfig, NeuralMatcher};

mod index;
pub use index::{HaitsmaIndex, PanakoIndex, WangIndex, match_best, match_ranked};

mod maps;

// ---------------------------------------------------------------------------
// Common types
// ---------------------------------------------------------------------------

/// Signed time offset of the query relative to the reference.
///
/// A **negative** offset means the query begins *before* the reference
/// (the query is longer or starts earlier). A **positive** offset means
/// the query starts *after* the reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TimeOffset {
    /// Offset in reference STFT frames (exact, native precision).
    pub frames: i64,
    /// Offset in milliseconds, derived from `frames` and the
    /// fingerprint's `frames_per_sec`.
    pub ms: i64,
}

impl TimeOffset {
    /// Build from a frame index offset and a reference frame rate.
    #[must_use]
    pub fn from_frames(frames: i64, frames_per_sec: f32) -> Self {
        let ms = (frames as f64 * 1000.0 / frames_per_sec as f64).round() as i64;
        Self { frames, ms }
    }

    /// The zero offset — query and reference are exactly aligned.
    pub const ZERO: TimeOffset = TimeOffset { frames: 0, ms: 0 };
}

/// Outcome of matching a query fingerprint against one reference.
///
/// `MatchResult` is always produced — even degenerate inputs yield
/// `MatchResult::NONE` rather than an error.  The `is_match` field tells
/// whether the score cleared the matcher's decision threshold.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MatchResult {
    /// `true` iff the score cleared the configured decision threshold.
    pub is_match: bool,
    /// Normalised confidence in `[0.0, 1.0]`, algorithm-agnostic
    /// (higher = more likely the same recording).
    pub score: f32,
    /// Raw aligned-evidence count (landmark votes, aligned Haitsma
    /// frames, or RANSAC inlier count).
    pub votes: u32,
    /// Peak prominence: peak evidence divided by background floor.
    ///
    /// **High** (≫ 1) → sharp spike, a true alignment. **~1** → flat
    /// histogram, random collisions. This is the primary false-positive
    /// guard.
    ///
    /// # Semantics differ by matcher
    ///
    /// - **Wang / Panako**: `peak_consolidated / (mean_rest + 1)` on the
    ///   offset or `(scale, offset)` histogram.
    /// - **Haitsma** (matcher and index): `0.5 / ber` — a BER-derived
    ///   proxy, **not** comparable to Wang/Panako prominence.
    /// - **Neural**: relative cosine excess over other lag positions
    ///   (SlidingMax) or `1.0` (Centroid / DTW).
    ///
    /// Do not threshold prominence with a single constant across matcher
    /// types.
    pub prominence: f32,
    /// Estimated alignment of the query within the reference.
    pub offset: TimeOffset,
    /// Estimated time-scale: `query_duration / reference_duration`.
    ///
    /// 1.0 for algorithms with no tempo model (Wang, Haitsma, neural).
    /// Panako reports the reciprocal of the internal Hough/RANSAC scale
    /// `s = ref_span / query_span` (i.e. `1/s`), clamped to `[0.5, 2.0]`.
    /// Note that [`PanakoMatchConfig::scale_min`] /
    /// [`PanakoMatchConfig::scale_max`] bound the *internal* search grid
    /// for `s`, not this public reciprocal.
    pub time_scale: f32,
}

impl MatchResult {
    /// Constant to return when no meaningful alignment exists.
    pub const NONE: MatchResult = MatchResult {
        is_match: false,
        score: 0.0,
        votes: 0,
        prominence: 0.0,
        offset: TimeOffset::ZERO,
        time_scale: 1.0,
    };
}

// ---------------------------------------------------------------------------
// The Matcher trait
// ---------------------------------------------------------------------------

/// One matcher per fingerprinting algorithm.
///
/// Implementors are **infallible** — empty or degenerate fingerprints
/// yield [`MatchResult::NONE`], never an error.
pub trait Matcher {
    /// The fingerprint type consumed by this matcher.
    type Fingerprint;

    /// Configuration for this matcher.
    type Config: Clone + Send + Sync;

    /// Build a new matcher with the given configuration.
    fn new(cfg: Self::Config) -> Self;

    /// Return a reference to the matcher's configuration.
    fn config(&self) -> &Self::Config;

    /// Match a query against a **single** reference.
    ///
    /// This is the core 1:1 operation. Both fingerprints **must** come from
    /// the same fingerprinter configuration (same `frames_per_sec`).
    ///
    /// # Frame-rate mismatch
    ///
    /// Both fingerprints **must** share the same `frames_per_sec`. A
    /// mismatch returns [`MatchResult::NONE`] in all builds (debug and
    /// release) rather than converting offsets with the wrong rate.
    fn match_one(&self, query: &Self::Fingerprint, reference: &Self::Fingerprint) -> MatchResult;
}

/// Relative tolerance for comparing fingerprint frame rates.
const FPS_REL_EPS: f32 = 1e-3;

/// Return `true` when two frame rates are compatible for matching.
///
/// Rates must be finite and positive, and agree within a relative
/// tolerance (or both be near zero). Used by every matcher to soft-fail
/// mismatched fingerprints in release builds (audit 67-5).
#[inline]
#[must_use]
pub(crate) fn frames_per_sec_compatible(a: f32, b: f32) -> bool {
    if !(a.is_finite() && b.is_finite()) || a <= 0.0 || b <= 0.0 {
        return false;
    }
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= FPS_REL_EPS * scale
}

// ---------------------------------------------------------------------------
// Score ordering helpers
// ---------------------------------------------------------------------------

/// Compare two `f32` scores safely (NaN-proof).
///
/// Uses `partial_cmp` with `unwrap_or(Ordering::Equal)` so NaN values
/// don't cause a panic during sorting. For ranking, you almost always
/// want `score_compare_desc`.
#[inline]
#[must_use]
pub fn score_compare(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// Compare two `MatchResult`s in descending order of quality.
///
/// Primary key: score (desc).  Secondary key: prominence (desc).
/// This matches the ranking convention used by `match_ranked`.
#[inline]
#[must_use]
pub fn match_result_compare_desc(a: &MatchResult, b: &MatchResult) -> Ordering {
    match score_compare(b.score, a.score) {
        Ordering::Equal => score_compare(b.prominence, a.prominence),
        other => other,
    }
}

/// Clamp a score into `[0.0, 1.0]`.
#[inline]
#[must_use]
pub fn clamp_score(s: f32) -> f32 {
    s.clamp(0.0, 1.0)
}

/// Compute prominence: `peak / (mean_of_rest + ε)`.
///
/// `values` is the raw histogram (or Hough accumulator) slice. `peak_idx`
/// is the index of the peak bin. Returns `peak as f32 / mean(rest)` or a
/// large sentinel when there is no background.
#[inline]
#[must_use]
pub fn compute_prominence(values: &[u32], peak_idx: usize) -> f32 {
    let peak = values[peak_idx] as f32;
    if peak == 0.0 {
        return 0.0;
    }
    let sum: u64 = values
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != peak_idx)
        .map(|(_, &v)| v as u64)
        .sum();
    let rest_count = (values.len() - 1) as f32;
    let mean_rest = if rest_count > 0.0 {
        sum as f32 / rest_count
    } else {
        0.0
    };
    peak / (mean_rest + 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_offset_zero() {
        assert_eq!(TimeOffset::ZERO.frames, 0);
        assert_eq!(TimeOffset::ZERO.ms, 0);
    }

    #[test]
    fn time_offset_wang_framerate() {
        // Wang: 62.5 fps → 16 ms/frame
        let off = TimeOffset::from_frames(100, 62.5);
        assert_eq!(off.frames, 100);
        assert_eq!(off.ms, 1600); // 100 * 16 ms
    }

    #[test]
    fn time_offset_haitsma_framerate() {
        // Haitsma: 78.125 fps → 12.8 ms/frame
        let off = TimeOffset::from_frames(78, 78.125);
        assert_eq!(off.frames, 78);
        // 78 * 12.8 = 998.4 → rounds to 998
        assert_eq!(off.ms, 998);
    }

    #[test]
    fn time_offset_negative() {
        let off = TimeOffset::from_frames(-50, 62.5);
        assert_eq!(off.frames, -50);
        assert_eq!(off.ms, -800);
    }

    #[test]
    fn match_result_none_is_default() {
        let n = MatchResult::NONE;
        assert!(!n.is_match);
        assert_eq!(n.score, 0.0);
        assert_eq!(n.votes, 0);
        assert_eq!(n.offset, TimeOffset::ZERO);
    }

    #[test]
    fn score_compare_nan_is_equal() {
        assert_eq!(score_compare(f32::NAN, f32::NAN), Ordering::Equal);
        assert_eq!(score_compare(f32::NAN, 1.0), Ordering::Equal);
    }

    #[test]
    fn clamp_score_bounds() {
        assert_eq!(clamp_score(-0.5), 0.0);
        assert_eq!(clamp_score(0.5), 0.5);
        assert_eq!(clamp_score(1.5), 1.0);
    }

    #[test]
    fn compute_prominence_clear_spike() {
        // 10 bins, one spike at index 3
        let hist = [1, 1, 1, 100, 1, 1, 1, 1, 1, 1];
        let p = compute_prominence(&hist, 3);
        assert!(p > 5.0, "expected high prominence, got {p}");
    }

    #[test]
    fn compute_prominence_flat() {
        let hist = [5, 5, 5, 5, 5];
        let p = compute_prominence(&hist, 2);
        assert!(
            p < 2.0,
            "expected low prominence for flat histogram, got {p}"
        );
    }

    #[test]
    fn match_result_compare_desc_ranks_by_score() {
        let a = MatchResult {
            score: 0.8,
            prominence: 1.0,
            ..MatchResult::NONE
        };
        let b = MatchResult {
            score: 0.3,
            prominence: 100.0,
            ..MatchResult::NONE
        };
        assert_eq!(match_result_compare_desc(&a, &b), Ordering::Less); // a scored higher → Less in desc
    }

    #[test]
    fn frames_per_sec_compatible_accepts_equal() {
        assert!(frames_per_sec_compatible(62.5, 62.5));
        assert!(frames_per_sec_compatible(78.125, 78.125));
    }

    #[test]
    fn frames_per_sec_compatible_rejects_mismatch() {
        assert!(!frames_per_sec_compatible(62.5, 31.25));
        assert!(!frames_per_sec_compatible(62.5, f32::NAN));
        assert!(!frames_per_sec_compatible(-1.0, 62.5));
        assert!(!frames_per_sec_compatible(0.0, 62.5));
    }
}
