//! Calibrated confidence scores: map each matcher's heterogeneous raw
//! evidence onto one shared probability scale.
//!
//! [`MatchResult::score`] lives in incompatible units per algorithm
//! (landmark contrib ratio, `1 − BER`, cosine), and `prominence` has a
//! different formula per matcher — even the two Haitsma paths disagree.
//! Thresholds therefore do not transfer, and fusing matchers (cascades,
//! ensembles, [`similarity`](crate::matching)-style APIs) needs a common
//! scale. This module provides it: one closed-form logistic map per
//! algorithm, each anchored at the measured score gap on the calibration
//! corpus (see `ROBUSTNESS.md`, "Calibrated confidence").
//!
//! # Design honesty
//!
//! The corpus separates perfectly with wide margins (n=12 positives,
//! n=55 negatives), so *fitting* logistic coefficients would be
//! statistically meaningless — perfect separation gives degenerate
//! (infinite) weights. Instead each map is a fixed two-parameter logistic
//! `σ(k·(score − mid))` with `mid` at the measured gap midpoint and `k`
//! set so the gap edges map to ≈0.01/0.99. Two numbers per algorithm,
//! auditable in one glance, refittable by users on their own catalogs
//! with the documented recipe. Constants are versioned with the algorithm
//! name (`WANG_V1_*`): a future `wang-v2` must not silently reuse `v1`
//! calibration.
//!
//! # Non-goals
//!
//! - Raw fields are untouched: [`MatchResult`], configs, thresholds, and
//!   `is_match` semantics are frozen. A caller that never invokes these
//!   maps observes zero change.
//! - `MatchResult::NONE` (score 0) maps to ≈0.01, not exactly 0 — an
//!   honest epsilon, documented at each call site.
//! - The neural map is provisional (no corpus coverage, model-dependent):
//!   it anchors at the matcher's own `min_cosine` with a conservative
//!   slope, and its docs say to refit per deployment.

use crate::matching::MatchResult;

#[inline]
fn logistic(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Wang v1 calibration: score gap (0.001, 0.405) on the corpus →
/// mid 0.20, slope 22 maps the edges to ≈0.011/0.989.
pub const WANG_V1_MID: f32 = 0.20;
/// See [`WANG_V1_MID`].
pub const WANG_V1_SLOPE: f32 = 22.0;

/// Panako v1 calibration: score gap (0.003, 0.480) →
/// mid 0.24, slope 19 maps the edges to ≈0.011/0.989.
pub const PANAKO_V1_MID: f32 = 0.24;
/// See [`PANAKO_V1_MID`].
pub const PANAKO_V1_SLOPE: f32 = 19.0;

/// Haitsma v1 calibration: score (`1 − BER`) gap (0.573, 0.888) →
/// mid 0.73, slope 29 maps the edges to ≈0.010/0.990.
pub const HAITSMA_V1_MID: f32 = 0.73;
/// See [`HAITSMA_V1_MID`].
pub const HAITSMA_V1_SLOPE: f32 = 29.0;

/// Neural calibration slope: deliberately conservative (k=20) — there is
/// no corpus coverage for embeddings, and cosine scales vary by model.
/// The midpoint is NOT fixed here: it anchors at the matcher's own
/// `min_cosine`, so the decision boundary always maps to exactly 0.5.
/// Refit per deployment; see the recalibration recipe in `ROBUSTNESS.md`.
pub const NEURAL_SLOPE: f32 = 20.0;

/// Estimated P(same recording | Wang evidence): logistic map of the
/// contrib-ratio score. Versioned `wang-v1`; see module docs.
#[inline]
#[must_use]
pub fn calibrated_wang(r: &MatchResult) -> f32 {
    logistic(WANG_V1_SLOPE * (r.score - WANG_V1_MID))
}

/// Estimated P(same recording | Panako evidence): logistic map of the
/// RANSAC-inlier-ratio score. Versioned `panako-v1`; see module docs.
#[inline]
#[must_use]
pub fn calibrated_panako(r: &MatchResult) -> f32 {
    logistic(PANAKO_V1_SLOPE * (r.score - PANAKO_V1_MID))
}

/// Estimated P(same recording | Haitsma evidence): logistic map of the
/// `1 − BER` score. Versioned `haitsma-v1`; see module docs.
///
/// Applies to both Haitsma paths (matcher and index): both report
/// `score = 1 − BER`, so the map is shared. Their *prominence* formulas
/// differ (`median_BER/(ber+ε)` vs `0.5/ber`) — that divergence is a
/// bounded calibration error on prominence-gated results, documented
/// here rather than hidden: near the decision boundary the two paths can
/// disagree on `is_match`, and the calibrated value of a result that only
/// one path would emit carries that path's selection bias.
#[inline]
#[must_use]
pub fn calibrated_haitsma(r: &MatchResult) -> f32 {
    logistic(HAITSMA_V1_SLOPE * (r.score - HAITSMA_V1_MID))
}

/// Estimated P(same recording | neural evidence): logistic map of the
/// cosine score anchored at `min_cosine` (the matcher's own decision
/// boundary → exactly 0.5). Provisional slope ([`NEURAL_SLOPE`]): no
/// corpus coverage, model-dependent — refit per deployment.
#[inline]
#[must_use]
pub fn calibrated_neural(r: &MatchResult, min_cosine: f32) -> f32 {
    logistic(NEURAL_SLOPE * (r.score - min_cosine))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::MatchResult;

    fn scored(score: f32) -> MatchResult {
        MatchResult {
            score,
            ..MatchResult::NONE
        }
    }

    #[test]
    fn maps_span_unit_interval_monotonically() {
        for f in [
            calibrated_wang as fn(&MatchResult) -> f32,
            calibrated_panako,
            calibrated_haitsma,
        ] {
            let a = f(&scored(0.0));
            let b = f(&scored(0.5));
            let c = f(&scored(1.0));
            assert!(a < b && b < c, "map must be monotone increasing");
            assert!(a < 0.05 && c > 0.95, "map must span the interval");
        }
    }

    #[test]
    fn gap_edges_map_near_001_099() {
        // Corpus gap edges (ROBUSTNESS.md): neg max → ≈0.01, pos min → ≈0.99.
        assert!((calibrated_wang(&scored(0.001)) - 0.011).abs() < 0.005);
        assert!((calibrated_wang(&scored(0.405)) - 0.989).abs() < 0.005);
        assert!((calibrated_panako(&scored(0.003)) - 0.011).abs() < 0.005);
        assert!((calibrated_panako(&scored(0.480)) - 0.989).abs() < 0.005);
        assert!((calibrated_haitsma(&scored(0.573)) - 0.010).abs() < 0.005);
        assert!((calibrated_haitsma(&scored(0.888)) - 0.990).abs() < 0.005);
    }

    #[test]
    fn none_maps_to_epsilon_not_zero() {
        for f in [
            calibrated_wang as fn(&MatchResult) -> f32,
            calibrated_panako,
            calibrated_haitsma,
        ] {
            let p = f(&MatchResult::NONE);
            assert!(
                p > 0.0 && p < 0.05,
                "NONE must map to a small epsilon, got {p}"
            );
        }
    }

    #[test]
    fn neural_anchors_at_min_cosine() {
        let at_boundary = MatchResult {
            score: 0.80,
            ..MatchResult::NONE
        };
        assert!((calibrated_neural(&at_boundary, 0.80) - 0.5).abs() < 1e-6);
        assert!(calibrated_neural(&scored(0.95), 0.80) > 0.9);
        assert!(calibrated_neural(&scored(0.65), 0.80) < 0.1);
        // Custom config recenters: boundary is always 0.5.
        assert!((calibrated_neural(&scored(0.70), 0.70) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn inherent_methods_delegate_to_free_functions() {
        use crate::matching::{
            HaitsmaIndex, HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoIndex,
            PanakoMatchConfig, PanakoMatcher, WangIndex, WangMatchConfig, WangMatcher,
        };

        let r = scored(0.5);
        let wang = WangMatcher::new(WangMatchConfig::default());
        assert_eq!(wang.calibrated_score(&r), calibrated_wang(&r));
        let panako = PanakoMatcher::new(PanakoMatchConfig::default());
        assert_eq!(panako.calibrated_score(&r), calibrated_panako(&r));
        let haitsma = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
        assert_eq!(haitsma.calibrated_score(&r), calibrated_haitsma(&r));

        assert_eq!(
            WangIndex::build(&[], 100).calibrated_score(&r),
            calibrated_wang(&r)
        );
        assert_eq!(
            HaitsmaIndex::build(&[], 100).calibrated_score(&r),
            calibrated_haitsma(&r)
        );
        assert_eq!(
            PanakoIndex::build(&[], 100).calibrated_score(&r),
            calibrated_panako(&r)
        );
    }

    #[cfg(feature = "neural")]
    #[test]
    fn neural_method_anchors_at_own_min_cosine() {
        use crate::matching::{Matcher, NeuralMatchConfig, NeuralMatcher};

        let cfg = NeuralMatchConfig::default();
        let matcher = NeuralMatcher::new(cfg.clone());
        let at_boundary = MatchResult {
            score: cfg.min_cosine,
            ..MatchResult::NONE
        };
        assert!((matcher.calibrated_score(&at_boundary) - 0.5).abs() < 1e-6);
    }
}
