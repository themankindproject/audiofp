//! Calibrated confidence scores: map each matcher's heterogeneous raw
//! evidence onto one shared probability scale.
//!
//! [`MatchResult::score`] lives in incompatible units per algorithm
//! (landmark contrib ratio, `1 − BER`), and `prominence` has a
//! different formula per matcher — even the two Haitsma paths disagree.
//! Thresholds therefore do not transfer, and fusing matchers (cascades,
//! ensembles, similarity-style APIs) needs a common scale. This module provides it: one closed-form logistic map per
//! algorithm, each anchored at the measured score gap on the calibration
//! corpus (see `ROBUSTNESS.md`, "Calibrated confidence").
//!
//! # Design honesty
//!
//! The corpus separates perfectly with wide margins (n=24 positives,
//! n=136 negatives), so *fitting* logistic coefficients would be
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
//! - The maps are *not* clamped and are not a shared floor: a
//!   [`MatchResult::NONE`] (score 0) maps to a small value whose magnitude
//!   depends on the algorithm's midpoint and slope — ≈0.012 for Wang,
//!   ≈0.010 for Panako, but ≈6e-10 for Haitsma.
//!   Treat the output as a probability, not as a signal that a match
//!   exists; `is_match` remains the decision. A non-finite `score`
//!   propagates as `NaN`.

use crate::matching::MatchResult;

#[inline]
fn logistic(z: f32) -> f32 {
    // `libm` rather than `f32::exp`: std defers to the platform C library,
    // so the same score could map to a different probability per OS.
    1.0 / (1.0 + libm::expf(-z))
}

/// Wang v1 calibration: score gap (0.001, 0.405) on the corpus →
/// mid 0.20, slope 22 maps the edges to ≈0.011/0.989.
pub const WANG_V1_MID: f32 = 0.20;
/// See [`WANG_V1_MID`].
pub const WANG_V1_SLOPE: f32 = 22.0;

/// Panako v1 calibration: score gap (0.003, 0.480) at fit time →
/// mid 0.24, slope 19 maps the edges to ≈0.011/0.989. The corpus
/// expansion added a weaker positive (acidjazz ogg, 0.394 → ≈0.949);
/// the map is unchanged — see `ROBUSTNESS.md` "Calibrated confidence".
pub const PANAKO_V1_MID: f32 = 0.24;
/// See [`PANAKO_V1_MID`].
pub const PANAKO_V1_SLOPE: f32 = 19.0;

/// Haitsma v1 calibration: score (`1 − BER`) gap (0.573, 0.888) →
/// mid 0.73, slope 29 maps the edges to ≈0.010/0.990.
pub const HAITSMA_V1_MID: f32 = 0.73;
/// See [`HAITSMA_V1_MID`].
pub const HAITSMA_V1_SLOPE: f32 = 29.0;

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
    fn maps_are_bit_exact_everywhere() {
        // `logistic` goes through `libm`, so these bits are the same on
        // every target and OS; `f32::exp` would make them depend on the
        // platform C library.
        let mut h = 0xcbf2_9ce4_8422_2325_u64;
        for i in 0..=1000_u16 {
            let r = scored(f32::from(i) / 1000.0);
            for p in [
                calibrated_wang(&r),
                calibrated_panako(&r),
                calibrated_haitsma(&r),
            ] {
                for b in p.to_bits().to_le_bytes() {
                    h ^= u64::from(b);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        assert_eq!(
            h, 0xf524_101b_481d_fb7d,
            "calibrated outputs changed: {h:#018x}"
        );
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
        // Panako's pos edge moved to 0.394 with the corpus expansion (an
        // acidjazz ogg pair); the shipped map is unchanged, so it now maps
        // to ≈0.949 — pinned here, see ROBUSTNESS.md "Calibrated confidence".
        assert!((calibrated_wang(&scored(0.001)) - 0.011).abs() < 0.005);
        assert!((calibrated_wang(&scored(0.405)) - 0.989).abs() < 0.005);
        assert!((calibrated_panako(&scored(0.003)) - 0.011).abs() < 0.005);
        assert!((calibrated_panako(&scored(0.394)) - 0.949).abs() < 0.005);
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
}
