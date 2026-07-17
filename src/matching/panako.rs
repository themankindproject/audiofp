//! Panako matcher — tempo-invariant 2-D Hough voter.
//!
//! # Status: stub (Phase 3)
//!
//! [`PanakoMatcher::match_one`] always returns [`MatchResult::NONE`].
//! Config fields document the intended API so callers can prepare for
//! [#100](https://github.com/themankindproject/audiofp/issues/100)
//! (2-D Hough over scale×offset + optional RANSAC).
//!
//! Until then, use [`super::WangMatcher`] for constant-tempo identification,
//! or extract Panako hashes for storage only.

use crate::classical::PanakoFingerprint;
use crate::matching::{MatchResult, Matcher};

/// Configuration for [`PanakoMatcher`].
///
/// Fields are accepted and stored but **ignored** while the matcher is a stub.
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
/// **Stub:** always returns [`MatchResult::NONE`] until Phase 3 lands.
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
        let _ = (query, reference, &self.cfg);
        MatchResult::NONE
    }
}
