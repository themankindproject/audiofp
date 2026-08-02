//! Haitsma–Kalker matcher — BER sliding window + sub-fingerprint LUT.
//!
//! Haitsma is a dense per-frame 32-bit code; matching is
//! **bit-error-rate (BER) minimisation** over alignments.
//!
//! # Two tiers
//!
//! **Exact BER** — slide query over reference at every offset `δ`,
//! compute Hamming via hardware `POPCNT` with per-frame early-abort.
//! Most offsets die in the first few frames.
//!
//! **Sub-fingerprint LUT** — when `reference.len() > 512`, build a
//! hash map `u32 → Vec<pos>` over reference frames. Haitsma's key
//! property: when `BER < ~0.35`, at least one query frame is bit-exact,
//! so probe each query frame's exact `u32` (plus optional 1–2 bit-flip
//! probes) → candidate offsets → run exact BER verification only there.
//! Turns `O(Q·R)` into `O(Q + candidates·overlap)`.
//!
//! # Performance
//!
//! - Exact: `O(Q·R)` popcounts with aggressive early-abort.
//! - LUT: `O(Q + candidates·overlap)` — sub-millisecond for song-length.
//! - Memory: transient hash map ≈ `r_len*12` bytes (`HashMap` under `std`).

extern crate alloc;

use alloc::vec::Vec;

use crate::classical::HaitsmaFingerprint;
use crate::matching::maps::HashMap;
use crate::matching::{MatchResult, Matcher, TimeOffset, clamp_score, frames_per_sec_compatible};

// ---------------------------------------------------------------------------
// Core BER computation
// ---------------------------------------------------------------------------

/// Hamming distance over `overlap` frames at offset `delta`.
///
/// Returns `u64::MAX` if the cumulative Hamming exceeds `best_sofar`
/// (early-abort). The caller initialises `best_sofar` to `u64::MAX`
/// when no bound exists.
///
/// `delta = t_reference − t_query`, so `delta ≥ 0` means query
/// starts after reference (query[0] ↔ ref[delta]).
#[inline]
pub(crate) fn hamming_at_offset(
    query: &[u32],
    reference: &[u32],
    delta: i64,
    overlap: usize,
    best_sofar: u64,
) -> u64 {
    let q_start = if delta >= 0 {
        0usize
    } else {
        (-delta) as usize
    };
    let r_start = if delta >= 0 { delta as usize } else { 0usize };

    let q_slice = &query[q_start..q_start + overlap];
    let r_slice = &reference[r_start..r_start + overlap];

    let mut hamming: u64 = 0;
    for (qa, ra) in q_slice.iter().zip(r_slice.iter()) {
        // POPCNT on the XOR of two 32-bit sub-fingerprints
        hamming += (qa ^ ra).count_ones() as u64;
        if hamming > best_sofar {
            return u64::MAX;
        }
    }
    hamming
}

/// Overlap length (in frames) at a given `delta`.
#[inline]
pub(crate) fn overlap_at(q_len: usize, r_len: usize, delta: i64) -> usize {
    if delta >= 0 {
        q_len.min(r_len.saturating_sub(delta as usize))
    } else {
        let d_abs = (-delta) as usize;
        q_len.saturating_sub(d_abs).min(r_len)
    }
}

// ---------------------------------------------------------------------------
// Sub-fingerprint LUT
// ---------------------------------------------------------------------------

/// Build a LUT: `u32` sub-fingerprint → list of reference frame indices.
fn build_lut(reference: &[u32]) -> HashMap<u32, Vec<usize>> {
    let mut lut: HashMap<u32, Vec<usize>> = HashMap::new();
    for (pos, &frame) in reference.iter().enumerate() {
        lut.entry(frame).or_default().push(pos);
    }
    lut
}

/// Probe all 1-bit-flip variants of `frame` (32 variants).
#[inline]
fn probe_1flip(frame: u32, lut: &HashMap<u32, Vec<usize>>, f: &mut impl FnMut(&Vec<usize>)) {
    if let Some(v) = lut.get(&frame) {
        f(v);
    }
    for bit in 0..32 {
        if let Some(v) = lut.get(&(frame ^ (1u32 << bit))) {
            f(v);
        }
    }
}

/// Probe exact + 1-bit + 2-bit-flip variants (1 + 32 + 496 = 529 probes).
#[inline]
fn probe_2flip(frame: u32, lut: &HashMap<u32, Vec<usize>>, f: &mut impl FnMut(&Vec<usize>)) {
    probe_1flip(frame, lut, f);
    for b1 in 0..32 {
        for b2 in (b1 + 1)..32 {
            if let Some(v) = lut.get(&(frame ^ (1u32 << b1) ^ (1u32 << b2))) {
                f(v);
            }
        }
    }
}

/// Probe exact only (no bit flips).
#[inline]
fn probe_exact(frame: u32, lut: &HashMap<u32, Vec<usize>>, f: &mut impl FnMut(&Vec<usize>)) {
    if let Some(v) = lut.get(&frame) {
        f(v);
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`HaitsmaMatcher`].
#[derive(Clone, Debug)]
pub struct HaitsmaMatchConfig {
    /// Maximum acceptable bit error rate. Default 0.35 (paper's block
    /// threshold).
    pub max_ber: f32,
    /// Minimum overlapping frames for a decision. Default 256 (~one
    /// sub-fingerprint block at 78.125 fps ≈ 3.3 s).
    pub min_overlap_frames: u32,
    /// Enable sub-fingerprint LUT acceleration for references with
    /// more than 512 frames. Default true.
    pub use_lut: bool,
    /// Bit-flip probes per query frame: 0 = exact only, 1 = +32
    /// single-bit-flip variants, 2 = +496 two-bit-flip variants.
    /// Default 0.
    ///
    /// **Recall caveat:** with `probe_bit_flips = 0` the LUT path only
    /// discovers an alignment when at least one query frame is
    /// *bit-exactly* present in the reference. Under codec/noise
    /// distortion this can miss a true match that the exhaustive
    /// exact-BER path (or a higher `probe_bit_flips`) would find. Raise
    /// this — or set `use_lut = false` — when matching noisy queries;
    /// the LUT and exact paths are only guaranteed to agree when a
    /// bit-exact query frame exists at the true offset.
    pub probe_bit_flips: u8,
}

impl Default for HaitsmaMatchConfig {
    fn default() -> Self {
        Self {
            max_ber: 0.35,
            min_overlap_frames: 256,
            use_lut: true,
            probe_bit_flips: 0,
        }
    }
}

/// Offline 1:1 Haitsma matcher (BER minimisation).
pub struct HaitsmaMatcher {
    cfg: HaitsmaMatchConfig,
}

impl Matcher for HaitsmaMatcher {
    type Fingerprint = HaitsmaFingerprint;
    type Config = HaitsmaMatchConfig;

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

        let q_frames = &query.frames;
        let r_frames = &reference.frames;
        let q_len = q_frames.len();
        let r_len = r_frames.len();

        if q_len == 0 || r_len == 0 {
            return MatchResult::NONE;
        }

        let min_overlap = self.cfg.min_overlap_frames as usize;

        // ---------- LUT path ----------
        let use_lut = self.cfg.use_lut && r_len > 512;
        if use_lut {
            let lut = build_lut(r_frames);
            let mut best_hamming = u64::MAX;
            let mut best_delta: i64 = 0;
            let mut best_overlap: usize = 0;

            // Iterate over query frames, probe LUT for each
            for (q_pos, &q_frame) in q_frames.iter().enumerate() {
                match self.cfg.probe_bit_flips {
                    0 => probe_exact(q_frame, &lut, &mut |positions| {
                        for &r_pos in positions {
                            let delta = r_pos as i64 - q_pos as i64;
                            let overlap = overlap_at(q_len, r_len, delta);
                            if overlap < min_overlap {
                                continue;
                            }
                            let h =
                                hamming_at_offset(q_frames, r_frames, delta, overlap, best_hamming);
                            if h < best_hamming {
                                best_hamming = h;
                                best_delta = delta;
                                best_overlap = overlap;
                            }
                        }
                    }),
                    1 => probe_1flip(q_frame, &lut, &mut |positions| {
                        for &r_pos in positions {
                            let delta = r_pos as i64 - q_pos as i64;
                            let overlap = overlap_at(q_len, r_len, delta);
                            if overlap < min_overlap {
                                continue;
                            }
                            let h =
                                hamming_at_offset(q_frames, r_frames, delta, overlap, best_hamming);
                            if h < best_hamming {
                                best_hamming = h;
                                best_delta = delta;
                                best_overlap = overlap;
                            }
                        }
                    }),
                    _ => probe_2flip(q_frame, &lut, &mut |positions| {
                        for &r_pos in positions {
                            let delta = r_pos as i64 - q_pos as i64;
                            let overlap = overlap_at(q_len, r_len, delta);
                            if overlap < min_overlap {
                                continue;
                            }
                            let h =
                                hamming_at_offset(q_frames, r_frames, delta, overlap, best_hamming);
                            if h < best_hamming {
                                best_hamming = h;
                                best_delta = delta;
                                best_overlap = overlap;
                            }
                        }
                    }),
                }
                // If we already have a perfect match, stop probing
                if best_hamming == 0 {
                    break;
                }
            }

            if best_hamming == u64::MAX {
                return MatchResult::NONE;
            }

            return build_result(
                best_hamming,
                best_delta,
                best_overlap,
                q_frames,
                r_frames,
                query.frames_per_sec,
                &self.cfg,
            );
        }

        // ---------- Exact BER path (scan all offsets) ----------
        let dmin: i64 = -((q_len as i64).saturating_sub(1));
        let dmax: i64 = (r_len as i64).saturating_sub(1);

        let mut best_hamming = u64::MAX;
        let mut best_delta: i64 = 0;
        let mut best_overlap: usize = 0;

        for delta in dmin..=dmax {
            let overlap = overlap_at(q_len, r_len, delta);
            if overlap < min_overlap {
                continue;
            }

            let h = hamming_at_offset(q_frames, r_frames, delta, overlap, best_hamming);
            if h < best_hamming {
                best_hamming = h;
                best_delta = delta;
                best_overlap = overlap;
            }
        }

        if best_hamming == u64::MAX {
            return MatchResult::NONE;
        }

        build_result(
            best_hamming,
            best_delta,
            best_overlap,
            q_frames,
            r_frames,
            query.frames_per_sec,
            &self.cfg,
        )
    }
}

// ---------------------------------------------------------------------------
// Result builder
// ---------------------------------------------------------------------------

fn build_result(
    best_hamming: u64,
    best_delta: i64,
    overlap: usize,
    q_frames: &[u32],
    r_frames: &[u32],
    frames_per_sec: f32,
    cfg: &HaitsmaMatchConfig,
) -> MatchResult {
    let total_bit_pairs = (overlap * 32) as u64;
    let ber = if total_bit_pairs > 0 {
        best_hamming as f32 / total_bit_pairs as f32
    } else {
        1.0
    };

    let score = clamp_score(1.0 - ber);
    let is_match = ber <= cfg.max_ber && (overlap as u32) >= cfg.min_overlap_frames;

    // Prominence: sample ~20 offsets across the range to estimate
    // typical (median) BER, then compute median / best.
    let prominence = {
        let q_len = q_frames.len();
        let r_len = r_frames.len();
        let mut bers: Vec<f32> = Vec::new();
        let step = ((q_len as i64 + r_len as i64) / 40).max(1);
        let dmin = -((q_len as i64).saturating_sub(1));
        let dmax = (r_len as i64).saturating_sub(1);

        for d in (dmin..=dmax).step_by(step as usize) {
            if d == best_delta {
                continue;
            }
            let ov = overlap_at(q_len, r_len, d);
            if ov < 32 {
                continue;
            }
            let h = hamming_at_offset(q_frames, r_frames, d, ov, u64::MAX);
            let b = h as f32 / (ov * 32) as f32;
            bers.push(b);
        }
        bers.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        if let Some(&median_ber) = bers.get(bers.len() / 2) {
            median_ber / (ber + 1e-6)
        } else {
            0.0
        }
    };

    let offset = TimeOffset::from_frames(best_delta, frames_per_sec);

    MatchResult {
        is_match,
        score,
        votes: overlap as u32,
        prominence,
        offset,
        time_scale: 1.0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

impl Default for HaitsmaMatcher {
    fn default() -> Self {
        Self::new(HaitsmaMatchConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic Haitsma fingerprint from known 32-bit frame values.
    fn make_fp(frames: &[u32]) -> HaitsmaFingerprint {
        HaitsmaFingerprint {
            frames: frames.to_vec(),
            frames_per_sec: 78.125,
        }
    }

    #[test]
    fn config_defaults() {
        let c = HaitsmaMatchConfig::default();
        assert!((c.max_ber - 0.35).abs() < 1e-6);
        assert_eq!(c.min_overlap_frames, 256);
        assert!(c.use_lut);
        assert_eq!(c.probe_bit_flips, 0);
    }

    #[test]
    fn empty_query_returns_none() {
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
        let q = make_fp(&[]);
        let r = make_fp(&[0xAAAAAAAA, 0x55555555]);
        assert_eq!(m.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn empty_reference_returns_none() {
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig::default());
        let q = make_fp(&[0xAAAAAAAA, 0x55555555]);
        let r = make_fp(&[]);
        assert_eq!(m.match_one(&q, &r), MatchResult::NONE);
    }

    #[test]
    fn self_match_ber_zero() {
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        });
        // Use enough frames for the prominence sampler to work
        let frames: Vec<u32> = (0..400).map(|i| (i * 7919) as u32).collect();
        let fp = make_fp(&frames);
        let r = m.match_one(&fp, &fp);
        assert!(r.is_match, "self-match must be positive");
        assert!(
            (r.score - 1.0).abs() < 0.001,
            "self-match BER must be ~0, got score={}",
            r.score
        );
        assert_eq!(r.offset.frames, 0, "self-match offset must be zero");
        assert_eq!(r.time_scale, 1.0);
        assert!(
            r.prominence > 2.0,
            "prominence should be high for self-match: {}",
            r.prominence
        );
    }

    #[test]
    fn self_match_lut_path() {
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            min_overlap_frames: 256,
            ..Default::default()
        });
        // 600 frames → triggers LUT path (r_len > 512)
        let frames: Vec<u32> = (0..600)
            .map(|i| (i as u32).wrapping_mul(0x01010101))
            .collect();
        let fp = make_fp(&frames);
        let r = m.match_one(&fp, &fp);
        assert!(r.is_match, "self-match via LUT must be positive");
        assert!(
            (r.score - 1.0).abs() < 0.001,
            "LUT self-match BER must be ~0"
        );
        assert_eq!(r.offset.frames, 0);
    }

    #[test]
    fn offset_recovery_positive() {
        // Query shifted +100 frames relative to reference
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);
        let ref_frames: Vec<u32> = (0..500).map(|i| i as u32).collect();
        let query_frames: Vec<u32> = (0..400).map(|i| (i + 100) as u32).collect();
        // query[i] == ref[i+100], so delta = +100
        let q = make_fp(&query_frames);
        let r = make_fp(&ref_frames);
        let result = m.match_one(&q, &r);
        assert!(result.is_match, "must match shifted copy");
        assert_eq!(result.offset.frames, 100);
    }

    #[test]
    fn offset_recovery_negative() {
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);
        // ref: 100..600, query: 50..450 — q[i]=50+i, r[i]=100+i
        // q[50] = 100 = r[0], so delta = -50
        let r3 = make_fp(&(100u32..600u32).collect::<Vec<_>>());
        let q3 = make_fp(&(50u32..450u32).collect::<Vec<_>>());
        let result = m.match_one(&q3, &r3);
        assert!(result.is_match, "must match shifted copy");
        assert_eq!(result.offset.frames, -50);
    }

    #[test]
    fn unrelated_signals_no_match() {
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);
        // Two completely different signals
        let q = make_fp(&(0u32..400u32).collect::<Vec<_>>());
        let r = make_fp(&(0xDEADBEEFu32..0xDEADBEEFu32 + 500).collect::<Vec<_>>());
        let result = m.match_one(&q, &r);
        // Even with min_overlap=1, BER should be ~0.5 (random) which >> 0.35
        assert!(!result.is_match, "unrelated signals must not match");
    }

    #[test]
    fn determinism() {
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        });
        let q = make_fp(&(0..400).map(|i| (i * 7) as u32).collect::<Vec<_>>());
        let r = make_fp(&(20..520).map(|i| (i * 7) as u32).collect::<Vec<_>>());
        let r1 = m.match_one(&q, &r);
        let r2 = m.match_one(&q, &r);
        assert_eq!(r1, r2, "match_one must be deterministic");
    }

    #[test]
    fn lut_path_matches_exact_path() {
        // Small input: under 512 ref frames → uses exact path
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            use_lut: true,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg.clone());
        let q = make_fp(&(0..200).map(|i| (i * 3 + 7) as u32).collect::<Vec<_>>());
        let r = make_fp(&(50..300).map(|i| (i * 3 + 7) as u32).collect::<Vec<_>>());

        // Exact path result
        let cfg_exact = HaitsmaMatchConfig {
            use_lut: false,
            ..cfg.clone()
        };
        let m_exact = HaitsmaMatcher::new(cfg_exact);
        let r_exact = m_exact.match_one(&q, &r);

        // LUT path result (with forced small input, LUT still used since use_lut=true
        // but r<512 so it won't use it)
        let r_lut = m.match_one(&q, &r);

        assert_eq!(r_exact, r_lut, "LUT and exact paths must agree");
    }

    #[test]
    fn lut_path_matches_exact_path_large() {
        // >512 reference frames → the LUT path is genuinely exercised
        // (the small-input test above only ever hits the exact path).
        let cfg_lut = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            use_lut: true,
            ..Default::default()
        };
        let cfg_exact = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            use_lut: false,
            ..Default::default()
        };
        let m_lut = HaitsmaMatcher::new(cfg_lut);
        let m_exact = HaitsmaMatcher::new(cfg_exact);

        // Distinct 32-bit values via a multiplicative hash.
        let reference: Vec<u32> = (0..700u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        // Bit-exact subsequence starting at frame 100 → offset +100.
        let query: Vec<u32> = reference[100..500].to_vec();
        let r = make_fp(&reference);
        let q = make_fp(&query);

        let res_lut = m_lut.match_one(&q, &r);
        let res_exact = m_exact.match_one(&q, &r);

        assert!(
            res_lut.is_match,
            "LUT path must find the bit-exact subsequence"
        );
        assert_eq!(
            res_lut.offset.frames, 100,
            "offset must be +100 (query after ref)"
        );
        assert!((res_lut.score - 1.0).abs() < 1e-6, "BER must be 0");
        assert_eq!(
            res_lut, res_exact,
            "LUT and exact paths must agree on a bit-exact subsequence"
        );
    }

    #[test]
    fn ber_increases_with_noise() {
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            max_ber: 1.0, // accept anything so we can compare scores
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);

        // Perfect match
        let q_clean = make_fp(&[0xAAAAAAAA, 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF]);
        let score_clean = m.match_one(&q_clean, &q_clean).score;
        assert!((score_clean - 1.0).abs() < 0.001);

        // Flip 1 bit in one frame → BER = 1/(32*5) ≈ 0.00625
        let mut q_noisy = q_clean.clone();
        q_noisy.frames[2] ^= 1;
        let score_noisy = m.match_one(&q_noisy, &q_clean).score;
        assert!(
            score_noisy < score_clean,
            "noisy score {} should be < clean {}",
            score_noisy,
            score_clean
        );
    }

    #[test]
    fn min_overlap_enforced() {
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1000,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);
        let fp = make_fp(&(0..50).map(|i| i as u32).collect::<Vec<_>>());
        let r = m.match_one(&fp, &fp);
        assert!(!r.is_match, "below min_overlap must not match");
    }

    #[test]
    fn prominence_spike_for_true_match() {
        let cfg = HaitsmaMatchConfig {
            min_overlap_frames: 1,
            ..Default::default()
        };
        let m = HaitsmaMatcher::new(cfg);

        // True match: identical data
        let frames: Vec<u32> = (0..400).map(|i| (i * 123 + 456) as u32).collect();
        let q = make_fp(&frames);
        let r = make_fp(&frames);
        let true_result = m.match_one(&q, &r);
        assert!(true_result.is_match);
        assert!(
            true_result.prominence > 1.5,
            "true match prominence too low: {}",
            true_result.prominence
        );

        // Random data → low prominence
        let mut rng = Xor32(42);
        let q_rand: Vec<u32> = (0..400).map(|_| rng.next()).collect();
        let r_rand: Vec<u32> = (0..400).map(|_| rng.next()).collect();
        let rand_result = m.match_one(&make_fp(&q_rand), &make_fp(&r_rand));
        // Random data might still not match, but if it does, prominence should be low
        if rand_result.is_match {
            assert!(
                rand_result.prominence < true_result.prominence,
                "random prominence {} should be < true prominence {}",
                rand_result.prominence,
                true_result.prominence
            );
        }
    }

    /// Trivial xorshift32 for test reproducibility (no dependency, no_std safe).
    struct Xor32(u32);
    impl Xor32 {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
    }
}
