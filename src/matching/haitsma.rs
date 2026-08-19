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

    // Process in chunks of 64 frames without a per-element early-abort.
    // This allows LLVM to auto-vectorize the XOR + POPCNT into SIMD
    // instructions (on x86: vpxor + vpopcntd or scalar popcnt unrolled).
    // The early-abort check every 64 frames retains the short-circuit
    // benefit while amortizing the branch cost.
    let mut hamming: u64 = 0;
    let mut q_iter = q_slice.chunks_exact(64);
    let mut r_iter = r_slice.chunks_exact(64);
    for (q_chunk, r_chunk) in q_iter.by_ref().zip(r_iter.by_ref()) {
        let mut block_sum: u64 = 0;
        for (qa, ra) in q_chunk.iter().zip(r_chunk.iter()) {
            block_sum += (qa ^ ra).count_ones() as u64;
        }
        hamming += block_sum;
        if hamming > best_sofar {
            return u64::MAX;
        }
    }
    // Scalar tail (< 64 remaining frames).
    for (qa, ra) in q_iter.remainder().iter().zip(r_iter.remainder().iter()) {
        hamming += (qa ^ ra).count_ones() as u64;
    }
    if hamming > best_sofar {
        return u64::MAX;
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

/// Running best alignment across candidate offsets, compared by **BER**
/// (hamming ÷ overlap-bits) — not by raw hamming.
///
/// The BER basis matters for the early-abort bound: candidates at
/// different deltas have different overlap lengths, so an absolute
/// "best hamming so far" lets a short-overlap candidate with a small
/// raw total suppress a longer candidate with a strictly better rate
/// (e.g. 256 frames @ 30 % BER = 2 458 bits beating 768 frames @
/// 5 % BER on totals while losing on rate). Passing
/// `best_ber × overlap × 32` as the abort bound keeps the prune
/// rate-normalized: a candidate is only aborted when its *rate* can no
/// longer beat the incumbent.
struct BestAlignment {
    ber: f64,
    hamming: u64,
    delta: i64,
    overlap: usize,
}

impl BestAlignment {
    fn new() -> Self {
        Self {
            ber: f64::INFINITY,
            hamming: u64::MAX,
            delta: 0,
            overlap: 0,
        }
    }

    fn found(&self) -> bool {
        self.ber < f64::INFINITY
    }

    /// Verify `delta` and update the incumbent if its BER is strictly
    /// better. Ties keep the first-found candidate (the exact path
    /// scans deltas in ascending order; the LUT path probes query
    /// frames in order), so selection is deterministic.
    #[inline]
    fn consider(
        &mut self,
        query: &[u32],
        reference: &[u32],
        q_len: usize,
        r_len: usize,
        delta: i64,
        min_overlap: usize,
    ) {
        let overlap = overlap_at(q_len, r_len, delta);
        if overlap < min_overlap {
            return;
        }
        // Saturating cast: `INFINITY as u64` → `u64::MAX` (never aborts);
        // finite bounds are far below u64::MAX for any real overlap.
        let bound = (self.ber * overlap as f64 * 32.0) as u64;
        let h = hamming_at_offset(query, reference, delta, overlap, bound);
        if h == u64::MAX {
            return; // aborted: rate cannot beat the incumbent
        }
        let ber = h as f64 / (overlap as f64 * 32.0);
        if ber < self.ber {
            self.ber = ber;
            self.hamming = h;
            self.delta = delta;
            self.overlap = overlap;
        }
    }
}

/// Build a LUT: `u32` sub-fingerprint → list of reference frame indices.
fn build_lut(reference: &[u32]) -> HashMap<u32, Vec<usize>> {
    let mut lut: HashMap<u32, Vec<usize>> = super::maps::hashmap_with_capacity(reference.len() / 2);
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

// audit C3: coarse-to-fine bounds for the exhaustive BER path. Inputs
// with `q_len + r_len >= COARSE_TO_FINE_THRESHOLD` total frames run the
// sampled sweep first; anything smaller keeps the untouched exhaustive
// scan (identical results to pre-C3 code).
const COARSE_TO_FINE_THRESHOLD: usize = 4096;
/// Frames skipped between consecutive probe frames inside one coarse
/// hamming estimate.
const COARSE_DELTA_SAMPLE_STRIDE: usize = 8;
/// Cap on the number of deltas sampled by the coarse sweep.
const COARSE_MAX_PROBES: usize = 2048;
/// How many top coarse candidates get full-resolution verification.
const COARSE_REFINE_CANDIDATES: usize = 8;
/// Full-resolution refinement window around the best coarse delta,
/// as a multiple of the coarse stride.
const COARSE_REFINE_WINDOW: i64 = 2;

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
    /// Use the coarse-to-fine accelerator on the exact-BER path (audit
    /// C3). Default false.
    ///
    /// When `true` and `query.frames + reference.frames ≥ 4096`, the
    /// exhaustive O(q·r) delta scan is replaced by a stride-sampled
    /// sweep over the delta grid, a full-resolution refinement window
    /// around the best coarse peak, and individual verification of the
    /// top coarse candidates. For the typical no-match / weak-match
    /// case this is orders of magnitude faster than the exhaustive
    /// scan.
    ///
    /// **Result caveat:** sampling can miss a needle-in-haystack
    /// bit-exact alignment that the exhaustive scan would find (a
    /// self-match whose true delta falls between coarse grid points
    /// scores ~0.5 on its neighbours). Results are therefore not
    /// guaranteed to match the exhaustive path. Leave this off unless
    /// you deliberately disable the LUT on very long references and
    /// accept the tradeoff.
    pub coarse_to_fine: bool,
}

impl Default for HaitsmaMatchConfig {
    fn default() -> Self {
        Self {
            max_ber: 0.35,
            min_overlap_frames: 256,
            use_lut: true,
            probe_bit_flips: 0,
            coarse_to_fine: false,
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

        // LUT path
        let use_lut = self.cfg.use_lut && r_len > 512;
        if use_lut {
            let lut = build_lut(r_frames);
            let mut best = BestAlignment::new();

            // Deduplicated body across the three probe modes: verify
            // each candidate offset by BER with a rate-normalized
            // early-abort bound.
            let consider = |q_pos: usize, positions: &Vec<usize>, best: &mut BestAlignment| {
                for &r_pos in positions {
                    let delta = r_pos as i64 - q_pos as i64;
                    best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
                }
            };

            for (q_pos, &q_frame) in q_frames.iter().enumerate() {
                match self.cfg.probe_bit_flips {
                    0 => probe_exact(q_frame, &lut, &mut |positions| {
                        consider(q_pos, positions, &mut best)
                    }),
                    1 => probe_1flip(q_frame, &lut, &mut |positions| {
                        consider(q_pos, positions, &mut best)
                    }),
                    _ => probe_2flip(q_frame, &lut, &mut |positions| {
                        consider(q_pos, positions, &mut best)
                    }),
                }
                if best.ber == 0.0 {
                    break;
                }
            }

            if !best.found() {
                return MatchResult::NONE;
            }

            return build_result(
                best.hamming,
                best.delta,
                best.overlap,
                q_frames,
                r_frames,
                query.frames_per_sec,
                &self.cfg,
            );
        }

        // Exact BER path (scan all offsets, ascending delta so
        // BER ties resolve to the smallest offset deterministically).
        //
        // audit C3: for very long references the exhaustive scan is
        // O(q·r·32) over ALL offsets — the early-abort only helps when a
        // good alignment exists. Inputs at or above
        // COARSE_TO_FINE_THRESHOLD total frames first run a cheap
        // sampled sweep over a stride-sampled delta grid, then verify a
        // narrow full-resolution window around the best coarse peak plus
        // the top coarse candidates individually. Below the threshold
        // the scan is exactly the original exhaustive loop, so all
        // small/medium inputs are bit-identical to before.
        let dmin: i64 = -((q_len as i64).saturating_sub(1));
        let dmax: i64 = (r_len as i64).saturating_sub(1);

        let mut best = BestAlignment::new();

        let use_coarse = self.cfg.coarse_to_fine && q_len + r_len >= COARSE_TO_FINE_THRESHOLD;
        if use_coarse {
            // ── Coarse sweep: stride-sampled deltas, thinned hamming ──
            let delta_stride = ((dmax - dmin + 1) as usize / COARSE_MAX_PROBES).max(1) as i64;
            let mut coarse: Vec<(f64, i64)> = Vec::new(); // (ber, delta)
            let mut delta = dmin;
            while delta <= dmax {
                let overlap = overlap_at(q_len, r_len, delta);
                if overlap >= min_overlap {
                    let (q_off, r_off) = if delta >= 0 {
                        (0usize, delta as usize)
                    } else {
                        ((-delta) as usize, 0usize)
                    };
                    // Hamming over every COARSE_DELTA_SAMPLE_STRIDE-th
                    // frame of the overlap — a cheap rate estimate.
                    let mut h: u64 = 0;
                    let mut n_sampled: usize = 0;
                    let mut f = 0usize;
                    while f < overlap {
                        h += (q_frames[q_off + f] ^ r_frames[r_off + f]).count_ones() as u64;
                        n_sampled += 1;
                        f += COARSE_DELTA_SAMPLE_STRIDE;
                    }
                    let ber = h as f64 / (n_sampled.max(1) * 32) as f64;
                    coarse.push((ber, delta));
                }
                delta += delta_stride;
            }
            if coarse.is_empty() {
                return MatchResult::NONE;
            }
            coarse.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(core::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });

            // ── Refine: full-resolution window around the best coarse
            //     peak, then the remaining top candidates. ──
            let best_delta = coarse[0].1;
            let win = delta_stride * COARSE_REFINE_WINDOW;
            let lo = (best_delta - win).max(dmin);
            let hi = (best_delta + win).min(dmax);
            for delta in lo..=hi {
                best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
            }
            for &(_, delta) in coarse.iter().take(COARSE_REFINE_CANDIDATES) {
                if delta < lo || delta > hi {
                    best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
                }
            }
        } else {
            // Exhaustive scan (unchanged) for anything below the
            // threshold: identical results to pre-C3 behaviour.
            for delta in dmin..=dmax {
                best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
            }
        }

        if !best.found() {
            return MatchResult::NONE;
        }

        build_result(
            best.hamming,
            best.delta,
            best.overlap,
            q_frames,
            r_frames,
            query.frames_per_sec,
            &self.cfg,
        )
    }
}

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

    // Prominence: sample offsets across the range to estimate typical
    // (median) BER, then compute median / best.
    let prominence = {
        let q_len = q_frames.len();
        let r_len = r_frames.len();
        let mut bers: Vec<f32> = Vec::new();
        // Sample ~40 offsets across the range (40 balances estimate
        // stability against per-offset BER cost).
        let step = ((q_len as i64 + r_len as i64) / 40).max(1);
        let dmin = -((q_len as i64).saturating_sub(1));
        let dmax = (r_len as i64).saturating_sub(1);

        for d in (dmin..=dmax).step_by(step as usize) {
            if d == best_delta {
                continue;
            }
            let ov = overlap_at(q_len, r_len, d);
            // Skip near-empty overlaps so the BER estimate is statistically
            // meaningful (32 bits = one frame's worth).
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

    // ── BER-normalized early-abort regression ──
    //
    // A short-overlap candidate with a small *absolute* hamming total
    // but a *worse* bit-error rate must not suppress a longer,
    // better-rate alignment via the early-abort bound. Pre-fix, the
    // bound was the raw incumbent total: 256 frames @ 30 % BER
    // (2 458 bits) evaluated first would abort 300 frames @ 28 % BER
    // (2 688 bits) mid-scan and win with the wrong offset.

    /// Deterministic xorshift32.
    struct XorShift(u32);
    impl XorShift {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
    }

    /// Copy `src`, flipping each bit independently with probability
    /// `pct`% (deterministic given the seed).
    fn flip_bits(src: &[u32], pct: u32, rng: &mut XorShift) -> Vec<u32> {
        src.iter()
            .map(|&f| {
                let mut v = f;
                for b in 0..32 {
                    if rng.next() % 100 < pct {
                        v ^= 1 << b;
                    }
                }
                v
            })
            .collect()
    }

    #[test]
    fn short_overlap_worse_ber_does_not_beat_longer_better_ber() {
        // Reference: 900 random frames.
        let mut rng = XorShift(0xBEEF_5EED);
        let r: Vec<u32> = (0..900).map(|_| rng.next() | 1).collect();

        // Query, 800 frames, two crafted alignments on disjoint windows:
        // - TRUE, delta = +600: Q[0..299]  = R[600..899] @ 28% flips
        //   → overlap 300, BER ≈ 0.28 (total ≈ 2 688 bits).
        // - SPURIOUS, delta = -544 (evaluated FIRST in the ascending
        //   exact-path scan): Q[544..799] = R[0..255] @ 30% flips
        //   → overlap 256, BER ≈ 0.30 (total ≈ 2 458 bits).
        // The middle stretch Q[300..543] is fresh random noise.
        let true_win = flip_bits(&r[600..900], 28, &mut rng);
        let spur_win = flip_bits(&r[0..256], 30, &mut rng);
        let mid: Vec<u32> = (0..244).map(|_| rng.next() | 1).collect();

        let mut q: Vec<u32> = Vec::with_capacity(800);
        q.extend_from_slice(&true_win); // 300
        q.extend_from_slice(&mid); // 244
        q.extend_from_slice(&spur_win); // 256 → 800 total
        assert_eq!(q.len(), 800);

        let query = make_fp(&q);
        let reference = make_fp(&r);

        // Exact path (LUT would find no bit-exact frames under 28/30%
        // flips and is covered by its own tests).
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            coarse_to_fine: true,
            ..Default::default()
        });
        let res = m.match_one(&query, &reference);
        assert!(res.is_match, "true alignment (BER≈0.28) must match");
        assert_eq!(
            res.offset.frames, 600,
            "must pick the longer better-rate alignment (delta 600), not the \
             short worse-rate one (delta -544); ber-based selection broken?",
        );

        // And the winner's score reflects the ≈0.28 BER, not ≈0.30.
        assert!(
            res.score > 0.70 && res.score < 0.73,
            "score {} should reflect BER ≈ 0.28",
            res.score
        );
    }

    #[test]
    fn config_defaults() {
        let c = HaitsmaMatchConfig::default();
        assert!((c.max_ber - 0.35).abs() < 1e-6);
        assert_eq!(c.min_overlap_frames, 256);
        assert!(c.use_lut);
        assert_eq!(c.probe_bit_flips, 0);
        assert!(!c.coarse_to_fine, "coarse-to-fine must be opt-in");
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

    // ── Coarse-to-fine path (audit C3) ──
    //
    // Only inputs with q_len + r_len >= COARSE_TO_FINE_THRESHOLD (4096)
    // exercise the sampled sweep + refinement; everything below keeps the
    // exhaustive scan, so the tests above are unchanged pre-C3 behaviour.

    #[test]
    fn coarse_path_recovers_true_offset_on_long_reference() {
        let ref_len = 4000usize;
        let query_len = 3000usize;
        assert!(ref_len + query_len >= super::COARSE_TO_FINE_THRESHOLD);

        let mut rng = XorShift(0xC0A5_E5ED);
        let reference: Vec<u32> = (0..ref_len).map(|_| rng.next() | 1).collect();
        // True delta 496 is ON the coarse grid (stride 3: dmin = -2999,
        // 496 + 2999 = 3495 ≡ 0 mod 3), so the sampled sweep sees the
        // low-BER spike and the refinement window recovers it exactly.
        let query = flip_bits(&reference[496..496 + query_len], 4, &mut rng);

        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            coarse_to_fine: true,
            ..Default::default()
        });
        let res = m.match_one(&make_fp(&query), &make_fp(&reference));
        assert!(res.is_match, "coarse path must find the true alignment");
        assert_eq!(
            res.offset.frames, 496,
            "coarse path must recover delta +496, got {}",
            res.offset.frames
        );
        assert!(
            res.score > 0.90 && res.score < 0.98,
            "score {} should reflect BER ≈ 0.04",
            res.score
        );
    }

    #[test]
    fn coarse_path_self_match_is_perfect() {
        // n = 4097 keeps the true delta (0) ON the coarse grid: stride =
        // (2n - 1) / 2048 = 4 and dmin = -4096 ≡ 0 mod 4.
        let n = 4097usize;
        assert!(2 * n >= super::COARSE_TO_FINE_THRESHOLD);
        let frames: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761))
            .collect();
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            coarse_to_fine: true,
            ..Default::default()
        });
        let res = m.match_one(&make_fp(&frames), &make_fp(&frames));
        assert!(res.is_match, "long self-match must be positive");
        assert!((res.score - 1.0).abs() < 0.001, "BER must be ~0");
        assert_eq!(res.offset.frames, 0);
    }

    #[test]
    fn coarse_path_no_match_on_unrelated_long_reference() {
        let mut rng = XorShift(0xDEADBEEF);
        let q: Vec<u32> = (0..3000).map(|_| rng.next() | 1).collect();
        let r: Vec<u32> = (0..4000).map(|_| rng.next() | 1).collect();
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            coarse_to_fine: true,
            ..Default::default()
        });
        let res = m.match_one(&make_fp(&q), &make_fp(&r));
        // Random-vs-random BER ≈ 0.5 >> max_ber 0.35.
        assert!(!res.is_match);
    }

    #[test]
    fn coarse_path_is_deterministic() {
        let ref_len = 5000usize;
        let query_len = 3000usize;
        let mut rng = XorShift(0x5EED_CAFE);
        let reference: Vec<u32> = (0..ref_len).map(|_| rng.next() | 1).collect();
        let query = flip_bits(&reference[1000..1000 + query_len], 3, &mut rng);

        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            coarse_to_fine: true,
            ..Default::default()
        });
        let a = m.match_one(&make_fp(&query), &make_fp(&reference));
        let b = m.match_one(&make_fp(&query), &make_fp(&reference));
        assert_eq!(a, b, "coarse path must be deterministic");
        assert!(a.is_match);
        assert_eq!(a.offset.frames, 1000);
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
