//! Haitsma–Kalker matcher — BER sliding window + sub-fingerprint LUT.
//!
//! Haitsma is a dense per-frame 32-bit code; matching is
//! **bit-error-rate (BER) minimisation** over alignments.
//!
//! # Two tiers
//!
//! **Exact BER** — slide query over reference at every offset `δ`,
//! compute Hamming via hardware `POPCNT`, checking a rate-normalised
//! early-abort bound every 64 frames once a good candidate exists.
//!
//! **Sub-fingerprint LUT** — when `reference.len() > 512`, build a
//! sorted `(u32, pos)` table over reference frames. Haitsma's key
//! property: when `BER < ~0.35`, at least one query frame is bit-exact,
//! so probe each query frame's exact `u32` (plus optional 1–2 bit-flip
//! probes) → candidate offsets → run exact BER verification only there.
//! Turns `O(Q·R)` into `O(Q + candidates·overlap)`.
//!
//! # Low-information guard
//!
//! Digital silence, a steady tone or a short repeating pattern is bit-identical
//! to itself at many alignments, so a BER near zero on such a region proves
//! nothing. A fingerprint — and the region an alignment covers — must hold at
//! least four distinct frames and four varying bit positions, otherwise the
//! result is [`MatchResult::NONE`](crate::matching::MatchResult::NONE). Runs
//! shorter than 32 frames are not checked. This is a screen for degenerate
//! input, not a measure of how distinctive a recording is.
//!
//! # Performance
//!
//! - Exact: `O(Q·R)` popcounts with aggressive early-abort.
//! - LUT: `O(Q + candidates·overlap)` — sub-millisecond for song-length.
//! - Memory: transient sorted LUT of `r_len` `(frame, position)` pairs
//!   (16 bytes each) plus a `q_len + r_len`-bit seen-offset set.

extern crate alloc;

use alloc::vec::Vec;

use crate::classical::HaitsmaFingerprint;
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

/// Shortest frame run the low-information check applies to. Below this there
/// is too little evidence to call a region uninformative, so short overlaps
/// (reachable only through an explicitly small `min_overlap_frames`) keep
/// their historical behaviour.
const INFORMATIVE_MIN_FRAMES: usize = 32;
/// Distinct 32-bit frames a region needs to carry any information.
const INFORMATIVE_DISTINCT_FRAMES: usize = 4;
/// Bit positions that must vary somewhere in a region.
const INFORMATIVE_VARYING_BITS: u32 = 4;

/// `true` when `frames` holds enough variation to be evidence of identity.
///
/// Digital silence, a steady tone, or a short repeating pattern all produce
/// a handful of distinct frames. Two such regions are bit-identical at many
/// alignments, so their BER of ~0 proves nothing: without this check a clip
/// of silence matched itself (and any other silence) with score 1.0.
///
/// A region is informative when it has at least four distinct frames and at
/// least four bit positions that are not constant across it. Runs shorter
/// than 32 frames are always accepted (see [`INFORMATIVE_MIN_FRAMES`]).
/// The scan stops as soon as both conditions hold, so real audio costs a few
/// frames; only genuinely uninformative regions are read to the end.
#[inline]
pub(crate) fn informative_frames(frames: &[u32]) -> bool {
    if frames.len() < INFORMATIVE_MIN_FRAMES {
        return true;
    }
    let mut seen = [0u32; INFORMATIVE_DISTINCT_FRAMES];
    let mut distinct = 0usize;
    let mut any = 0u32;
    let mut all = u32::MAX;
    for &frame in frames {
        any |= frame;
        all &= frame;
        if distinct < INFORMATIVE_DISTINCT_FRAMES && !seen[..distinct].contains(&frame) {
            seen[distinct] = frame;
            distinct += 1;
        }
        if distinct == INFORMATIVE_DISTINCT_FRAMES
            && (any ^ all).count_ones() >= INFORMATIVE_VARYING_BITS
        {
            return true;
        }
    }
    false
}

/// [`informative_frames`] for both sides of the `overlap` frames aligned at
/// `delta` (`delta = t_reference − t_query`, as in [`hamming_at_offset`]).
#[inline]
pub(crate) fn informative_overlap(
    query: &[u32],
    reference: &[u32],
    delta: i64,
    overlap: usize,
) -> bool {
    let q_start = if delta >= 0 { 0 } else { (-delta) as usize };
    let r_start = if delta >= 0 { delta as usize } else { 0 };
    informative_frames(&query[q_start..q_start + overlap])
        && informative_frames(&reference[r_start..r_start + overlap])
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
pub(crate) struct BestAlignment {
    pub(crate) ber: f64,
    pub(crate) hamming: u64,
    pub(crate) delta: i64,
    pub(crate) overlap: usize,
}

impl BestAlignment {
    pub(crate) fn new() -> Self {
        Self {
            ber: f64::INFINITY,
            hamming: u64::MAX,
            delta: 0,
            overlap: 0,
        }
    }

    pub(crate) fn found(&self) -> bool {
        self.ber < f64::INFINITY
    }

    /// Verify `delta` and update the incumbent if its BER is strictly
    /// better and the aligned region carries information (see
    /// [`informative_frames`]). Ties keep the first-found candidate (the
    /// exact path scans deltas in ascending order; the LUT path probes query
    /// frames in order), so selection is deterministic.
    #[inline]
    pub(crate) fn consider(
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
        // The informativeness scan runs only for a candidate that would
        // become the new best, so it is off the hot path.
        if ber < self.ber && informative_overlap(query, reference, delta, overlap) {
            self.ber = ber;
            self.hamming = h;
            self.delta = delta;
            self.overlap = overlap;
        }
    }
}

/// Sub-fingerprint LUT: `(frame, position)` pairs sorted by frame, then
/// position.
///
/// One allocation and one sort. The previous `HashMap<u32, Vec<usize>>`
/// paid one `Vec` per distinct frame — nearly every frame of a real
/// reference — and since `match_one` rebuilds the LUT on every call, that
/// build was most of its cost (≈ 5 ms of 5.6 ms on a 4-minute reference).
/// Each frame's positions come back ascending, as the map's insertion
/// order gave them, so candidates are considered in the same order.
///
/// Bit-flip probing does up to 529 lookups per query frame, nearly all
/// misses, so a one-bit-per-slot membership filter (≥ 16 slots per
/// reference frame, ~3–6 % false positives) answers most of them before
/// the binary search.
struct Lut {
    pairs: Vec<(u32, usize)>,
    filter: Vec<u64>,
    /// `32 − log2(filter bits)`: a slot is the top bits of a
    /// multiplicative hash of the frame.
    shift: u32,
}

impl Lut {
    fn build(reference: &[u32]) -> Self {
        let mut pairs: Vec<(u32, usize)> = reference
            .iter()
            .enumerate()
            .map(|(pos, &frame)| (frame, pos))
            .collect();
        pairs.sort_unstable();
        let bits = (reference.len() * 16)
            .next_power_of_two()
            .clamp(64, 1 << 24);
        let shift = 32 - bits.trailing_zeros();
        let mut filter = alloc::vec![0_u64; bits / 64];
        for &frame in reference {
            let slot = Self::slot(frame, shift);
            filter[slot / 64] |= 1 << (slot % 64);
        }
        Self {
            pairs,
            filter,
            shift,
        }
    }

    #[inline]
    fn slot(frame: u32, shift: u32) -> usize {
        (frame.wrapping_mul(0x9E37_79B1) >> shift) as usize
    }

    /// Every `(frame, position)` entry for `frame`, positions ascending.
    #[inline]
    fn get(&self, frame: u32) -> &[(u32, usize)] {
        let slot = Self::slot(frame, self.shift);
        if self.filter[slot / 64] & (1 << (slot % 64)) == 0 {
            return &[];
        }
        let start = self.pairs.partition_point(|&(f, _)| f < frame);
        let len = self.pairs[start..].partition_point(|&(f, _)| f == frame);
        &self.pairs[start..start + len]
    }
}

/// Probe all 1-bit-flip variants of `frame` (32 variants).
#[inline]
fn probe_1flip(frame: u32, lut: &Lut, f: &mut impl FnMut(&[(u32, usize)])) {
    probe_exact(frame, lut, f);
    for bit in 0..32 {
        probe_exact(frame ^ (1u32 << bit), lut, f);
    }
}

/// Probe exact + 1-bit + 2-bit-flip variants (1 + 32 + 496 = 529 probes).
#[inline]
fn probe_2flip(frame: u32, lut: &Lut, f: &mut impl FnMut(&[(u32, usize)])) {
    probe_1flip(frame, lut, f);
    for b1 in 0..32 {
        for b2 in (b1 + 1)..32 {
            probe_exact(frame ^ (1u32 << b1) ^ (1u32 << b2), lut, f);
        }
    }
}

/// Probe exact only (no bit flips).
#[inline]
fn probe_exact(frame: u32, lut: &Lut, f: &mut impl FnMut(&[(u32, usize)])) {
    let hits = lut.get(frame);
    if !hits.is_empty() {
        f(hits);
    }
}

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

impl HaitsmaMatcher {
    /// Estimated P(same recording | Haitsma evidence): logistic map of the
    /// `1 − BER` score, versioned `haitsma-v1` (see
    /// `matching::calibration`).
    ///
    /// Shared with the `HaitsmaIndex::query` path — both report
    /// `score = 1 − BER`. Their prominence formulas differ (documented in
    /// `calibrated_haitsma`), so treat near-boundary index results as
    /// carrying that path's selection bias.
    #[must_use]
    pub fn calibrated_score(&self, r: &MatchResult) -> f32 {
        super::calibration::calibrated_haitsma(r)
    }
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

        // Silence, a steady tone or a short repeating pattern is not
        // evidence of identity: it aligns with itself at BER ~0 everywhere.
        if q_len == 0
            || r_len == 0
            || !informative_frames(q_frames)
            || !informative_frames(r_frames)
        {
            return MatchResult::NONE;
        }

        let min_overlap = self.cfg.min_overlap_frames as usize;

        // LUT path
        let use_lut = self.cfg.use_lut && r_len > 512;
        if use_lut {
            let lut = Lut::build(r_frames);
            let mut best = BestAlignment::new();

            // Deduplicated body across the three probe modes: verify
            // each candidate offset by BER with a rate-normalized
            // early-abort bound.
            // A repeated exact frame hit can identify the same alignment
            // hundreds of times. Remember deltas for this entire query;
            // consideration still occurs in first-seen order for stable ties.
            // Deltas span `[-(q_len - 1), r_len - 1]`, so a bitset over that
            // range replaces a `BTreeSet` (one node per candidate).
            let d_min = -(q_len as i64 - 1);
            let mut seen_deltas = alloc::vec![0_u64; (q_len + r_len).div_ceil(64)];

            for (q_pos, &q_frame) in q_frames.iter().enumerate() {
                let mut consider = |hits: &[(u32, usize)]| {
                    for &(_, r_pos) in hits {
                        let delta = r_pos as i64 - q_pos as i64;
                        let bit = (delta - d_min) as usize;
                        let (word, mask) = (bit / 64, 1_u64 << (bit % 64));
                        if seen_deltas[word] & mask != 0 {
                            continue;
                        }
                        seen_deltas[word] |= mask;
                        best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
                    }
                };
                match self.cfg.probe_bit_flips {
                    0 => probe_exact(q_frame, &lut, &mut consider),
                    1 => probe_1flip(q_frame, &lut, &mut consider),
                    _ => probe_2flip(q_frame, &lut, &mut consider),
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
                reference.frames_per_sec,
                &self.cfg,
            );
        }

        // Exact BER path (scan all offsets, ascending delta so
        // BER ties resolve to the smallest offset deterministically).
        let dmin: i64 = -((q_len as i64).saturating_sub(1));
        let dmax: i64 = (r_len as i64).saturating_sub(1);

        let mut best = BestAlignment::new();

        for delta in dmin..=dmax {
            best.consider(q_frames, r_frames, q_len, r_len, delta, min_overlap);
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
            reference.frames_per_sec,
            &self.cfg,
        )
    }
}

/// BER, score and match verdict of an alignment already accepted by
/// [`BestAlignment::consider`], which enforces `min_overlap_frames`.
pub(crate) fn ber_score_match(hamming: u64, overlap: usize, max_ber: f32) -> (f32, f32, bool) {
    let total_bit_pairs = (overlap * 32) as u64;
    let ber = if total_bit_pairs > 0 {
        hamming as f32 / total_bit_pairs as f32
    } else {
        1.0
    };
    (ber, clamp_score(1.0 - ber), ber <= max_ber)
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
    let (ber, score, is_match) = ber_score_match(best_hamming, overlap, cfg.max_ber);

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
    use alloc::vec;

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

    use crate::matching::XorShift;

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
        let mut rng = XorShift(42);
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

    // ── Low-information guard ──

    /// Frame sequences that carry no evidence of identity.
    fn degenerate_sequences(len: usize) -> Vec<(&'static str, Vec<u32>)> {
        vec![
            ("silence", vec![0; len]),
            ("all ones", vec![u32::MAX; len]),
            ("alternating 0/1", (0..len).map(|i| i as u32 % 2).collect()),
            (
                "period 3",
                (0..len)
                    .map(|i| [0xAAAA_AAAA_u32, 0x5555_5555, 0][i % 3])
                    .collect(),
            ),
        ]
    }

    fn random_frames(seed: u32, len: usize) -> Vec<u32> {
        let mut rng = XorShift(seed);
        (0..len).map(|_| rng.next()).collect()
    }

    #[test]
    fn silence_and_repeated_frames_are_not_evidence() {
        // 400 frames exercises the exact path; 900 (> 512) the LUT path.
        for len in [400, 900] {
            for (name, frames) in degenerate_sequences(len) {
                let fp = make_fp(&frames);
                for use_lut in [true, false] {
                    let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
                        use_lut,
                        ..Default::default()
                    });
                    assert_eq!(
                        m.match_one(&fp, &fp),
                        MatchResult::NONE,
                        "{name} len={len} use_lut={use_lut}: self-match"
                    );
                    // Either side degenerate is enough to reject.
                    let real = make_fp(&random_frames(0x1234_5678, 900));
                    assert_eq!(
                        m.match_one(&fp, &real),
                        MatchResult::NONE,
                        "{name} as query"
                    );
                    assert_eq!(
                        m.match_one(&real, &fp),
                        MatchResult::NONE,
                        "{name} as reference"
                    );
                }
            }
        }
    }

    #[test]
    fn real_content_still_matches_and_is_unaffected_by_the_guard() {
        let reference = random_frames(0xBEEF_0001, 900);
        let excerpt = make_fp(&reference[200..600]);
        let reference = make_fp(&reference);
        for use_lut in [true, false] {
            let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
                use_lut,
                ..Default::default()
            });
            let r = m.match_one(&excerpt, &reference);
            assert!(r.is_match, "use_lut={use_lut}: {r:?}");
            assert_eq!(r.offset.frames, 200);
            assert_eq!(r.score, 1.0);
        }
    }

    #[test]
    fn informative_frames_thresholds() {
        // < 32 frames are never judged.
        assert!(informative_frames(&[]));
        assert!(informative_frames(&[0; 31]));
        // 32 identical frames are uninformative.
        assert!(!informative_frames(&[0; 32]));
        assert!(!informative_frames(&[0xDEAD_BEEF; 32]));
        // Three distinct frames are not enough, even with many varying bits.
        let three: Vec<u32> = (0..64)
            .map(|i| [0u32, u32::MAX, 0x0F0F_0F0F][i % 3])
            .collect();
        assert!(!informative_frames(&three));
        // Four distinct frames that differ in four bit positions are enough.
        let four: Vec<u32> = (0..64).map(|i| (i % 4) as u32 | 0xF0).collect();
        assert!(!informative_frames(&four), "only 2 varying bits");
        let four_wide: Vec<u32> = (0..64)
            .map(|i| [0u32, 0b0011, 0b1100, 0b1111][i % 4])
            .collect();
        assert!(informative_frames(&four_wide));
        // Real audio is informative.
        assert!(informative_frames(&random_frames(7, 64)));
    }

    #[test]
    fn short_overlap_requests_keep_historical_behaviour() {
        // With an explicit tiny `min_overlap_frames`, runs under 32 frames
        // are not screened, so a 20-frame constant still self-matches.
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            min_overlap_frames: 16,
            use_lut: false,
            ..Default::default()
        });
        let fp = make_fp(&[7; 20]);
        assert!(m.match_one(&fp, &fp).is_match);
    }

    #[test]
    fn silent_alignment_does_not_beat_a_real_one() {
        // The query is real content padded with silence; the reference holds
        // the same real content. The true alignment must win.
        let real = random_frames(0xC0FF_EE01, 500);
        let mut query = real.clone();
        query.extend(vec![0; 300]);
        let mut reference = real;
        reference.extend(vec![0; 300]);
        let m = HaitsmaMatcher::new(HaitsmaMatchConfig {
            use_lut: false,
            ..Default::default()
        });
        let r = m.match_one(&make_fp(&query), &make_fp(&reference));
        assert!(r.is_match, "{r:?}");
        assert_eq!(r.offset.frames, 0);
        assert_eq!(r.score, 1.0);
    }
}
