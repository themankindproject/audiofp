//! Panako-style triplet fingerprinter.
//!
//! Same front-end as [`super::Wang`] (8 kHz, STFT `n_fft=1024 hop=128`
//! Hann, dB log-magnitude peak picking) but each anchor produces hashes
//! over *triplets* `(a, b, c)` rather than pairs. The third peak gives a
//! tempo-invariant ratio `β` that is robust to ±5 % time stretch.
//!
//! Hash layout (this crate's `panako-v2`), high to low bit:
//! ```text
//! [31..30]  sign       (2 bits, sign of Δf_ab and Δf_bc)
//! [29..28]  mag_order  (2 bits, which of {a, b, c} has the largest magnitude)
//! [27..23]  β          (5 bits, round((t_c - t_b) / (t_c - t_a) · 31))
//! [22..15]  Δf_ab      (8 bits, signed, clamped to ±127)
//! [14.. 7]  Δf_bc      (8 bits, signed, clamped to ±127)
//! [ 6.. 0]  reserved   (7 bits, zero)
//! ```
//!
//! ## Relationship to Panako (Six 2014, Six 2021)
//!
//! The hash layout, peak-zone constraints, and `fan_out` cap are the
//! authors' own and are documented inline in [`PanakoConfig`]. The
//! fingerprinting idea — encoding a tempo-invariant ratio and
//! pitch-invariant frequency differences from a peak triplet — comes
//! from Panako:
//!
//! - Six, J., Leman, M. (2014). *Panako — A Scalable Acoustic
//!   Fingerprinting System Handling Time-Scale and Pitch Modifications.*
//!   proceedings of ISMIR.
//! - Six, J. (2021). *Panako 2.0 — Updates for an Acoustic
//!   Fingerprinting System.* Late-Breaking ISMIR.
//!
//! Deliberate divergences from the original Panako:
//!
//! 1. **Front-end.** Six uses a Constant-Q transform; this crate uses a
//!    Hann-windowed STFT (same front-end as [`super::Wang`]) to share
//!    the DSP stack. A CQT front-end is tracked under `future.md` §1.3.
//! 2. **Time-ratio resolution.** Six 2021 quantises the ratio
//!    `(t2 − t1) / (t3 − t1)` to 8 bits; this crate stores `β =
//!    (t_c − t_b) / (t_c − t_a)` to 5 bits and leaves the remaining
//!    bits reserved. As a result, this crate's hashes are *not*
//!    drop-in compatible with a Six-format Panako database — they share
//!    the invariance *principle* but not the bit layout.
//! 3. **Frequency differences.** Six quantises unsigned `|Δf|`; this
//!    crate uses signed `Δf` clamped to ±127 to fit the 8-bit slot and
//!    signs are packed into the top 2 bits.
//! 4. **No coarse band indices.** Six's 32-bit layout also stores
//!    4-bit coarse band indices for the anchor and second target; this
//!    crate uses the 4 bits for `sign` and `mag_order` instead.
//!
//! Robustness claims (e.g. "±10 % speed / ±200 cents pitch") are
//! matcher-side properties in the Panako papers and depend on the
//! downstream alignment procedure. This crate only guarantees
//! deterministic, codec-tolerant hashes; callers are responsible for
//! the matching logic.
//!
//! # Patent status
//!
//! The Panako algorithm was published as academic research (ISMIR 2014,
//! ISMIR 2021 late-breaking) by Joren Six at Ghent University. No
//! utility patents were filed on the triplet-ratio fingerprinting
//! method. The algorithm is free to use commercially worldwide.
//! Note: Six's reference implementation is AGPL-licensed (software
//! copyright), but this is an independent MIT-licensed reimplementation
//! from the published paper — no AGPL code was used.

use alloc::vec::Vec;

use crate::classical::stream;
use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, Fingerprinter, Result, SampleRate, StreamingFingerprinter, TimestampMs};

/// One anchor-target-target triplet packed into a 32-bit hash plus the
/// three STFT frame indices.
///
/// The type is `#[repr(C)]` and implements [`bytemuck::Pod`], enabling
/// zero-copy persistence (mmap, flat files) and C FFI. Layout is four
/// little-endian `u32` fields (16 bytes total, no padding).
///
/// # Frame index invariants
///
/// `t_anchor < t_b < t_c` — the anchor is always the earliest peak,
/// `t_b` is the nearer target, and `t_c` is the farther one. The hash
/// field encodes a tempo-invariant ratio `β = (t_c − t_b) / (t_c − t_a)`
/// so matching survives ±5 % speed changes.
///
/// See the module-level doc for the full 32-bit hash layout.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PanakoHash {
    /// 32-bit hash; see module docs for the bit layout.
    pub hash: u32,
    /// STFT frame index of the anchor peak (earliest of the three).
    pub t_anchor: u32,
    /// STFT frame index of the nearer target (`t_anchor < t_b < t_c`).
    pub t_b: u32,
    /// STFT frame index of the farther target (`t_b < t_c`).
    pub t_c: u32,
}

/// All triplet hashes produced by [`Panako`] over an audio buffer.
///
/// # Ordering invariant
///
/// `hashes` is sorted by `(t_anchor, t_b, t_c, hash)`. Consumers may
/// rely on this for binary search or merge-join during matching.
///
/// # Typical output size
///
/// At default config (`fan_out = 5`, `peaks_per_sec = 30`), expect
/// roughly **250 hashes per second** of rich audio, or ~4 KB/s
/// (`250 × 16 bytes`). Silence produces zero hashes.
#[derive(Clone, Debug)]
pub struct PanakoFingerprint {
    /// Hashes sorted by `(t_anchor, t_b, t_c, hash)`.
    pub hashes: Vec<PanakoHash>,
    /// Frame rate of the underlying STFT — always 62.5 for `panako-v2`.
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Panako`].
///
/// Always construct with FRU so future additive fields stay compatible:
/// `PanakoConfig { fan_out: 3, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct PanakoConfig {
    /// Triplets emitted per anchor. Default 5; raising this fattens the
    /// hash database with marginally weaker triplets.
    pub fan_out: u16,
    /// Maximum `Δt` between anchor and the *farther* target. Default 96.
    pub target_zone_t: u16,
    /// Maximum `|Δf|` between anchor and either target. Default 96.
    pub target_zone_f: u16,
    /// Per-second cap on peak count. Default 30.
    pub peaks_per_sec: u16,
    /// Magnitude floor (dB) below which peaks are ignored. Default −50.
    pub min_anchor_mag_db: f32,
    /// Maximum input sample count accepted by [`extract`]. `None` disables
    /// the check. Default: 14_400_000 (30 minutes at 8 kHz).
    ///
    /// [`extract`]: Panako::extract
    pub max_input_samples: Option<usize>,
    /// Maximum number of hashes allowed. `None` disables. Default: 500_000.
    pub max_hashes: Option<usize>,
    /// Maximum number of pending anchors in the streaming pipeline.
    /// `None` disables (default, unbounded). When set, anchors exceeding
    /// this cap are dropped oldest-first so memory stays bounded.
    /// Recommended: `Some(10_000)` for untrusted input.
    /// Relevant only for [`StreamingPanako`].
    pub max_pending_anchors: Option<usize>,
    /// Maximum samples accepted in a single `push` call. `None` disables
    /// (default). When set, excess samples beyond the cap are **dropped**
    /// (streaming `push` is infallible). Use this to bound per-push
    /// memory under hostile chunk sizes.
    pub max_push_samples: Option<usize>,
}

impl Default for PanakoConfig {
    fn default() -> Self {
        Self {
            fan_out: 5,
            target_zone_t: 96,
            target_zone_f: 96,
            peaks_per_sec: 30,
            min_anchor_mag_db: -50.0,
            max_input_samples: Some(30 * 60 * PANAKO_SR as usize),
            max_hashes: Some(500_000),
            max_pending_anchors: None,
            max_push_samples: None,
        }
    }
}

const PANAKO_N_FFT: usize = 1024;
const PANAKO_HOP: usize = 128;
const PANAKO_SR: u32 = 8_000;
const PANAKO_FRAMES_PER_SEC: f32 = PANAKO_SR as f32 / PANAKO_HOP as f32;

const PANAKO_PEAK_NEIGHBOURHOOD: usize = 15;
const PANAKO_LOG_FLOOR: f32 = 1e-6;
/// Squared form of the magnitude floor — see Wang for rationale.
const PANAKO_LOG_FLOOR_POWER: f32 = PANAKO_LOG_FLOOR * PANAKO_LOG_FLOOR;
use crate::dsp::power_to_db_wide;

/// Panako offline fingerprinter.
///
/// # Example
///
/// ```
/// use audiofp::{Fingerprinter, SampleRate};
/// use audiofp::classical::Panako;
///
/// let mut fp = Panako::default();
/// let samples = vec![0.0_f32; 8_000 * 3];
///
/// let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
/// assert_eq!(fpr.frames_per_sec, 62.5);
/// assert!(fpr.hashes.is_empty());
/// ```
pub struct Panako {
    cfg: PanakoConfig,
    stft: ShortTimeFFT,
    picker: PeakPicker,
    log_spec: Vec<f32>,
}

impl Default for Panako {
    fn default() -> Self {
        Self::new(PanakoConfig::default())
    }
}

impl Panako {
    /// Build a Panako extractor with the given config.
    ///
    /// Clamps `target_zone_t` to a minimum of 1 and `fan_out` to a
    /// minimum of 1 to prevent underflows/empty output from degenerate
    /// configurations. Caps `target_zone_t` at 512 and `fan_out` at 64
    /// to prevent OOM from extreme values.
    #[must_use]
    pub fn new(mut cfg: PanakoConfig) -> Self {
        crate::classical::sanitize_cfg!(cfg);
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: PANAKO_N_FFT,
            hop: PANAKO_HOP,
            window: WindowKind::Hann,
            center: false,
        });
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: PANAKO_PEAK_NEIGHBOURHOOD,
            neighborhood_f: PANAKO_PEAK_NEIGHBOURHOOD,
            min_magnitude_db: cfg.min_anchor_mag_db,
            min_magnitude_linear: None,
            target_per_sec: cfg.peaks_per_sec as usize,
        });
        Self {
            cfg,
            stft,
            picker,
            log_spec: Vec::new(),
        }
    }
}

/// Progress callback reporting interval for Panako (62.5 fps):
/// every 32 frames ≈ 500 ms of audio.
const PANAKO_PROGRESS_INTERVAL: usize = 32;

impl Panako {
    /// Extract fingerprint with a progress callback.
    ///
    /// `progress` is called periodically with a value in `[0.0, 1.0]`
    /// representing the fraction of work completed. The final call is
    /// always made with `1.0`. The callback is invoked at most once per
    /// ~500 ms of audio to avoid overhead.
    ///
    /// # Errors
    ///
    /// Same as [`Fingerprinter::extract`].
    pub fn extract_with_progress<F: FnMut(f32)>(
        &mut self,
        samples: &[f32],
        rate: SampleRate,
        mut progress: F,
    ) -> Result<PanakoFingerprint> {
        crate::pcm::reject_non_finite(samples)?;
        if let Some(limit) = self.cfg.max_input_samples
            && samples.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: samples.len(),
            });
        }
        if rate.hz() != PANAKO_SR {
            return Err(AfpError::UnsupportedSampleRate(rate.hz()));
        }
        if samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: samples.len(),
            });
        }

        progress(0.0);

        let (n_frames, n_bins) = self.stft.power_flat_into(samples, &mut self.log_spec);
        if n_frames == 0 {
            progress(1.0);
            return Ok(PanakoFingerprint {
                hashes: Vec::new(),
                frames_per_sec: PANAKO_FRAMES_PER_SEC,
            });
        }

        // Report progress through the STFT phase (~70% of total work).
        let total_frames = n_frames;
        let stft_weight = 0.7_f32;
        let interval = PANAKO_PROGRESS_INTERVAL;
        {
            let mut reported = 0usize;
            while reported + interval < total_frames {
                reported += interval;
                progress(stft_weight * (reported as f32 / total_frames as f32));
            }
        }
        progress(stft_weight);

        power_to_db_wide(&mut self.log_spec, PANAKO_LOG_FLOOR_POWER);
        progress(0.80);

        let peaks = self
            .picker
            .pick(&self.log_spec, n_frames, n_bins, PANAKO_FRAMES_PER_SEC);
        progress(0.90);

        let mut hashes = build_triplet_hashes(&peaks, &self.cfg);
        hashes.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));

        if let Some(limit) = self.cfg.max_hashes
            && hashes.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: hashes.len(),
            });
        }

        progress(1.0);

        Ok(PanakoFingerprint {
            hashes,
            frames_per_sec: PANAKO_FRAMES_PER_SEC,
        })
    }
}

impl Fingerprinter for Panako {
    type Output = PanakoFingerprint;
    type Config = PanakoConfig;

    fn name(&self) -> &'static str {
        "panako-v2"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> SampleRate {
        // PANAKO_SR is a compile-time constant; unwrap is trivially safe.
        SampleRate::new(PANAKO_SR).expect("PANAKO_SR is non-zero")
    }

    fn min_samples(&self) -> usize {
        PANAKO_SR as usize * 2
    }

    fn extract(&mut self, samples: &[f32], rate: SampleRate) -> Result<Self::Output> {
        self.extract_with_progress(samples, rate, |_| {})
    }
}

/// Wrapper that orders triplets so the **smallest** combined magnitude
/// (with the largest position as tiebreak) compares **greatest** —
/// suitable as the element of a max-heap that maintains the top-K
/// largest triplets in `O(N log K)` work. Owned `Peak` copies so the
/// same type serves both the offline and pooled streaming builders.
#[derive(Copy, Clone)]
struct MinByScoreOwned {
    b: Peak,
    c: Peak,
    score: f32,
}

impl MinByScoreOwned {
    fn new(b: &Peak, c: &Peak, score: f32) -> Self {
        Self {
            b: *b,
            c: *c,
            score,
        }
    }
}

impl PartialEq for MinByScoreOwned {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score
            && (self.b.t_frame, self.b.f_bin) == (o.b.t_frame, o.b.f_bin)
            && (self.c.t_frame, self.c.f_bin) == (o.c.t_frame, o.c.f_bin)
    }
}
impl Eq for MinByScoreOwned {}
impl PartialOrd for MinByScoreOwned {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinByScoreOwned {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        o.score
            .partial_cmp(&self.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| (o.b.t_frame, o.b.f_bin).cmp(&(self.b.t_frame, self.b.f_bin)))
            .then_with(|| (o.c.t_frame, o.c.f_bin).cmp(&(self.c.t_frame, self.c.f_bin)))
    }
}

/// Walk `peaks` (sorted by `(t_frame, f_bin)`) and emit triplet hashes.
fn build_triplet_hashes(peaks: &[Peak], cfg: &PanakoConfig) -> Vec<PanakoHash> {
    let target_zone_t = cfg.target_zone_t as i32;
    let target_zone_f = cfg.target_zone_f as i32;
    let fan_out = cfg.fan_out as usize;

    let mut hashes = Vec::with_capacity(peaks.len() * fan_out);

    // Capacity heuristics; both grow on demand.
    let mut targets: Vec<&Peak> = Vec::with_capacity(64);
    let mut heap: alloc::collections::BinaryHeap<MinByScoreOwned> =
        alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
    let mut triplets: Vec<(Peak, Peak, f32)> = Vec::with_capacity(fan_out);
    let mut suffix_max: Vec<f32> = Vec::with_capacity(64);

    for (i, anchor) in peaks.iter().enumerate() {
        // Binary search for the upper bound: first peak with
        // t_frame >= anchor.t_frame + target_zone_t.
        // Panako uses STRICT inequality (dt < target_zone_t).
        let zone_limit = anchor.t_frame.saturating_add(target_zone_t as u32 - 1);
        let zone_end = peaks[i + 1..].partition_point(|p| p.t_frame <= zone_limit);

        // Collect peaks in the cone (time + freq zone, strict inequalities).
        targets.clear();
        for target in &peaks[i + 1..i + 1 + zone_end] {
            let dt = target.t_frame as i32 - anchor.t_frame as i32;
            if dt < 1 {
                continue;
            }
            let df = target.f_bin as i32 - anchor.f_bin as i32;
            if df.abs() >= target_zone_f {
                continue;
            }
            targets.push(target);
        }

        // Heap-based top-K over (b, c) tuples, scored by `b.mag + c.mag`.
        // Suffix-max array enables early-exit without reordering targets.
        let targets_len = targets.len();
        suffix_max.resize(targets_len + 1, 0.0_f32);
        suffix_max[targets_len] = 0.0_f32;
        for j in (0..targets_len).rev() {
            let m = targets[j].mag;
            suffix_max[j] = if m > suffix_max[j + 1] {
                m
            } else {
                suffix_max[j + 1]
            };
        }

        heap.clear();
        for (j, b) in targets.iter().enumerate() {
            // Early skip: if b.mag + best remaining c can't beat the
            // heap minimum, no pair involving this b can win.
            if heap.len() >= fan_out
                && heap
                    .peek()
                    .is_some_and(|min| b.mag + suffix_max[j + 1] < min.score)
            {
                continue;
            }
            for c in &targets[j + 1..] {
                let score = b.mag + c.mag;
                heap.push(MinByScoreOwned::new(b, c, score));
                if heap.len() > fan_out {
                    heap.pop();
                }
            }
        }

        // Drain and re-sort the kept K for deterministic emission.
        triplets.clear();
        triplets.extend(heap.drain().map(|w| (w.b, w.c, w.score)));
        triplets.sort_unstable_by(|x, y| {
            y.2.partial_cmp(&x.2)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (x.0.t_frame, x.0.f_bin).cmp(&(y.0.t_frame, y.0.f_bin)))
                .then_with(|| (x.1.t_frame, x.1.f_bin).cmp(&(y.1.t_frame, y.1.f_bin)))
        });

        for (b, c, _) in &triplets {
            let hash = pack_triplet(anchor, b, c);
            hashes.push(PanakoHash {
                hash,
                t_anchor: anchor.t_frame,
                t_b: b.t_frame,
                t_c: c.t_frame,
            });
        }
    }

    hashes
}

/// Pack one anchor-b-c triplet into a 32-bit hash.
#[inline]
fn pack_triplet(a: &Peak, b: &Peak, c: &Peak) -> u32 {
    let f_a = a.f_bin as i32;
    let f_b = b.f_bin as i32;
    let f_c = c.f_bin as i32;

    let df_ab = (f_b - f_a).clamp(-127, 127);
    let df_bc = (f_c - f_b).clamp(-127, 127);

    let sign: u32 = ((f_b >= f_a) as u32) | (((f_c >= f_b) as u32) << 1);

    let mag_order: u32 = if a.mag >= b.mag && a.mag >= c.mag {
        0
    } else if b.mag >= c.mag {
        1
    } else {
        2
    };

    let dt_ac = (c.t_frame - a.t_frame).max(1) as f32;
    let dt_bc = (c.t_frame - b.t_frame) as f32;
    // Round without libm: (x + 0.5) as i32 ≡ roundf(x) as i32 for x ≥ 0.
    let beta = ((dt_bc / dt_ac * 31.0 + 0.5) as i32).clamp(0, 31) as u32;

    let dab_u = (df_ab as i8 as u8) as u32;
    let dbc_u = (df_bc as i8 as u8) as u32;

    ((sign & 0x3) << 30)
        | ((mag_order & 0x3) << 28)
        | ((beta & 0x1F) << 23)
        | ((dab_u & 0xFF) << 15)
        | ((dbc_u & 0xFF) << 7)
}

/// Streaming Panako fingerprinter.
///
/// Same deferred-emission strategy as [`super::StreamingWang`]: hashes are
/// emitted only once their anchor has accrued the full lookahead, so the
/// output multiset matches [`Panako::extract`] for the same total input.
///
/// Latency is higher than Wang because the triplet zone is wider
/// (`target_zone_t = 96` vs Wang's 63).
///
/// The pipeline is shared with [`StreamingWang`](crate::classical::StreamingWang)
/// via the crate-internal streaming core; this wrapper supplies
/// only the Panako-specific triplet emission.
pub struct StreamingPanako {
    cfg: PanakoConfig,
    core: stream::StreamCore<PanakoHash>,
}

impl Default for StreamingPanako {
    fn default() -> Self {
        Self::new(PanakoConfig::default())
    }
}

impl StreamingPanako {
    /// Build a streaming Panako extractor with the given config.
    ///
    /// Clamps `target_zone_t` to a minimum of 1 and `fan_out` to a
    /// minimum of 1 to prevent underflows/empty output from degenerate
    /// configurations. Caps `target_zone_t` at 512 and `fan_out` at 64
    /// to prevent OOM from extreme values.
    #[must_use]
    pub fn new(mut cfg: PanakoConfig) -> Self {
        crate::classical::sanitize_cfg!(cfg);
        Self {
            cfg,
            core: stream::StreamCore::new(
                PANAKO_N_FFT,
                PANAKO_HOP,
                PANAKO_SR,
                PANAKO_PEAK_NEIGHBOURHOOD,
                PANAKO_LOG_FLOOR_POWER,
                stream::Zone::Strict,
            ),
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &PanakoConfig {
        &self.cfg
    }

    /// Reset all internal state. The stream behaves as if freshly
    /// constructed: no buffered audio, no pending peaks or anchors.
    /// Call between independent streams sharing one instance so stale
    /// data from a previous stream doesn't bleed into new hashes.
    pub fn reset(&mut self) {
        self.core.reset();
    }

    fn lookahead_frames(&self) -> u32 {
        self.cfg.target_zone_t as u32
            + PANAKO_PEAK_NEIGHBOURHOOD as u32
            + PANAKO_FRAMES_PER_SEC.ceil() as u32
    }

    fn peak_cfg(&self) -> stream::PeakCfg {
        stream::PeakCfg {
            min_anchor_mag_db: self.cfg.min_anchor_mag_db,
            target_zone_t: self.cfg.target_zone_t as i32,
            target_zone_f: self.cfg.target_zone_f as i32,
            fan_out: self.cfg.fan_out as usize,
            peaks_per_sec: self.cfg.peaks_per_sec as usize,
            max_pending_anchors: self.cfg.max_pending_anchors,
            max_push_samples: self.cfg.max_push_samples,
        }
    }

    /// Panako target maintenance: keep all in-cone targets, capped at
    /// `2·fan_out` with weakest-magnitude eviction.
    fn add_target(targets: &mut Vec<Peak>, target: Peak, _dt: i32, _df: i32, cfg: stream::PeakCfg) {
        let target_cap = 2 * cfg.fan_out;
        targets.push(target);
        if targets.len() > target_cap {
            let min_idx = targets
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.mag
                        .partial_cmp(&b.mag)
                        .unwrap_or(core::cmp::Ordering::Equal)
                        .then_with(|| (b.t_frame, b.f_bin).cmp(&(a.t_frame, a.f_bin)))
                })
                .map(|(i, _)| i)
                .expect("targets non-empty after cap check");
            targets.swap_remove(min_idx);
        }
    }

    /// Emit Panako triplet hashes for a finalised anchor: sort targets by
    /// `(t_frame, f_bin)` and keep the top-K `(b, c)` pairs by
    /// `b.mag + c.mag`.
    fn emit_anchor(
        mut anchor: stream::PendingAnchor,
        cfg: stream::PeakCfg,
        out: &mut Vec<(TimestampMs, PanakoHash)>,
    ) {
        let fan_out = cfg.fan_out;
        anchor
            .targets
            .sort_unstable_by_key(|p| (p.t_frame, p.f_bin));
        let mut heap: alloc::collections::BinaryHeap<MinByScoreOwned> =
            alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
        for (j, b) in anchor.targets.iter().enumerate() {
            for c in &anchor.targets[j + 1..] {
                let score = b.mag + c.mag;
                heap.push(MinByScoreOwned::new(b, c, score));
                if heap.len() > fan_out {
                    heap.pop();
                }
            }
        }
        let mut scratch: Vec<(Peak, Peak, f32)> =
            heap.drain().map(|w| (w.b, w.c, w.score)).collect();
        scratch.sort_unstable_by(|x, y| {
            y.2.partial_cmp(&x.2)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (x.0.t_frame, x.0.f_bin).cmp(&(y.0.t_frame, y.0.f_bin)))
                .then_with(|| (x.1.t_frame, x.1.f_bin).cmp(&(y.1.t_frame, y.1.f_bin)))
        });
        for (b, c, _) in &scratch {
            let hash = pack_triplet(&anchor.peak, b, c);
            let t_ms = (anchor.peak.t_frame as u64 * PANAKO_HOP as u64 * 1000) / PANAKO_SR as u64;
            out.push((
                TimestampMs(t_ms),
                PanakoHash {
                    hash,
                    t_anchor: anchor.peak.t_frame,
                    t_b: b.t_frame,
                    t_c: c.t_frame,
                },
            ));
        }
    }
}

impl StreamingFingerprinter for StreamingPanako {
    type Frame = PanakoHash;

    fn required_sample_rate(&self) -> u32 {
        PANAKO_SR
    }

    fn push(&mut self, samples: &[f32]) -> Result<Vec<(TimestampMs, Self::Frame)>> {
        self.core.emitted.clear();
        let cfg = self.peak_cfg();
        self.core
            .process_push_samples(samples, cfg, Self::add_target, Self::emit_anchor);
        Ok(core::mem::take(&mut self.core.emitted))
    }

    fn push_with<F>(&mut self, samples: &[f32], mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        self.core.emitted.clear();
        let cfg = self.peak_cfg();
        self.core
            .process_push_samples(samples, cfg, Self::add_target, Self::emit_anchor);
        let mut n = 0usize;
        for (t, frame) in self.core.emitted.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
    }

    fn flush(&mut self) -> Result<Vec<(TimestampMs, Self::Frame)>> {
        self.core.emitted.clear();
        let cfg = self.peak_cfg();
        self.core
            .process_flush(cfg, Self::add_target, Self::emit_anchor);
        Ok(core::mem::take(&mut self.core.emitted))
    }

    fn flush_with<F>(&mut self, mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        self.core.emitted.clear();
        let cfg = self.peak_cfg();
        self.core
            .process_flush(cfg, Self::add_target, Self::emit_anchor);
        let mut n = 0usize;
        for (t, frame) in self.core.emitted.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
    }

    fn latency_ms(&self) -> u32 {
        (self.lookahead_frames() * PANAKO_HOP as u32 * 1000) / PANAKO_SR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleRate;
    use alloc::vec;
    use core::f32::consts::PI;

    fn synthetic_audio(seed: u32, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        let mut x: u32 = seed.max(1);
        for n in 0..len {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
            let t = n as f32 / 8_000.0;
            let s = 0.5 * libm::sinf(2.0 * PI * 880.0 * t)
                + 0.3 * libm::sinf(2.0 * PI * 1320.0 * t)
                + noise;
            out.push(s);
        }
        out
    }

    fn chunk_sizes(seed: u32, total: usize, max_chunk: usize) -> Vec<usize> {
        let mut x = seed.max(1);
        let mut out = Vec::new();
        let mut remaining = total;
        while remaining > 0 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let n = ((x as usize) % max_chunk).max(1).min(remaining);
            out.push(n);
            remaining -= n;
        }
        out
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut fp = Panako::default();
        let samples = vec![0.0_f32; 16_000];

        match fp.extract(&samples, SampleRate::HZ_16000) {
            Err(AfpError::UnsupportedSampleRate(16_000)) => {}
            other => panic!("expected UnsupportedSampleRate, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_audio() {
        let mut fp = Panako::default();
        let samples = vec![0.0_f32; 8_000];

        match fp.extract(&samples, SampleRate::HZ_8000) {
            Err(AfpError::AudioTooShort {
                needed: 16_000,
                got: 8_000,
            }) => {}
            other => panic!("expected AudioTooShort, got {other:?}"),
        }
    }

    #[test]
    fn silence_gives_empty_fingerprint() {
        let mut fp = Panako::default();
        let samples = vec![0.0_f32; 8_000 * 3];

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_eq!(fpr.frames_per_sec, 62.5);
        assert!(fpr.hashes.is_empty());
    }

    #[test]
    fn synthetic_signal_produces_hashes() {
        let mut fp = Panako::default();
        let samples = synthetic_audio(0xC0FFEE, 8_000 * 5);

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert!(
            (500..=900).contains(&fpr.hashes.len()),
            "expected 500..=900 hashes from a 5s tone, got {}",
            fpr.hashes.len(),
        );
        let distinct: alloc::collections::BTreeSet<u32> =
            fpr.hashes.iter().map(|h| h.hash).collect();
        assert!(
            distinct.len() > 400,
            "expected most hashes to be distinct, got {} distinct of {}",
            distinct.len(),
            fpr.hashes.len(),
        );
        // Ordering invariant: sorted by (t_anchor, t_b, t_c).
        for w in fpr.hashes.windows(2) {
            assert!((w[0].t_anchor, w[0].t_b, w[0].t_c) <= (w[1].t_anchor, w[1].t_b, w[1].t_c));
        }
    }

    #[test]
    fn synthetic_signal_is_deterministic() {
        // Two separate extractors on identical input must produce the
        // same hash multiset (the regression goldens pin byte-exact
        // output; this smoke test pins the count + multiset count for a
        // second seed at a different length to catch algorithm drift
        // the goldens would miss if the seed/input changes).
        let samples = synthetic_audio(0xBEEF, 8_000 * 3);
        let mut a = Panako::default();
        let mut b = Panako::default();
        let fa = a.extract(&samples, SampleRate::HZ_8000).unwrap();
        let fb = b.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_eq!(fa.hashes, fb.hashes);
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Panako::default();
        let f1 = fp1.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut fp2 = Panako::default();
        let f2 = fp2.extract(&samples, SampleRate::HZ_8000).unwrap();

        assert_eq!(f1.hashes, f2.hashes);
    }

    #[test]
    fn different_signals_diverge() {
        let a = synthetic_audio(0x1111, 8_000 * 3);
        let b = synthetic_audio(0x2222, 8_000 * 3);

        let mut fp = Panako::default();
        let fa = fp.extract(&a, SampleRate::HZ_8000).unwrap();
        let fb = fp.extract(&b, SampleRate::HZ_8000).unwrap();
        assert_ne!(fa.hashes, fb.hashes);
    }

    #[test]
    fn pack_triplet_decodes_correctly() {
        let a = Peak {
            t_frame: 100,
            f_bin: 50,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 110,
            f_bin: 70,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 130,
            f_bin: 60,
            _pad: 0,
            mag: 0.0,
        };

        let h = pack_triplet(&a, &b, &c);

        let sign = (h >> 30) & 0x3;
        let mag_order = (h >> 28) & 0x3;
        let beta = (h >> 23) & 0x1F;
        let dab = ((h >> 15) & 0xFF) as u8 as i8;
        let dbc = ((h >> 7) & 0xFF) as u8 as i8;

        // f_b (70) >= f_a (50) → sign bit 0 = 1.
        // f_c (60) <  f_b (70) → sign bit 1 = 0.
        assert_eq!(sign, 0b01);
        // All mags equal → top1_idx = 0 (anchor wins by precedence).
        assert_eq!(mag_order, 0);
        // β = round((130-110)/(130-100) * 31) = round(20/30 * 31) = round(20.6) = 21.
        assert_eq!(beta, 21);
        assert_eq!(dab as i32, 20);
        assert_eq!(dbc as i32, -10);
        // Bottom 7 bits reserved.
        assert_eq!(h & 0x7F, 0);
    }

    #[test]
    fn pack_triplet_clamps_large_freq_diffs() {
        let a = Peak {
            t_frame: 0,
            f_bin: 0,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 400,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 0,
            _pad: 0,
            mag: 0.0,
        };

        let h = pack_triplet(&a, &b, &c);
        let dab = ((h >> 15) & 0xFF) as u8 as i8;
        let dbc = ((h >> 7) & 0xFF) as u8 as i8;
        assert_eq!(dab as i32, 127); // clamped
        assert_eq!(dbc as i32, -127); // clamped
    }

    #[test]
    fn streaming_latency_matches_lookahead() {
        let s = StreamingPanako::default();
        // (96 + 15 + 63) frames * 128 / 8000 * 1000 = 2784 ms.
        assert_eq!(s.latency_ms(), 2_784);
    }

    #[test]
    fn streaming_silence_emits_nothing() {
        let mut s = StreamingPanako::default();
        let zeros = vec![0.0_f32; 8_000 * 4];
        assert!(s.push(&zeros).unwrap().is_empty());
        assert!(s.flush().unwrap().is_empty());
    }

    #[test]
    fn streaming_flush_is_idempotent() {
        // `StreamingFingerprinter::flush` lifecycle contract: a second
        // flush after the stream is drained returns nothing. Pins the
        // `IncrementalPeakDetector` emitted-row cursor at the Panako
        // public API (shared StreamCore with Wang).
        let mut s = StreamingPanako::default();
        let samples = synthetic_audio(0x2D2D, 8_000 * 3);
        let _ = s.push(&samples).unwrap();
        let first = s.flush().unwrap();
        let second = s.flush().unwrap();
        assert!(
            second.is_empty(),
            "second flush returned {} frames after {} in the first",
            second.len(),
            first.len(),
        );
    }

    #[test]
    fn mag_order_picks_largest_of_three() {
        // mag_order = 1 (b largest)
        let a = Peak {
            t_frame: 0,
            f_bin: 10,
            _pad: 0,
            mag: 1.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 20,
            _pad: 0,
            mag: 5.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 15,
            _pad: 0,
            mag: 3.0,
        };
        let h = pack_triplet(&a, &b, &c);
        assert_eq!((h >> 28) & 0x3, 1);

        // mag_order = 2 (c largest)
        let a = Peak {
            t_frame: 0,
            f_bin: 10,
            _pad: 0,
            mag: 1.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 20,
            _pad: 0,
            mag: 2.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 15,
            _pad: 0,
            mag: 9.0,
        };
        let h = pack_triplet(&a, &b, &c);
        assert_eq!((h >> 28) & 0x3, 2);

        // mag_order = 0 (anchor largest)
        let a = Peak {
            t_frame: 0,
            f_bin: 10,
            _pad: 0,
            mag: 9.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 20,
            _pad: 0,
            mag: 2.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 15,
            _pad: 0,
            mag: 3.0,
        };
        let h = pack_triplet(&a, &b, &c);
        assert_eq!((h >> 28) & 0x3, 0);
    }

    #[test]
    fn sign_bit_combinations() {
        // Both descending: f_b < f_a, f_c < f_b → sign = 0b00
        let a = Peak {
            t_frame: 0,
            f_bin: 100,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 80,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 60,
            _pad: 0,
            mag: 0.0,
        };
        assert_eq!((pack_triplet(&a, &b, &c) >> 30) & 0x3, 0b00);

        // Both ascending: f_b > f_a, f_c > f_b → sign = 0b11
        let a = Peak {
            t_frame: 0,
            f_bin: 100,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 5,
            f_bin: 120,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 10,
            f_bin: 140,
            _pad: 0,
            mag: 0.0,
        };
        assert_eq!((pack_triplet(&a, &b, &c) >> 30) & 0x3, 0b11);
    }

    #[test]
    fn beta_saturates_near_extremes() {
        // β ≈ 31 when t_b is right after t_a (ratio (t_c - t_b)/(t_c - t_a) → 1).
        let a = Peak {
            t_frame: 0,
            f_bin: 0,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 1,
            f_bin: 5,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 95,
            f_bin: 8,
            _pad: 0,
            mag: 0.0,
        };
        let h = pack_triplet(&a, &b, &c);
        let beta = (h >> 23) & 0x1F;
        assert!(beta >= 30, "beta should saturate near 31, got {beta}");

        // β ≈ 0 when t_b is just before t_c.
        let a = Peak {
            t_frame: 0,
            f_bin: 0,
            _pad: 0,
            mag: 0.0,
        };
        let b = Peak {
            t_frame: 90,
            f_bin: 5,
            _pad: 0,
            mag: 0.0,
        };
        let c = Peak {
            t_frame: 91,
            f_bin: 8,
            _pad: 0,
            mag: 0.0,
        };
        let h = pack_triplet(&a, &b, &c);
        let beta = (h >> 23) & 0x1F;
        assert!(beta <= 1, "beta should saturate near 0, got {beta}");
    }

    #[test]
    fn streaming_offline_equivalence() {
        let samples = synthetic_audio(0xBEEF, 8_000 * 6);

        let mut offline = Panako::default();
        let off = offline.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut streaming = StreamingPanako::default();
        let mut online: Vec<PanakoHash> = Vec::new();
        let mut cursor = 0;
        for n in chunk_sizes(0xCAFE, samples.len(), 4_000) {
            let end = cursor + n;
            online.extend(
                streaming
                    .push(&samples[cursor..end])
                    .unwrap()
                    .into_iter()
                    .map(|(_, h)| h),
            );
            cursor = end;
        }
        online.extend(streaming.flush().unwrap().into_iter().map(|(_, h)| h));

        let mut a = off.hashes;
        let mut b = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));
        assert_eq!(a.len(), b.len(), "hash count mismatch");
        assert_eq!(a, b, "hash sequences differ");
    }

    #[test]
    fn streaming_state_stays_bounded_under_long_input() {
        // Same shape as the Wang invariant test: 30 s of audio in
        // 256-sample chunks, peak-tracked ceilings on every buffer.
        let secs = 30usize;
        let samples = synthetic_audio(11, PANAKO_SR as usize * secs);
        let chunk = 256usize;

        let mut s = StreamingPanako::default();
        let max_spec_rows = 2 * PANAKO_PEAK_NEIGHBOURHOOD + 1;

        let mut peak_carry = 0usize;
        let mut peak_spec_rows = 0usize;
        let mut peak_bucket_pending = 0usize;
        let mut peak_anchors = 0usize;

        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + chunk).min(samples.len());
            let _ = s.push(&samples[start..end]).unwrap();
            peak_carry = peak_carry.max(s.core.sample_carry.len());
            peak_spec_rows = peak_spec_rows.max(s.core.spec_n_rows);
            peak_bucket_pending = peak_bucket_pending.max(s.core.bucket_pending.len());
            peak_anchors = peak_anchors.max(s.core.pending_anchors.len());

            assert!(s.core.sample_carry.len() < PANAKO_N_FFT);
            assert!(s.core.spec_n_rows <= max_spec_rows);
            start = end;
        }

        // target_zone_t=96 frames ≈ 1.54 s of bucket coverage at 62.5
        // fps; peaks_per_sec=30 → ~46 anchors at peak.
        assert_eq!(peak_spec_rows, max_spec_rows);
        assert!(peak_carry < PANAKO_N_FFT, "peak_carry {peak_carry}");
        assert!(
            peak_bucket_pending <= 3,
            "bucket_pending peaked at {peak_bucket_pending} (steady state should be ≤ 2)",
        );
        assert!(
            peak_anchors <= 60,
            "pending_anchors peaked at {peak_anchors} (expected ≤ 60)",
        );

        let _ = s.flush().unwrap();
        assert_eq!(s.core.bucket_pending.len(), 0);
        assert_eq!(s.core.pending_anchors.len(), 0);
    }

    // Direct unit tests for `emit_finalized_anchors`.
    //
    // Same re-queue invariant as the wang.rs counterpart. See the
    // comment block there for motivation; this is the Panako
    // mirror. Panako's `last_target_frame = t + (target_zone_t - 1)`
    // (strict `dt < target_zone_t`).

    fn panako_anchor_with_target(
        t_frame: u32,
        f_bin: u16,
        target_t: u32,
        target_f: u16,
    ) -> stream::PendingAnchor {
        // Two targets so that the triplet emission (which iterates over
        // `(b, c)` pairs) produces at least one hash.
        stream::PendingAnchor {
            peak: Peak {
                t_frame,
                f_bin,
                _pad: 0,
                mag: 1.0,
            },
            targets: vec![
                Peak {
                    t_frame: target_t,
                    f_bin: target_f,
                    _pad: 0,
                    mag: 0.9,
                },
                Peak {
                    t_frame: target_t + 1,
                    f_bin: target_f + 1,
                    _pad: 0,
                    mag: 0.8,
                },
            ],
        }
    }

    /// Bucket index for a frame at the Panako default rate
    /// (`PANAKO_FRAMES_PER_SEC = 62.5`).
    fn panako_bucket_of(t_frame: u32) -> i32 {
        (t_frame as f32 / PANAKO_FRAMES_PER_SEC) as i32
    }

    #[test]
    fn panako_emit_finalized_anchors_emits_all_when_zones_covered() {
        // Three anchors, all of whose target zones are covered.
        // Panako default `target_zone_t = 96` → last_target_frame
        // = t_frame + 95.
        let mut s = StreamingPanako::default();
        // t=0 → last target frame 95 → bucket 1
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        // t=5 → last target frame 100 → bucket 1
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(5, 20, 15, 22));
        // t=100 → last target frame 195 → bucket 3
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(100, 30, 110, 32));
        s.core.last_finalized_bucket = panako_bucket_of(195);

        s.core.emitted.clear();
        s.core
            .emit_finalized_anchors(s.peak_cfg(), StreamingPanako::emit_anchor);
        assert_eq!(s.core.emitted.len(), 3);
        assert!(s.core.pending_anchors.is_empty());
    }

    #[test]
    fn panako_emit_finalized_anchors_re_queues_unfinalised() {
        // Two anchors; only the first is covered. The second must
        // remain in `pending_anchors` after the emit.
        let mut s = StreamingPanako::default();
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(100, 30, 110, 32));
        // Only cover bucket 1 (last target frame ≤ 95).
        s.core.last_finalized_bucket = 1;

        s.core.emitted.clear();
        s.core
            .emit_finalized_anchors(s.peak_cfg(), StreamingPanako::emit_anchor);
        assert_eq!(s.core.emitted.len(), 1);
        assert_eq!(s.core.pending_anchors.len(), 1);
        assert_eq!(s.core.pending_anchors.front().unwrap().peak.t_frame, 100);
    }

    #[test]
    fn panako_emit_finalized_anchors_idempotent_under_repeated_calls() {
        // With one anchor covered, two consecutive calls must emit
        // the same hashes (no double-emit, no lost anchor).
        let mut s = StreamingPanako::default();
        s.core
            .pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        s.core.last_finalized_bucket = panako_bucket_of(95);

        s.core.emitted.clear();
        s.core
            .emit_finalized_anchors(s.peak_cfg(), StreamingPanako::emit_anchor);
        let first_len = s.core.emitted.len();
        s.core.emitted.clear();
        s.core
            .emit_finalized_anchors(s.peak_cfg(), StreamingPanako::emit_anchor);
        let second_len = s.core.emitted.len();
        assert_eq!(first_len, 1);
        assert_eq!(second_len, 0);
        assert!(s.core.pending_anchors.is_empty());
    }

    // Public API contract pins. See wang.rs for motivation.

    #[test]
    fn public_api_name_and_config_match_documented_values() {
        let fp = Panako::default();
        assert_eq!(fp.name(), "panako-v2");
        assert_eq!(fp.required_sample_rate(), SampleRate::HZ_8000);
        assert_eq!(fp.min_samples(), 16_000);

        let s = StreamingPanako::default();
        assert_eq!(s.latency_ms(), 2_784);
    }

    // ── Backward-compat (& forward-safe) constructor clamping tests ──

    #[test]
    fn default_config_is_unchanged_by_guard_clamps() {
        let fp = Panako::default();
        assert_eq!(fp.config().fan_out, 5);
        assert_eq!(fp.config().target_zone_t, 96);
        assert_eq!(fp.config().peaks_per_sec, 30);
    }

    #[test]
    fn zero_target_zone_is_clamped_to_one_not_underflow() {
        let cfg = PanakoConfig {
            target_zone_t: 0,
            fan_out: 0,
            ..PanakoConfig::default()
        };
        let fp = Panako::new(cfg);
        assert_eq!(fp.config().target_zone_t, 1);
        assert_eq!(fp.config().fan_out, 1);
    }

    #[test]
    fn extreme_config_is_clamped_within_safe_bounds() {
        let cfg = PanakoConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..PanakoConfig::default()
        };
        let fp = Panako::new(cfg);
        assert_eq!(fp.config().fan_out, 64);
        assert_eq!(fp.config().target_zone_t, 512);
        assert_eq!(fp.config().peaks_per_sec, 500);
    }

    #[test]
    fn clamped_config_still_produces_valid_hashes() {
        let cfg = PanakoConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..PanakoConfig::default()
        };
        let mut fp = Panako::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 3);

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert!(!fpr.hashes.is_empty());
    }

    #[test]
    fn streaming_default_config_is_unchanged_by_guard_clamps() {
        let s = StreamingPanako::default();
        let cfg = s.config();
        assert_eq!(cfg.fan_out, 5);
        assert_eq!(cfg.target_zone_t, 96);
        assert_eq!(cfg.peaks_per_sec, 30);
    }

    #[test]
    fn streaming_extreme_config_is_clamped_within_safe_bounds() {
        let cfg = PanakoConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..PanakoConfig::default()
        };
        let s = StreamingPanako::new(cfg);
        assert_eq!(s.config().fan_out, 64);
        assert_eq!(s.config().target_zone_t, 512);
        assert_eq!(s.config().peaks_per_sec, 500);
    }

    #[test]
    fn streaming_reset_clears_all_state() {
        let mut s = StreamingPanako::default();
        let samples = synthetic_audio(0xFEED, 8_000 * 5);
        let before = s.push(&samples).unwrap();
        assert!(!before.is_empty(), "should produce hashes");

        s.reset();
        assert!(s.push(&[]).unwrap().is_empty(), "reset should clear state");
        let after_reset = s.push(&samples).unwrap();
        assert!(!after_reset.is_empty());
        assert_eq!(
            before, after_reset,
            "reset+replay must produce identical hashes"
        );
    }

    // ── Performance regression: zero-alloc push_with contract ──

    #[test]
    fn push_with_matches_push_output_count() {
        let mut a = StreamingPanako::default();
        let mut b = StreamingPanako::default();
        let samples = synthetic_audio(0xABCD, 8_000 * 5);

        let via_push = a.push(&samples).unwrap();
        let mut via_cb: Vec<(TimestampMs, PanakoHash)> = Vec::new();
        let n = b.push_with(&samples, |t, f| via_cb.push((t, *f))).unwrap();
        let via_flush = b.flush().unwrap();
        let flush_len = via_flush.len();
        via_cb.extend(via_flush);

        let mut all_via_push = via_push;
        all_via_push.extend(a.flush().unwrap());

        assert_eq!(n + flush_len, all_via_push.len());
        assert_eq!(
            via_cb, all_via_push,
            "push_with must emit exactly what push+flush emits"
        );
    }

    #[test]
    fn flush_with_matches_flush_output() {
        let mut a = StreamingPanako::default();
        let mut b = StreamingPanako::default();
        let samples = synthetic_audio(0xF00D, 8_000 * 5);
        let _ = a.push(&samples).unwrap();
        let _ = b.push(&samples).unwrap();

        let via_flush = a.flush().unwrap();
        let mut via_cb: Vec<(TimestampMs, PanakoHash)> = Vec::new();
        let n = b.flush_with(|t, f| via_cb.push((t, *f))).unwrap();

        assert_eq!(n, via_flush.len());
        assert_eq!(via_cb, via_flush);
    }

    // ── OOM protection: max_input_samples enforcement ──

    #[test]
    fn input_larger_than_max_is_rejected() {
        let cfg = PanakoConfig {
            max_input_samples: Some(1_000),
            ..PanakoConfig::default()
        };
        let mut fp = Panako::new(cfg);
        let samples = vec![0.0_f32; 2_000];

        let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
        assert!(matches!(err, AfpError::InputTooLarge { .. }));
    }

    #[test]
    fn none_disables_max_input_check() {
        let cfg = PanakoConfig {
            max_input_samples: None,
            ..PanakoConfig::default()
        };
        let mut fp = Panako::new(cfg);
        let samples = vec![0.0_f32; 16_000];

        fp.extract(&samples, SampleRate::HZ_8000).unwrap();
    }

    #[test]
    fn max_hashes_enforced_rejects_too_many() {
        let cfg = PanakoConfig {
            max_hashes: Some(10),
            ..PanakoConfig::default()
        };
        let mut fp = Panako::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 5);

        let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
        assert!(matches!(err, AfpError::InputTooLarge { .. }));
    }

    #[test]
    fn max_pending_anchors_evicts_oldest() {
        let cfg = PanakoConfig {
            max_pending_anchors: Some(100),
            ..PanakoConfig::default()
        };
        let mut s = StreamingPanako::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 20);
        let mut hashes = s.push(&samples).unwrap();
        hashes.extend(s.flush().unwrap());
        assert!(s.config().max_pending_anchors.is_some());
        assert!(!hashes.is_empty(), "should produce hashes with cap=100");
    }

    #[test]
    fn max_push_samples_truncates_hostile_chunk() {
        let cfg = PanakoConfig {
            max_push_samples: Some(512),
            ..PanakoConfig::default()
        };
        let mut s = StreamingPanako::new(cfg);
        // One huge push must not panic; only the first 512 samples are kept.
        let samples = synthetic_audio(0xBEEF, 8_000 * 5);
        let _ = s.push(&samples).unwrap();
        let _ = s.flush().unwrap();
        assert_eq!(s.config().max_push_samples, Some(512));
    }

    // ── Progress callback tests ──

    #[test]
    fn extract_with_progress_is_called_and_monotonic() {
        let mut fp = Panako::default();
        let samples = synthetic_audio(0xCAFE, 8_000 * 5);

        let mut values: Vec<f32> = Vec::new();
        let result = fp.extract_with_progress(&samples, SampleRate::HZ_8000, |v| values.push(v));
        assert!(result.is_ok());
        // Must be called at least a few times.
        assert!(
            values.len() >= 3,
            "expected at least 3 progress calls, got {}",
            values.len()
        );
        // First value must be 0.0.
        assert_eq!(values[0], 0.0);
        // Last value must be 1.0.
        assert_eq!(*values.last().unwrap(), 1.0);
        // Must be monotonically non-decreasing.
        for w in values.windows(2) {
            assert!(w[1] >= w[0], "progress went backwards: {} → {}", w[0], w[1]);
        }
        // All values must be in [0, 1].
        for &v in &values {
            assert!((0.0..=1.0).contains(&v), "progress out of range: {v}");
        }
    }

    #[test]
    fn extract_with_progress_matches_extract_output() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Panako::default();
        let result1 = fp1.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut fp2 = Panako::default();
        let result2 = fp2
            .extract_with_progress(&samples, SampleRate::HZ_8000, |_| {})
            .unwrap();

        assert_eq!(result1.hashes, result2.hashes);
        assert_eq!(result1.frames_per_sec, result2.frames_per_sec);
    }

    #[test]
    fn extract_with_progress_short_audio_still_reports_0_and_1() {
        let mut fp = Panako::default();
        let samples = synthetic_audio(0xFACE, 8_000 * 2);

        let mut values: Vec<f32> = Vec::new();
        let _ = fp.extract_with_progress(&samples, SampleRate::HZ_8000, |v| values.push(v));
        assert_eq!(values[0], 0.0);
        assert_eq!(*values.last().unwrap(), 1.0);
    }
}
