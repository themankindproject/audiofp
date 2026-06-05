//! Panako-style triplet fingerprinter.
//!
//! Same front-end as [`super::Wang`] (8 kHz, STFT `n_fft=1024 hop=128`
//! Hann, dB log-magnitude peak picking) but each anchor produces hashes
//! over *triplets* `(a, b, c)` rather than pairs. The third peak gives a
//! tempo-invariant ratio `β` that is robust to ±5 % time stretch.
//!
//! Hash layout (Six 2021 §3.2), high to low bit:
//! ```text
//! [31..30]  sign       (2 bits, sign of Δf_ab and Δf_bc)
//! [29..28]  mag_order  (2 bits, which of {a, b, c} has the largest magnitude)
//! [27..23]  β          (5 bits, round((t_c - t_b) / (t_c - t_a) · 31))
//! [22..15]  Δf_ab      (8 bits, signed, clamped to ±127)
//! [14.. 7]  Δf_bc      (8 bits, signed, clamped to ±127)
//! [ 6.. 0]  reserved   (7 bits, zero)
//! ```

use alloc::vec::Vec;

use libm::{log10f, roundf};

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, AudioBuffer, Fingerprinter, Result, StreamingFingerprinter, TimestampMs};

/// One anchor-target-target triplet packed into a 32-bit hash plus the
/// three frame indices.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PanakoHash {
    /// 32-bit hash; see module docs for the layout.
    pub hash: u32,
    /// STFT frame index of the anchor peak.
    pub t_anchor: u32,
    /// STFT frame index of the first (closer) target.
    pub t_b: u32,
    /// STFT frame index of the second (farther) target.
    pub t_c: u32,
}

/// All triplet hashes produced by [`Panako`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct PanakoFingerprint {
    /// Hashes sorted by `(t_anchor, t_b, t_c, hash)`.
    pub hashes: Vec<PanakoHash>,
    /// Frame rate of the underlying STFT — always 62.5 for `panako-v2`.
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Panako`].
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
}

impl Default for PanakoConfig {
    fn default() -> Self {
        Self {
            fan_out: 5,
            target_zone_t: 96,
            target_zone_f: 96,
            peaks_per_sec: 30,
            min_anchor_mag_db: -50.0,
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

/// Panako offline fingerprinter.
///
/// # Example
///
/// ```
/// use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
/// use audiofp::classical::Panako;
///
/// let mut fp = Panako::default();
/// let samples = vec![0.0_f32; 8_000 * 3];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
/// let fpr = fp.extract(buf).unwrap();
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
    #[must_use]
    pub fn new(cfg: PanakoConfig) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: PANAKO_N_FFT,
            hop: PANAKO_HOP,
            window: WindowKind::Hann,
            center: false,
        });
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: PANAKO_PEAK_NEIGHBOURHOOD,
            neighborhood_f: PANAKO_PEAK_NEIGHBOURHOOD,
            min_magnitude: cfg.min_anchor_mag_db,
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

impl Fingerprinter for Panako {
    type Output = PanakoFingerprint;
    type Config = PanakoConfig;

    fn name(&self) -> &'static str {
        "panako-v2"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> u32 {
        PANAKO_SR
    }

    fn min_samples(&self) -> usize {
        PANAKO_SR as usize * 2
    }

    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
        if audio.rate.hz() != PANAKO_SR {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        if audio.samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: audio.samples.len(),
            });
        }

        let (n_frames, n_bins) = self.stft.power_flat_into(audio.samples, &mut self.log_spec);
        if n_frames == 0 {
            return Ok(PanakoFingerprint {
                hashes: Vec::new(),
                frames_per_sec: PANAKO_FRAMES_PER_SEC,
            });
        }

        // power → dB log-magnitude in-place (20·log10(sqrt(p)) ≡ 10·log10(p)).
        for v in self.log_spec.iter_mut() {
            *v = 10.0 * log10f(v.max(PANAKO_LOG_FLOOR_POWER));
        }

        let peaks = self
            .picker
            .pick(&self.log_spec, n_frames, n_bins, PANAKO_FRAMES_PER_SEC);

        let mut hashes = build_triplet_hashes(&peaks, &self.cfg);
        hashes.sort_unstable_by_key(|h| (h.t_anchor, h.t_b, h.t_c, h.hash));

        Ok(PanakoFingerprint {
            hashes,
            frames_per_sec: PANAKO_FRAMES_PER_SEC,
        })
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

    let mut targets: Vec<&Peak> = Vec::with_capacity(64);
    let mut heap: alloc::collections::BinaryHeap<MinByScoreOwned> =
        alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
    let mut triplets: Vec<(Peak, Peak, f32)> = Vec::with_capacity(fan_out);

    for (i, anchor) in peaks.iter().enumerate() {
        // Collect all peaks in the cone.
        targets.clear();
        for target in &peaks[i + 1..] {
            let dt = target.t_frame as i32 - anchor.t_frame as i32;
            if dt < 1 {
                continue;
            }
            if dt >= target_zone_t {
                break;
            }
            let df = target.f_bin as i32 - anchor.f_bin as i32;
            if df.abs() >= target_zone_f {
                continue;
            }
            targets.push(target);
        }

        // Heap-based top-K over (b, c) tuples, scored by `b.mag + c.mag`.
        heap.clear();
        for (j, b) in targets.iter().enumerate() {
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
    let beta = (roundf(dt_bc / dt_ac * 31.0) as i32).clamp(0, 31) as u32;

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
/// Anchor pending finalisation, with all observed targets in cone.
struct PendingAnchorPanako {
    peak: Peak,
    targets: alloc::vec::Vec<Peak>,
}

/// Streaming Panako fingerprinter — fully incremental.
///
/// Same rolling-spectrogram + per-bucket-finalisation strategy as
/// [`super::StreamingWang`]. Per-anchor state collects ALL targets in
/// the cone (rather than top-K) because Panako's hash builder enumerates
/// `(b, c)` pairs over them; the heap-based top-K is applied at the
/// pair level when the anchor is finalised.
///
/// Output is bit-exactly equivalent to [`Panako::extract`].
pub struct StreamingPanako {
    cfg: PanakoConfig,

    stft: ShortTimeFFT,
    sample_carry: Vec<f32>,

    spec: Vec<f32>,
    spec_n_rows: usize,
    spec_n_bins: usize,
    spec_first_frame: u32,

    n_frames_total: u32,
    last_pd_frame: i32,

    peak_det: crate::dsp::peaks::IncrementalPeakDetector,
    peak_row_max: Vec<f32>,
    frame_scratch: Vec<f32>,

    bucket_pending: alloc::collections::BTreeMap<u32, Vec<Peak>>,
    last_finalized_bucket: i32,

    pending_anchors: alloc::collections::VecDeque<PendingAnchorPanako>,

    /// Pooled scratch for the sorted (b, c, score) triplet list.
    /// Stores owned Peak copies to avoid lifetime issues on the struct.
    triplet_scratch: Vec<(Peak, Peak, f32)>,
}

impl Default for StreamingPanako {
    fn default() -> Self {
        Self::new(PanakoConfig::default())
    }
}

impl StreamingPanako {
    /// Build a streaming Panako extractor with the given config.
    #[must_use]
    pub fn new(cfg: PanakoConfig) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: PANAKO_N_FFT,
            hop: PANAKO_HOP,
            window: WindowKind::Hann,
            center: false,
        });
        let n_bins = stft.n_bins();
        let window_capacity = 2 * PANAKO_PEAK_NEIGHBOURHOOD + 1;
        Self {
            cfg,
            stft,
            sample_carry: Vec::new(),
            spec: alloc::vec![0.0_f32; window_capacity * n_bins],
            spec_n_rows: 0,
            spec_n_bins: n_bins,
            spec_first_frame: 0,
            n_frames_total: 0,
            last_pd_frame: -1,
            peak_det: crate::dsp::peaks::IncrementalPeakDetector::new(
                PANAKO_PEAK_NEIGHBOURHOOD,
                PANAKO_PEAK_NEIGHBOURHOOD,
                n_bins,
            ),
            peak_row_max: alloc::vec![0.0_f32; n_bins],
            frame_scratch: alloc::vec![0.0_f32; n_bins],
            bucket_pending: alloc::collections::BTreeMap::new(),
            last_finalized_bucket: -1,
            pending_anchors: alloc::collections::VecDeque::new(),
            triplet_scratch: Vec::new(),
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &PanakoConfig {
        &self.cfg
    }

    fn lookahead_frames(&self) -> u32 {
        self.cfg.target_zone_t as u32
            + PANAKO_PEAK_NEIGHBOURHOOD as u32
            + PANAKO_FRAMES_PER_SEC.ceil() as u32
    }

    /// Append the current contents of `self.frame_scratch` to the
    /// rolling spec buffer, dropping the oldest row if at capacity.
    /// Avoids the per-frame `Vec::clone` the borrow checker would
    /// otherwise force on a `(&mut self, &[f32])` signature.
    fn append_frame_scratch_row(&mut self) {
        debug_assert_eq!(self.frame_scratch.len(), self.spec_n_bins);
        let cap = 2 * PANAKO_PEAK_NEIGHBOURHOOD + 1;
        if self.spec_n_rows == cap {
            self.spec.copy_within(self.spec_n_bins.., 0);
            self.spec_first_frame += 1;
            self.spec_n_rows -= 1;
        }
        let dst_start = self.spec_n_rows * self.spec_n_bins;
        let n_bins = self.spec_n_bins;
        // Disjoint borrow of `self.spec` (mut) and `self.frame_scratch`
        // (shared) — different fields of `self`, so this is sound.
        self.spec[dst_start..dst_start + n_bins].copy_from_slice(&self.frame_scratch);
        self.spec_n_rows += 1;
    }

    fn detect_rows(&mut self, from_row: usize, to_row: usize) {
        if self.spec_n_rows == 0 || from_row > to_row {
            return;
        }
        let n_bins = self.spec_n_bins;

        for row in from_row..=to_row {
            if row >= self.spec_n_rows {
                break;
            }
            let abs_f = self.spec_first_frame + row as u32;
            let bucket = (abs_f as f32 / PANAKO_FRAMES_PER_SEC) as u32;
            for bin in 0..n_bins {
                let idx = row * n_bins + bin;
                let v = self.spec[idx];
                if v > self.cfg.min_anchor_mag_db && v >= self.peak_row_max[bin] {
                    let peak = Peak {
                        t_frame: abs_f,
                        f_bin: bin as u16,
                        _pad: 0,
                        mag: v,
                    };
                    self.bucket_pending.entry(bucket).or_default().push(peak);
                }
            }
        }
    }

    fn finalize_bucket(&mut self, bucket: u32) {
        let mut peaks = match self.bucket_pending.remove(&bucket) {
            Some(p) => p,
            None => return,
        };
        // Sort by mag desc, then `(t, f)` ascending. The positional
        // tiebreak is unique per peak, so equal-magnitude peaks at the
        // truncation boundary resolve identically to the offline
        // `adaptive_per_second`.
        peaks.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
        });
        peaks.truncate(self.cfg.peaks_per_sec as usize);
        peaks.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));

        let target_zone_t = self.cfg.target_zone_t as i32;
        let target_zone_f = self.cfg.target_zone_f as i32;

        for peak in peaks {
            // Add as TARGET to older anchors whose cone covers it
            // (Panako uses STRICT inequalities `dt < target_zone_t`,
            // `|df| < target_zone_f` — match `build_triplet_hashes`).
            for anchor in self.pending_anchors.iter_mut() {
                let dt = peak.t_frame as i32 - anchor.peak.t_frame as i32;
                if dt < 1 || dt >= target_zone_t {
                    continue;
                }
                let df = peak.f_bin as i32 - anchor.peak.f_bin as i32;
                if df.abs() >= target_zone_f {
                    continue;
                }
                anchor.targets.push(peak);
            }
            self.pending_anchors.push_back(PendingAnchorPanako {
                peak,
                targets: Vec::new(),
            });
        }
        self.last_finalized_bucket = bucket as i32;
    }

    fn finalize_buckets(&mut self) {
        if self.last_pd_frame < 0 {
            return;
        }
        let current_bucket = (self.last_pd_frame as f32 / PANAKO_FRAMES_PER_SEC) as i32;
        let to_finalize: Vec<u32> = self
            .bucket_pending
            .keys()
            .filter(|&&b| (b as i32) > self.last_finalized_bucket && (b as i32) < current_bucket)
            .cloned()
            .collect();
        for bucket in to_finalize {
            self.finalize_bucket(bucket);
        }
    }

    fn emit_finalized_anchors(&mut self) -> Vec<(TimestampMs, PanakoHash)> {
        let mut emitted = Vec::new();
        // Anchor's last possible target frame is `t + (target_zone_t - 1)`
        // because Panako uses strict `dt < target_zone_t`.
        let last_dt = self.cfg.target_zone_t as u32 - 1;
        // Pop-and-push pattern: take the front anchor, decide whether its
        // target zone is fully observed, and if not put it back. This avoids
        // an `unwrap` after a separate `front()` peek and stays a clean
        // `while let` over the pop result.
        while let Some(anchor) = self.pending_anchors.pop_front() {
            let last_target_frame = anchor.peak.t_frame + last_dt;
            let last_target_bucket = (last_target_frame as f32 / PANAKO_FRAMES_PER_SEC) as i32;
            if self.last_finalized_bucket < last_target_bucket {
                self.pending_anchors.push_front(anchor);
                break;
            }
            self.build_triplets_for_anchor(anchor, &mut emitted);
        }
        emitted
    }

    fn build_triplets_for_anchor(
        &mut self,
        anchor: PendingAnchorPanako,
        out: &mut Vec<(TimestampMs, PanakoHash)>,
    ) {
        let fan_out = self.cfg.fan_out as usize;
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
        self.triplet_scratch.clear();
        self.triplet_scratch
            .extend(heap.drain().map(|w| (w.b, w.c, w.score)));
        self.triplet_scratch.sort_unstable_by(|x, y| {
            y.2.partial_cmp(&x.2)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (x.0.t_frame, x.0.f_bin).cmp(&(y.0.t_frame, y.0.f_bin)))
                .then_with(|| (x.1.t_frame, x.1.f_bin).cmp(&(y.1.t_frame, y.1.f_bin)))
        });
        for (b, c, _) in &self.triplet_scratch {
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

    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)> {
        self.sample_carry.extend_from_slice(samples);

        let mut off = 0usize;
        while self.sample_carry.len() - off >= PANAKO_N_FFT {
            self.stft.process_frame_power(
                &self.sample_carry[off..off + PANAKO_N_FFT],
                &mut self.frame_scratch,
            );
            for v in self.frame_scratch.iter_mut() {
                *v = 10.0 * libm::log10f(v.max(PANAKO_LOG_FLOOR_POWER));
            }
            self.append_frame_scratch_row();

            self.n_frames_total += 1;
            off += PANAKO_HOP;

            if let Some(ripe_abs) = self
                .peak_det
                .push_row(&self.frame_scratch, &mut self.peak_row_max)
            {
                let row_idx = (ripe_abs - self.spec_first_frame) as usize;
                self.detect_rows(row_idx, row_idx);
                self.last_pd_frame = ripe_abs as i32;
            }
        }

        if off > 0 {
            self.sample_carry.drain(0..off);
        }

        self.finalize_buckets();
        self.emit_finalized_anchors()
    }

    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)> {
        let n_bins = self.spec_n_bins;
        let min_mag = self.cfg.min_anchor_mag_db;
        let spec = &self.spec;
        let spec_first_frame = self.spec_first_frame;
        let bucket_pending = &mut self.bucket_pending;
        let last_pd = &mut self.last_pd_frame;

        self.peak_det
            .flush(&mut self.peak_row_max, |ripe_abs, max_row| {
                let row_idx = (ripe_abs - spec_first_frame) as usize;
                let bucket = (ripe_abs as f32 / PANAKO_FRAMES_PER_SEC) as u32;
                for (bin, &row_max) in max_row.iter().enumerate().take(n_bins) {
                    let idx = row_idx * n_bins + bin;
                    let v = spec[idx];
                    if v > min_mag && v >= row_max {
                        let peak = Peak {
                            t_frame: ripe_abs,
                            f_bin: bin as u16,
                            _pad: 0,
                            mag: v,
                        };
                        bucket_pending.entry(bucket).or_default().push(peak);
                    }
                }
                *last_pd = ripe_abs as i32;
            });

        let buckets: Vec<u32> = self.bucket_pending.keys().cloned().collect();
        for bucket in buckets {
            self.finalize_bucket(bucket);
        }
        let mut emitted = Vec::new();
        while let Some(anchor) = self.pending_anchors.pop_front() {
            self.build_triplets_for_anchor(anchor, &mut emitted);
        }
        emitted
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
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_16000,
        };
        match fp.extract(buf) {
            Err(AfpError::UnsupportedSampleRate(16_000)) => {}
            other => panic!("expected UnsupportedSampleRate(16000), got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_audio() {
        let mut fp = Panako::default();
        let samples = vec![0.0_f32; 8_000];
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        match fp.extract(buf) {
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
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let fpr = fp.extract(buf).unwrap();
        assert_eq!(fpr.frames_per_sec, 62.5);
        assert!(fpr.hashes.is_empty());
    }

    #[test]
    fn synthetic_signal_produces_hashes() {
        let mut fp = Panako::default();
        let samples = synthetic_audio(0xC0FFEE, 8_000 * 5);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let fpr = fp.extract(buf).unwrap();
        assert!(!fpr.hashes.is_empty(), "expected hashes from a 5s tone");
        for w in fpr.hashes.windows(2) {
            assert!((w[0].t_anchor, w[0].t_b, w[0].t_c) <= (w[1].t_anchor, w[1].t_b, w[1].t_c));
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Panako::default();
        let f1 = fp1
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut fp2 = Panako::default();
        let f2 = fp2
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        assert_eq!(f1.hashes, f2.hashes);
    }

    #[test]
    fn different_signals_diverge() {
        let a = synthetic_audio(0x1111, 8_000 * 3);
        let b = synthetic_audio(0x2222, 8_000 * 3);

        let mut fp = Panako::default();
        let fa = fp
            .extract(AudioBuffer {
                samples: &a,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        let fb = fp
            .extract(AudioBuffer {
                samples: &b,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
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
        assert!(s.push(&zeros).is_empty());
        assert!(s.flush().is_empty());
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
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut streaming = StreamingPanako::default();
        let mut online: Vec<PanakoHash> = Vec::new();
        let mut cursor = 0;
        for n in chunk_sizes(0xCAFE, samples.len(), 4_000) {
            let end = cursor + n;
            online.extend(
                streaming
                    .push(&samples[cursor..end])
                    .into_iter()
                    .map(|(_, h)| h),
            );
            cursor = end;
        }
        online.extend(streaming.flush().into_iter().map(|(_, h)| h));

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
            let _ = s.push(&samples[start..end]);
            peak_carry = peak_carry.max(s.sample_carry.len());
            peak_spec_rows = peak_spec_rows.max(s.spec_n_rows);
            peak_bucket_pending = peak_bucket_pending.max(s.bucket_pending.len());
            peak_anchors = peak_anchors.max(s.pending_anchors.len());

            assert!(s.sample_carry.len() < PANAKO_N_FFT);
            assert!(s.spec_n_rows <= max_spec_rows);
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

        let _ = s.flush();
        assert_eq!(s.bucket_pending.len(), 0);
        assert_eq!(s.pending_anchors.len(), 0);
    }

    // -----------------------------------------------------------------
    // Direct unit tests for `emit_finalized_anchors`.
    //
    // Same re-queue invariant as the wang.rs counterpart. See the
    // comment block there for motivation; this is the Panako
    // mirror. Panako's `last_target_frame = t + (target_zone_t - 1)`
    // (strict `dt < target_zone_t`).
    // -----------------------------------------------------------------

    fn panako_anchor_with_target(
        t_frame: u32,
        f_bin: u16,
        target_t: u32,
        target_f: u16,
    ) -> PendingAnchorPanako {
        // Two targets so that `build_triplets_for_anchor` (which
        // iterates over `(b, c)` pairs) produces at least one hash.
        PendingAnchorPanako {
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
        s.pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        // t=5 → last target frame 100 → bucket 1
        s.pending_anchors
            .push_back(panako_anchor_with_target(5, 20, 15, 22));
        // t=100 → last target frame 195 → bucket 3
        s.pending_anchors
            .push_back(panako_anchor_with_target(100, 30, 110, 32));
        s.last_finalized_bucket = panako_bucket_of(195);

        let emitted = s.emit_finalized_anchors();
        assert_eq!(emitted.len(), 3);
        assert!(s.pending_anchors.is_empty());
    }

    #[test]
    fn panako_emit_finalized_anchors_re_queues_unfinalised() {
        // Two anchors; only the first is covered. The second must
        // remain in `pending_anchors` after the emit.
        let mut s = StreamingPanako::default();
        s.pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        s.pending_anchors
            .push_back(panako_anchor_with_target(100, 30, 110, 32));
        // Only cover bucket 1 (last target frame ≤ 95).
        s.last_finalized_bucket = 1;

        let emitted = s.emit_finalized_anchors();
        assert_eq!(emitted.len(), 1);
        assert_eq!(s.pending_anchors.len(), 1);
        assert_eq!(s.pending_anchors.front().unwrap().peak.t_frame, 100);
    }

    #[test]
    fn panako_emit_finalized_anchors_idempotent_under_repeated_calls() {
        // With one anchor covered, two consecutive calls must emit
        // the same hashes (no double-emit, no lost anchor).
        let mut s = StreamingPanako::default();
        s.pending_anchors
            .push_back(panako_anchor_with_target(0, 10, 10, 12));
        s.last_finalized_bucket = panako_bucket_of(95);

        let first = s.emit_finalized_anchors();
        let second = s.emit_finalized_anchors();
        assert_eq!(first.len(), 1);
        assert!(second.is_empty());
        assert!(s.pending_anchors.is_empty());
    }

    // -----------------------------------------------------------------
    // Public API contract pins. See wang.rs for motivation.
    // -----------------------------------------------------------------

    #[test]
    fn public_api_name_and_config_match_documented_values() {
        let fp = Panako::default();
        assert_eq!(fp.name(), "panako-v2");
        assert_eq!(fp.required_sample_rate(), 8_000);
        assert_eq!(fp.min_samples(), 16_000);

        let s = StreamingPanako::default();
        // 2 784 ms at the documented defaults.
        assert_eq!(s.latency_ms(), 2_784);
    }
}
