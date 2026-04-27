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

        let (power_flat, n_frames, n_bins) = self.stft.power_flat(audio.samples);
        if n_frames == 0 {
            return Ok(PanakoFingerprint {
                hashes: Vec::new(),
                frames_per_sec: PANAKO_FRAMES_PER_SEC,
            });
        }

        // power → dB log-magnitude (20·log10(sqrt(p)) ≡ 10·log10(p)).
        self.log_spec.clear();
        self.log_spec.resize(power_flat.len(), 0.0);
        for (i, &p) in power_flat.iter().enumerate() {
            self.log_spec[i] = 10.0 * log10f(p.max(PANAKO_LOG_FLOOR_POWER));
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
/// largest triplets in `O(N log K)` work.
#[derive(Copy, Clone)]
struct MinByScore<'a>(&'a Peak, &'a Peak, f32);

impl PartialEq for MinByScore<'_> {
    fn eq(&self, o: &Self) -> bool {
        self.2 == o.2
            && (self.0.t_frame, self.0.f_bin) == (o.0.t_frame, o.0.f_bin)
            && (self.1.t_frame, self.1.f_bin) == (o.1.t_frame, o.1.f_bin)
    }
}
impl Eq for MinByScore<'_> {}
impl PartialOrd for MinByScore<'_> {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinByScore<'_> {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        o.2.partial_cmp(&self.2)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| (o.0.t_frame, o.0.f_bin).cmp(&(self.0.t_frame, self.0.f_bin)))
            .then_with(|| (o.1.t_frame, o.1.f_bin).cmp(&(self.1.t_frame, self.1.f_bin)))
    }
}

/// Walk `peaks` (sorted by `(t_frame, f_bin)`) and emit triplet hashes.
fn build_triplet_hashes(peaks: &[Peak], cfg: &PanakoConfig) -> Vec<PanakoHash> {
    let target_zone_t = cfg.target_zone_t as i32;
    let target_zone_f = cfg.target_zone_f as i32;
    let fan_out = cfg.fan_out as usize;

    let mut hashes = Vec::with_capacity(peaks.len() * fan_out);

    let mut targets: Vec<&Peak> = Vec::with_capacity(64);
    let mut heap: alloc::collections::BinaryHeap<MinByScore> =
        alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
    let mut triplets: Vec<(&Peak, &Peak, f32)> = Vec::with_capacity(fan_out);

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
                heap.push(MinByScore(b, c, score));
                if heap.len() > fan_out {
                    heap.pop();
                }
            }
        }

        // Drain and re-sort the kept K for deterministic emission.
        triplets.clear();
        triplets.extend(heap.drain().map(|w| (w.0, w.1, w.2)));
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

    pd_max: Vec<f32>,
    pd_temp: Vec<f32>,
    pd_col_in: Vec<f32>,
    pd_col_out: Vec<f32>,
    frame_scratch: Vec<f32>,

    bucket_pending: alloc::collections::BTreeMap<u32, Vec<Peak>>,
    last_finalized_bucket: i32,

    pending_anchors: alloc::collections::VecDeque<PendingAnchorPanako>,
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
            pd_max: Vec::new(),
            pd_temp: Vec::new(),
            pd_col_in: Vec::new(),
            pd_col_out: Vec::new(),
            frame_scratch: alloc::vec![0.0_f32; n_bins],
            bucket_pending: alloc::collections::BTreeMap::new(),
            last_finalized_bucket: -1,
            pending_anchors: alloc::collections::VecDeque::new(),
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

    fn append_row(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.spec_n_bins);
        let cap = 2 * PANAKO_PEAK_NEIGHBOURHOOD + 1;
        if self.spec_n_rows == cap {
            self.spec.copy_within(self.spec_n_bins.., 0);
            self.spec_first_frame += 1;
            self.spec_n_rows -= 1;
        }
        let dst_start = self.spec_n_rows * self.spec_n_bins;
        self.spec[dst_start..dst_start + self.spec_n_bins].copy_from_slice(row);
        self.spec_n_rows += 1;
    }

    fn detect_rows(&mut self, from_row: usize, to_row: usize) {
        if self.spec_n_rows == 0 || from_row > to_row {
            return;
        }
        let n_rows = self.spec_n_rows;
        let n_bins = self.spec_n_bins;
        let used = n_rows * n_bins;

        self.pd_max.clear();
        self.pd_max.resize(used, 0.0);
        self.pd_temp.clear();
        self.pd_temp.resize(used, 0.0);
        self.pd_col_in.clear();
        self.pd_col_in.resize(n_rows, 0.0);
        self.pd_col_out.clear();
        self.pd_col_out.resize(n_rows, 0.0);

        crate::dsp::peaks::rolling_max_2d_pooled(
            &self.spec[..used],
            n_rows,
            n_bins,
            PANAKO_PEAK_NEIGHBOURHOOD,
            PANAKO_PEAK_NEIGHBOURHOOD,
            &mut self.pd_max,
            &mut self.pd_temp,
            &mut self.pd_col_in,
            &mut self.pd_col_out,
        );

        for row in from_row..=to_row {
            if row >= n_rows {
                break;
            }
            let abs_f = self.spec_first_frame + row as u32;
            let bucket = (abs_f as f32 / PANAKO_FRAMES_PER_SEC) as u32;
            for bin in 0..n_bins {
                let idx = row * n_bins + bin;
                let v = self.spec[idx];
                if v > self.cfg.min_anchor_mag_db && v >= self.pd_max[idx] {
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
        peaks.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
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
        while let Some(front) = self.pending_anchors.front() {
            let last_target_frame = front.peak.t_frame + last_dt;
            let last_target_bucket = (last_target_frame as f32 / PANAKO_FRAMES_PER_SEC) as i32;
            if self.last_finalized_bucket < last_target_bucket {
                break;
            }
            let anchor = self.pending_anchors.pop_front().unwrap();
            self.build_triplets_for_anchor(anchor, &mut emitted);
        }
        emitted
    }

    fn build_triplets_for_anchor(
        &self,
        anchor: PendingAnchorPanako,
        out: &mut Vec<(TimestampMs, PanakoHash)>,
    ) {
        // Use the same heap-based top-K logic the offline builder uses,
        // but on this anchor's pre-collected target list.
        let fan_out = self.cfg.fan_out as usize;
        let mut heap: alloc::collections::BinaryHeap<MinByScore> =
            alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
        for (j, b) in anchor.targets.iter().enumerate() {
            for c in &anchor.targets[j + 1..] {
                let score = b.mag + c.mag;
                heap.push(MinByScore(b, c, score));
                if heap.len() > fan_out {
                    heap.pop();
                }
            }
        }
        let mut triplets: Vec<(&Peak, &Peak, f32)> =
            heap.into_iter().map(|w| (w.0, w.1, w.2)).collect();
        triplets.sort_unstable_by(|x, y| {
            y.2.partial_cmp(&x.2)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (x.0.t_frame, x.0.f_bin).cmp(&(y.0.t_frame, y.0.f_bin)))
                .then_with(|| (x.1.t_frame, x.1.f_bin).cmp(&(y.1.t_frame, y.1.f_bin)))
        });
        for (b, c, _) in triplets {
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
        let nbht = PANAKO_PEAK_NEIGHBOURHOOD as u32;

        while self.sample_carry.len() >= PANAKO_N_FFT {
            self.stft
                .process_frame_power(&self.sample_carry[0..PANAKO_N_FFT], &mut self.frame_scratch);
            for v in self.frame_scratch.iter_mut() {
                *v = 10.0 * libm::log10f(v.max(PANAKO_LOG_FLOOR_POWER));
            }
            let row_copy: Vec<f32> = self.frame_scratch.clone();
            self.append_row(&row_copy);

            let frame_idx = self.n_frames_total;
            self.n_frames_total += 1;
            self.sample_carry.drain(0..PANAKO_HOP);

            if frame_idx >= nbht {
                let abs_ripe = frame_idx - nbht;
                let row_idx = (abs_ripe - self.spec_first_frame) as usize;
                self.detect_rows(row_idx, row_idx);
                self.last_pd_frame = abs_ripe as i32;
            }
        }

        self.finalize_buckets();
        self.emit_finalized_anchors()
    }

    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)> {
        if self.spec_n_rows > 0 && self.n_frames_total > 0 {
            let detect_to_abs = self.n_frames_total as i32 - 1;
            if detect_to_abs > self.last_pd_frame {
                let from_abs = (self.last_pd_frame + 1).max(self.spec_first_frame as i32) as u32;
                let to_abs = detect_to_abs as u32;
                let from_row = (from_abs - self.spec_first_frame) as usize;
                let to_row = (to_abs - self.spec_first_frame) as usize;
                self.detect_rows(from_row, to_row);
                self.last_pd_frame = detect_to_abs;
            }
        }
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
}
