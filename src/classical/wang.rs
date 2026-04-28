//! Wang-style landmark fingerprinter.
//!
//! The algorithm (Avery Wang, "An Industrial-Strength Audio Search
//! Algorithm", 2003 — the "Shazam paper"):
//!
//! 1. Resample the input to 8 kHz mono *(caller's responsibility)*.
//! 2. Take a Hann-windowed STFT with `n_fft = 1024`, `hop = 128` →
//!    62.5 frames/s, 513 frequency bins.
//! 3. Convert the magnitude spectrogram to dB log-magnitude.
//! 4. Pick spectral peaks in a 31×31 neighbourhood, capped at 30/s.
//! 5. For each anchor peak, take the strongest `fan_out` peaks within
//!    `Δt ∈ [1, target_zone_t]` and `|Δf| ≤ target_zone_f`; pack each
//!    `(anchor, target)` pair into a 32-bit hash.
//!
//! Hash layout (high to low bit):
//! ```text
//! [31..23]  f_a_q  (9 bits, anchor frequency, quantised to 512 buckets)
//! [22..14]  f_b_q  (9 bits, target frequency, same quantisation)
//! [13.. 0]  Δt     (14 bits, frames between anchor and target, clamped 1..=16383)
//! ```

use alloc::vec::Vec;

use libm::log10f;

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, AudioBuffer, Fingerprinter, Result, StreamingFingerprinter, TimestampMs};

/// One anchor-target landmark pair packed into a 32-bit hash.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WangHash {
    /// 32-bit hash: `f_a_q (9) | f_b_q (9) | Δt (14)`, MSB first.
    pub hash: u32,
    /// STFT frame index of the anchor peak.
    pub t_anchor: u32,
}

/// All hashes produced by [`Wang`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct WangFingerprint {
    /// Hashes sorted by `(t_anchor, hash)`.
    pub hashes: Vec<WangHash>,
    /// Frame rate of the underlying STFT — always 62.5 for `wang-v1`
    /// (`8000 / 128`).
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Wang`].
#[derive(Clone, Debug)]
pub struct WangConfig {
    /// `F`: target peaks paired with each anchor. Default 10; embedded
    /// builds typically lower this to 5.
    pub fan_out: u16,
    /// Maximum `Δt` (frames) between anchor and target. Default 63.
    pub target_zone_t: u16,
    /// Maximum `|Δf|` (FFT bins) between anchor and target. Default 64.
    pub target_zone_f: u16,
    /// Per-second cap on peak count. Default 30.
    pub peaks_per_sec: u16,
    /// Magnitude floor (dB) below which peaks are ignored. Default −50.
    pub min_anchor_mag_db: f32,
}

impl Default for WangConfig {
    fn default() -> Self {
        Self {
            fan_out: 10,
            target_zone_t: 63,
            target_zone_f: 64,
            peaks_per_sec: 30,
            min_anchor_mag_db: -50.0,
        }
    }
}

const WANG_N_FFT: usize = 1024;
const WANG_HOP: usize = 128;
const WANG_SR: u32 = 8_000;
const WANG_FRAMES_PER_SEC: f32 = WANG_SR as f32 / WANG_HOP as f32;
/// Quantisation buckets for the 9-bit frequency field.
const WANG_FREQ_BUCKETS: u32 = 512;
const WANG_PEAK_NEIGHBOURHOOD: usize = 15;
const WANG_LOG_FLOOR: f32 = 1e-6;
/// Squared form of the magnitude floor — fed to `log10(power)` instead of
/// `log10(magnitude)`, which lets us skip the per-bin `sqrt` in STFT.
/// Equivalent to `WANG_LOG_FLOOR.powi(2)`.
const WANG_LOG_FLOOR_POWER: f32 = WANG_LOG_FLOOR * WANG_LOG_FLOOR;

/// Wang offline fingerprinter.
///
/// # Example
///
/// ```
/// use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
/// use audiofp::classical::Wang;
///
/// let mut fp = Wang::default();
/// // 3 seconds of silence — produces an empty fingerprint, not an error.
/// let samples = vec![0.0_f32; 8_000 * 3];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
/// let fpr = fp.extract(buf).unwrap();
/// assert_eq!(fpr.frames_per_sec, 62.5);
/// assert!(fpr.hashes.is_empty());
/// ```
pub struct Wang {
    cfg: WangConfig,
    stft: ShortTimeFFT,
    /// Cached peak picker — pools its scratch buffers across calls so
    /// repeated `extract` invocations don't re-allocate.
    picker: PeakPicker,
    /// Pooled log-magnitude buffer reused between calls.
    log_spec: Vec<f32>,
}

impl Default for Wang {
    fn default() -> Self {
        Self::new(WangConfig::default())
    }
}

impl Wang {
    /// Build a Wang extractor with the given config.
    #[must_use]
    pub fn new(cfg: WangConfig) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: WANG_N_FFT,
            hop: WANG_HOP,
            window: WindowKind::Hann,
            // No reflect-padding: hashes are most stable when the first
            // frame starts at sample 0 of the input buffer.
            center: false,
        });
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: WANG_PEAK_NEIGHBOURHOOD,
            neighborhood_f: WANG_PEAK_NEIGHBOURHOOD,
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

impl Fingerprinter for Wang {
    type Output = WangFingerprint;
    type Config = WangConfig;

    fn name(&self) -> &'static str {
        "wang-v1"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> u32 {
        WANG_SR
    }

    fn min_samples(&self) -> usize {
        WANG_SR as usize * 2
    }

    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
        if audio.rate.hz() != WANG_SR {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        if audio.samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: audio.samples.len(),
            });
        }

        // Compute power (|X|²) directly from the FFT — skips a per-bin
        // sqrt that the dB conversion would immediately undo.
        // 20 · log10(sqrt(p)) ≡ 10 · log10(p).
        let (power_flat, n_frames, n_bins) = self.stft.power_flat(audio.samples);
        if n_frames == 0 {
            return Ok(WangFingerprint {
                hashes: Vec::new(),
                frames_per_sec: WANG_FRAMES_PER_SEC,
            });
        }

        // Convert power → dB log-magnitude in-place into the pooled buffer.
        self.log_spec.clear();
        self.log_spec.resize(power_flat.len(), 0.0);
        for (i, &p) in power_flat.iter().enumerate() {
            self.log_spec[i] = 10.0 * log10f(p.max(WANG_LOG_FLOOR_POWER));
        }

        let peaks = self
            .picker
            .pick(&self.log_spec, n_frames, n_bins, WANG_FRAMES_PER_SEC);

        let mut hashes = build_hashes(&peaks, &self.cfg);
        // Stable, deterministic ordering for round-trip and golden tests.
        hashes.sort_unstable_by_key(|h| (h.t_anchor, h.hash));

        Ok(WangFingerprint {
            hashes,
            frames_per_sec: WANG_FRAMES_PER_SEC,
        })
    }
}

/// Wrapper that orders `Peak`s such that the **smallest** magnitude (with
/// the largest position as tiebreak) compares **greatest**. Used as the
/// element of a max-heap to maintain the top-K largest candidates with
/// `O(N log K)` work instead of an `O(N log N)` full sort.
#[derive(Copy, Clone)]
struct MinByMag<'a>(&'a Peak);

impl PartialEq for MinByMag<'_> {
    fn eq(&self, o: &Self) -> bool {
        self.0.mag == o.0.mag && self.0.t_frame == o.0.t_frame && self.0.f_bin == o.0.f_bin
    }
}
impl Eq for MinByMag<'_> {}
impl PartialOrd for MinByMag<'_> {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinByMag<'_> {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        // Reverse mag ordering (smallest first). Reverse position ordering
        // (largest position first) so the final sort's deterministic
        // (mag desc, pos asc) ordering still wins for kept elements.
        o.0.mag
            .partial_cmp(&self.0.mag)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| (o.0.t_frame, o.0.f_bin).cmp(&(self.0.t_frame, self.0.f_bin)))
    }
}

/// Walk `peaks` (sorted by `(t_frame, f_bin)`) and emit landmark hashes.
fn build_hashes(peaks: &[Peak], cfg: &WangConfig) -> Vec<WangHash> {
    let mut hashes = Vec::with_capacity(peaks.len() * cfg.fan_out as usize);
    let target_zone_t = cfg.target_zone_t as i32;
    let target_zone_f = cfg.target_zone_f as i32;
    let fan_out = cfg.fan_out as usize;

    let mut heap: alloc::collections::BinaryHeap<MinByMag> =
        alloc::collections::BinaryHeap::with_capacity(fan_out + 1);
    let mut targets: Vec<&Peak> = Vec::with_capacity(fan_out);

    for (i, anchor) in peaks.iter().enumerate() {
        heap.clear();
        for target in &peaks[i + 1..] {
            let dt = target.t_frame as i32 - anchor.t_frame as i32;
            if dt < 1 {
                continue;
            }
            if dt > target_zone_t {
                // Peaks are sorted by t_frame, so once we exceed the zone
                // for this anchor, no later peak can fit either.
                break;
            }
            let df = target.f_bin as i32 - anchor.f_bin as i32;
            if df.abs() > target_zone_f {
                continue;
            }
            heap.push(MinByMag(target));
            if heap.len() > fan_out {
                // Drop the current smallest — the heap top, by our reversed Ord.
                heap.pop();
            }
        }

        // Drain the heap and re-sort the kept K for deterministic emission.
        targets.clear();
        targets.extend(heap.drain().map(|w| w.0));
        targets.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
        });

        for target in &targets {
            let f_a_q = quantise_freq(anchor.f_bin);
            let f_b_q = quantise_freq(target.f_bin);
            let dt = ((target.t_frame - anchor.t_frame) & 0x3FFF).max(1);
            let hash = ((f_a_q & 0x1FF) << 23) | ((f_b_q & 0x1FF) << 14) | (dt & 0x3FFF);
            hashes.push(WangHash {
                hash,
                t_anchor: anchor.t_frame,
            });
        }
    }
    hashes
}

/// FFT gives 513 bins; pack into 9 bits (512 buckets) per spec.
#[inline]
fn quantise_freq(bin: u16) -> u32 {
    (bin as u32 * WANG_FREQ_BUCKETS) / 513
}

/// Owned wrapper around `Peak` whose `Ord` reverses magnitude (and
/// position tiebreak), so a `BinaryHeap<MinByMagOwned>` of size `K`
/// behaves as a min-heap that retains the top-K largest peaks.
#[derive(Copy, Clone)]
struct MinByMagOwned(Peak);

impl PartialEq for MinByMagOwned {
    fn eq(&self, o: &Self) -> bool {
        self.0.mag == o.0.mag && self.0.t_frame == o.0.t_frame && self.0.f_bin == o.0.f_bin
    }
}
impl Eq for MinByMagOwned {}
impl PartialOrd for MinByMagOwned {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinByMagOwned {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        o.0.mag
            .partial_cmp(&self.0.mag)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| (o.0.t_frame, o.0.f_bin).cmp(&(self.0.t_frame, self.0.f_bin)))
    }
}

/// Anchor pending finalisation, with its top-K target heap.
struct PendingAnchor {
    peak: Peak,
    targets: alloc::collections::BinaryHeap<MinByMagOwned>,
}

/// Streaming Wang fingerprinter — fully incremental.
///
/// Maintains a rolling spectrogram window (`2·neighborhood_t + 1` rows),
/// detects peaks frame-by-frame as they ripen, accumulates per-second
/// candidate buckets, finalises buckets via the per-second adaptive
/// threshold, and grows per-anchor target heaps until each anchor's
/// target zone is fully observed. Per-push CPU cost is proportional to
/// the number of new frames (not the total stream length).
///
/// The output hash multiset matches what [`Wang::extract`] would produce
/// for the same total input — verified by the `streaming_offline_*`
/// tests, including the 1-sample-per-push pathological case.
///
/// # Example
///
/// ```
/// use audiofp::{SampleRate, StreamingFingerprinter};
/// use audiofp::classical::StreamingWang;
///
/// let mut s = StreamingWang::default();
/// // Feed 4 seconds of silence in two chunks; nothing should be emitted.
/// let zeros = vec![0.0_f32; 8_000 * 2];
/// assert!(s.push(&zeros).is_empty());
/// assert!(s.push(&zeros).is_empty());
/// assert!(s.flush().is_empty());
/// ```
pub struct StreamingWang {
    cfg: WangConfig,

    // Front-end.
    stft: ShortTimeFFT,
    sample_carry: alloc::vec::Vec<f32>,

    // Rolling log-power spectrogram window (contiguous, row-major).
    // Capacity = `2*nbht + 1` rows.
    spec: alloc::vec::Vec<f32>,
    spec_n_rows: usize,
    spec_n_bins: usize,
    spec_first_frame: u32,

    // Frame counter and detection cursor.
    n_frames_total: u32,
    last_pd_frame: i32,

    // Pooled peak-detection scratch.
    pd_max: alloc::vec::Vec<f32>,
    pd_temp: alloc::vec::Vec<f32>,
    pd_col_in: alloc::vec::Vec<f32>,
    pd_col_out: alloc::vec::Vec<f32>,

    // Reusable scratch row for STFT output.
    frame_scratch: alloc::vec::Vec<f32>,

    // Per-second adaptive thresholding.
    bucket_pending: alloc::collections::BTreeMap<u32, alloc::vec::Vec<Peak>>,
    last_finalized_bucket: i32,

    // Anchors awaiting finalisation, in t-order.
    pending_anchors: alloc::collections::VecDeque<PendingAnchor>,
}

impl Default for StreamingWang {
    fn default() -> Self {
        Self::new(WangConfig::default())
    }
}

impl StreamingWang {
    /// Build a streaming Wang extractor with the given config.
    #[must_use]
    pub fn new(cfg: WangConfig) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: WANG_N_FFT,
            hop: WANG_HOP,
            window: WindowKind::Hann,
            center: false,
        });
        let n_bins = stft.n_bins();
        let window_capacity = 2 * WANG_PEAK_NEIGHBOURHOOD + 1;
        Self {
            cfg,
            stft,
            sample_carry: alloc::vec::Vec::new(),
            spec: alloc::vec![0.0_f32; window_capacity * n_bins],
            spec_n_rows: 0,
            spec_n_bins: n_bins,
            spec_first_frame: 0,
            n_frames_total: 0,
            last_pd_frame: -1,
            pd_max: alloc::vec::Vec::new(),
            pd_temp: alloc::vec::Vec::new(),
            pd_col_in: alloc::vec::Vec::new(),
            pd_col_out: alloc::vec::Vec::new(),
            frame_scratch: alloc::vec![0.0_f32; n_bins],
            bucket_pending: alloc::collections::BTreeMap::new(),
            last_finalized_bucket: -1,
            pending_anchors: alloc::collections::VecDeque::new(),
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &WangConfig {
        &self.cfg
    }

    /// Frames an anchor must have *after* it before all of its targets
    /// are observed. Used only for [`latency_ms`] — emission timing in
    /// the incremental implementation is driven by anchor finalisation.
    ///
    /// [`latency_ms`]: StreamingWang::latency_ms
    fn lookahead_frames(&self) -> u32 {
        self.cfg.target_zone_t as u32
            + WANG_PEAK_NEIGHBOURHOOD as u32
            + WANG_FRAMES_PER_SEC.ceil() as u32
    }

    /// Append the current contents of `self.frame_scratch` to the
    /// rolling spec buffer, dropping the oldest row if at capacity.
    /// Avoids the per-frame `Vec::clone` the borrow checker would
    /// otherwise force on a `(&mut self, &[f32])` signature.
    fn append_frame_scratch_row(&mut self) {
        debug_assert_eq!(self.frame_scratch.len(), self.spec_n_bins);
        let cap = 2 * WANG_PEAK_NEIGHBOURHOOD + 1;
        if self.spec_n_rows == cap {
            self.spec.copy_within(self.spec_n_bins.., 0);
            self.spec_first_frame += 1;
            self.spec_n_rows -= 1;
        }
        let dst_start = self.spec_n_rows * self.spec_n_bins;
        let n_bins = self.spec_n_bins;
        // Disjoint borrow: `self.spec` (mut) and `self.frame_scratch`
        // (shared) are different fields of `self`, so this is sound.
        self.spec[dst_start..dst_start + n_bins].copy_from_slice(&self.frame_scratch);
        self.spec_n_rows += 1;
    }

    /// Run rolling-max on the current spec buffer and extract peaks at
    /// rows `[from_row_inclusive, to_row_inclusive]` (in spec-buffer-relative
    /// indices). Push survivors into [`bucket_pending`].
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
            WANG_PEAK_NEIGHBOURHOOD,
            WANG_PEAK_NEIGHBOURHOOD,
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
            let bucket = (abs_f as f32 / WANG_FRAMES_PER_SEC) as u32;
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

    /// Finalise one bucket: apply per-second adaptive threshold (top
    /// `peaks_per_sec` by magnitude), then for each surviving peak in
    /// `(t, f)` order, grow target heaps of older anchors and register
    /// the peak as a new anchor.
    fn finalize_bucket(&mut self, bucket: u32) {
        let mut peaks = match self.bucket_pending.remove(&bucket) {
            Some(p) => p,
            None => return,
        };
        // Match the offline picker's `adaptive_per_second`: sort by mag
        // desc only, no positional tiebreak.
        peaks.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        peaks.truncate(self.cfg.peaks_per_sec as usize);
        // Re-sort by `(t, f)` so downstream iteration matches the offline
        // hash builder's order.
        peaks.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));

        let target_zone_t = self.cfg.target_zone_t as i32;
        let target_zone_f = self.cfg.target_zone_f as i32;
        let fan_out = self.cfg.fan_out as usize;

        for peak in peaks {
            // Add this peak as a TARGET to every still-pending anchor whose
            // zone covers it.
            for anchor in self.pending_anchors.iter_mut() {
                let dt = peak.t_frame as i32 - anchor.peak.t_frame as i32;
                if dt < 1 || dt > target_zone_t {
                    continue;
                }
                let df = peak.f_bin as i32 - anchor.peak.f_bin as i32;
                if df.abs() > target_zone_f {
                    continue;
                }
                anchor.targets.push(MinByMagOwned(peak));
                if anchor.targets.len() > fan_out {
                    anchor.targets.pop();
                }
            }
            // Register this peak as a new ANCHOR.
            self.pending_anchors.push_back(PendingAnchor {
                peak,
                targets: alloc::collections::BinaryHeap::with_capacity(fan_out + 1),
            });
        }
        self.last_finalized_bucket = bucket as i32;
    }

    /// Finalise every bucket whose ALL frames have been peak-detected.
    /// Conservative: bucket B is finalisable iff `bucket(last_pd_frame) > B`.
    fn finalize_buckets(&mut self) {
        if self.last_pd_frame < 0 {
            return;
        }
        let current_bucket = (self.last_pd_frame as f32 / WANG_FRAMES_PER_SEC) as i32;
        let to_finalize: alloc::vec::Vec<u32> = self
            .bucket_pending
            .keys()
            .filter(|&&b| (b as i32) > self.last_finalized_bucket && (b as i32) < current_bucket)
            .cloned()
            .collect();
        for bucket in to_finalize {
            self.finalize_bucket(bucket);
        }
    }

    /// Pop anchors whose target zone is fully observed (i.e. the bucket
    /// containing the last possible target frame has been finalised),
    /// build hashes from their accumulated target heap, and return them.
    fn emit_finalized_anchors(&mut self) -> alloc::vec::Vec<(TimestampMs, WangHash)> {
        let mut emitted = alloc::vec::Vec::new();
        while let Some(front) = self.pending_anchors.front() {
            let last_target_frame = front.peak.t_frame + self.cfg.target_zone_t as u32;
            let last_target_bucket = (last_target_frame as f32 / WANG_FRAMES_PER_SEC) as i32;
            if self.last_finalized_bucket < last_target_bucket {
                break;
            }
            let anchor = self.pending_anchors.pop_front().unwrap();
            self.build_hashes_for_anchor(anchor, &mut emitted);
        }
        emitted
    }

    /// Drain an anchor's target heap, sort by `(mag desc, position asc)`
    /// for deterministic emission, then emit the corresponding hashes.
    fn build_hashes_for_anchor(
        &self,
        anchor: PendingAnchor,
        out: &mut alloc::vec::Vec<(TimestampMs, WangHash)>,
    ) {
        let mut targets: alloc::vec::Vec<Peak> = anchor.targets.into_iter().map(|w| w.0).collect();
        targets.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
        });
        for target in &targets {
            let f_a_q = quantise_freq(anchor.peak.f_bin);
            let f_b_q = quantise_freq(target.f_bin);
            let dt = ((target.t_frame - anchor.peak.t_frame) & 0x3FFF).max(1);
            let hash = ((f_a_q & 0x1FF) << 23) | ((f_b_q & 0x1FF) << 14) | (dt & 0x3FFF);
            let t_ms = (anchor.peak.t_frame as u64 * WANG_HOP as u64 * 1000) / WANG_SR as u64;
            out.push((
                TimestampMs(t_ms),
                WangHash {
                    hash,
                    t_anchor: anchor.peak.t_frame,
                },
            ));
        }
    }
}

impl StreamingFingerprinter for StreamingWang {
    type Frame = WangHash;

    fn push(&mut self, samples: &[f32]) -> alloc::vec::Vec<(TimestampMs, Self::Frame)> {
        self.sample_carry.extend_from_slice(samples);

        let nbht = WANG_PEAK_NEIGHBOURHOOD as u32;

        // 1. Compute new STFT frames one at a time, detecting peaks at
        // each frame as soon as it becomes ripe (i.e. its full forward
        // neighbourhood is in the buffer).
        //
        // Walk frames with an offset cursor so we drain `sample_carry`
        // exactly once at the end of the call instead of shifting the
        // tail by `WANG_HOP` after every frame; the loop becomes
        // O(frames) instead of O(frames × buffer).
        let mut off = 0usize;
        while self.sample_carry.len() - off >= WANG_N_FFT {
            self.stft.process_frame_power(
                &self.sample_carry[off..off + WANG_N_FFT],
                &mut self.frame_scratch,
            );
            for v in self.frame_scratch.iter_mut() {
                *v = 10.0 * libm::log10f(v.max(WANG_LOG_FLOOR_POWER));
            }
            // Append `self.frame_scratch` directly via disjoint field
            // borrow, avoiding a per-frame `Vec::clone` of the row.
            self.append_frame_scratch_row();

            let frame_idx = self.n_frames_total;
            self.n_frames_total += 1;
            off += WANG_HOP;

            // After adding frame `frame_idx`, frame `frame_idx - nbht`
            // becomes ripe (its forward neighbourhood is now in the
            // buffer; backward neighbourhood is offline-equivalent
            // because the buffer's left edge matches the offline
            // saturating clip when applicable).
            if frame_idx >= nbht {
                let abs_ripe = frame_idx - nbht;
                let row_idx = (abs_ripe - self.spec_first_frame) as usize;
                self.detect_rows(row_idx, row_idx);
                self.last_pd_frame = abs_ripe as i32;
            }
        }

        if off > 0 {
            self.sample_carry.drain(0..off);
        }

        // 2. Finalise any buckets whose frames are all detected.
        self.finalize_buckets();

        // 3. Emit hashes for anchors whose target zone is fully observed.
        self.emit_finalized_anchors()
    }

    fn flush(&mut self) -> alloc::vec::Vec<(TimestampMs, Self::Frame)> {
        // Detect peaks at remaining frames (those whose forward context
        // would otherwise extend past end-of-stream — same boundary the
        // offline picker handles via `saturating_sub`).
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

        // Finalise every remaining bucket — no more peaks can arrive.
        let buckets: alloc::vec::Vec<u32> = self.bucket_pending.keys().cloned().collect();
        for bucket in buckets {
            self.finalize_bucket(bucket);
        }

        // Emit every remaining anchor — no more targets can arrive.
        let mut emitted = alloc::vec::Vec::new();
        while let Some(anchor) = self.pending_anchors.pop_front() {
            self.build_hashes_for_anchor(anchor, &mut emitted);
        }
        emitted
    }

    fn latency_ms(&self) -> u32 {
        (self.lookahead_frames() * WANG_HOP as u32 * 1000) / WANG_SR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleRate;
    use alloc::vec;
    use core::f32::consts::PI;

    fn synthetic_audio(seed: u32, len: usize) -> Vec<f32> {
        // Two-tone with low-amplitude noise: stable across runs (no rng),
        // but rich enough to produce many peaks.
        let mut out = Vec::with_capacity(len);
        let mut x: u32 = seed.max(1);
        for n in 0..len {
            // xorshift32 — deterministic noise.
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

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut fp = Wang::default();
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
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 8_000]; // 1 second, need 2
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
        let mut fp = Wang::default();
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
        let mut fp = Wang::default();
        let samples = synthetic_audio(0xC0FFEE, 8_000 * 5);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let fpr = fp.extract(buf).unwrap();
        assert!(!fpr.hashes.is_empty(), "expected hashes from a 5s tone");
        // Ordering invariant.
        for w in fpr.hashes.windows(2) {
            assert!((w[0].t_anchor, w[0].hash) <= (w[1].t_anchor, w[1].hash));
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Wang::default();
        let buf1 = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let f1 = fp1.extract(buf1).unwrap();

        let mut fp2 = Wang::default();
        let buf2 = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let f2 = fp2.extract(buf2).unwrap();

        assert_eq!(f1.hashes.len(), f2.hashes.len());
        for (a, b) in f1.hashes.iter().zip(f2.hashes.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn different_signals_diverge() {
        let samples_a = synthetic_audio(0x1111, 8_000 * 3);
        let samples_b = synthetic_audio(0x2222, 8_000 * 3);

        let mut fp = Wang::default();
        let fa = fp
            .extract(AudioBuffer {
                samples: &samples_a,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        let fb = fp
            .extract(AudioBuffer {
                samples: &samples_b,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        // Different noise streams must yield non-identical hash sequences.
        assert_ne!(fa.hashes, fb.hashes);
    }

    #[test]
    fn hash_packing_round_trips() {
        // Smoke: feed a known peak set and verify hash-field decode.
        // Build fake peaks: one anchor, one target inside zone.
        let peaks = alloc::vec![
            Peak {
                t_frame: 100,
                f_bin: 50,
                _pad: 0,
                mag: -10.0
            },
            Peak {
                t_frame: 110,
                f_bin: 70,
                _pad: 0,
                mag: -12.0
            },
        ];
        let cfg = WangConfig::default();
        let hashes = build_hashes(&peaks, &cfg);
        assert_eq!(hashes.len(), 1);
        let h = hashes[0].hash;
        // Decode
        let f_a_q = (h >> 23) & 0x1FF;
        let f_b_q = (h >> 14) & 0x1FF;
        let dt = h & 0x3FFF;
        assert_eq!(f_a_q, quantise_freq(50));
        assert_eq!(f_b_q, quantise_freq(70));
        assert_eq!(dt, 10);
        assert_eq!(hashes[0].t_anchor, 100);
    }

    #[test]
    fn streaming_latency_matches_lookahead() {
        let s = StreamingWang::default();
        // (63 target_zone + 15 picker + 63 adaptive bucket) * 128 / 8000 ≈ 2256 ms.
        assert_eq!(s.latency_ms(), 2_256);
    }

    #[test]
    fn streaming_empty_push_is_empty() {
        let mut s = StreamingWang::default();
        assert!(s.push(&[]).is_empty());
        assert!(s.flush().is_empty());
    }

    #[test]
    fn streaming_silence_emits_nothing() {
        let mut s = StreamingWang::default();
        let zeros = vec![0.0_f32; 8_000 * 4];
        assert!(s.push(&zeros).is_empty());
        assert!(s.flush().is_empty());
    }

    /// xorshift32 → split into deterministic pseudo-random chunk sizes.
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

    /// Sanity check that the incremental impl emits the *same* hashes
    /// across a sequence of fixed-size chunks regardless of the chunk
    /// size — no spurious quadratic state, no per-push artefacts.
    #[test]
    fn streaming_chunk_size_invariant() {
        let samples = synthetic_audio(0xFACE, 8_000 * 4);

        let collect = |chunk_size: usize| -> Vec<WangHash> {
            let mut s = StreamingWang::default();
            let mut out = Vec::new();
            for chunk in samples.chunks(chunk_size) {
                out.extend(s.push(chunk).into_iter().map(|(_, h)| h));
            }
            out.extend(s.flush().into_iter().map(|(_, h)| h));
            out.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
            out
        };

        let baseline = collect(8_000); // 1-second chunks
        for chunk_size in [128, 1024, 4321, 16_000] {
            assert_eq!(
                collect(chunk_size),
                baseline,
                "chunk_size = {chunk_size} produced different hashes than 8000",
            );
        }
    }

    #[test]
    fn streaming_offline_equivalence() {
        let samples = synthetic_audio(0xBEEF, 8_000 * 6);

        // Offline reference.
        let mut offline = Wang::default();
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        // Streaming with random chunks.
        let mut streaming = StreamingWang::default();
        let mut online = Vec::new();
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

        // Same multiset of hashes.
        let mut a: Vec<WangHash> = off.hashes;
        let mut b: Vec<WangHash> = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        assert_eq!(a.len(), b.len(), "hash count mismatch");
        assert_eq!(a, b, "hash sequences differ");
    }

    #[test]
    fn smaller_fan_out_yields_fewer_hashes() {
        let samples = synthetic_audio(0xFEED, 8_000 * 4);
        let buf_a = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let buf_b = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };

        let mut wide = Wang::new(WangConfig {
            fan_out: 10,
            ..WangConfig::default()
        });
        let mut narrow = Wang::new(WangConfig {
            fan_out: 3,
            ..WangConfig::default()
        });
        let f_wide = wide.extract(buf_a).unwrap();
        let f_narrow = narrow.extract(buf_b).unwrap();
        assert!(
            f_narrow.hashes.len() < f_wide.hashes.len(),
            "narrow={} wide={}",
            f_narrow.hashes.len(),
            f_wide.hashes.len(),
        );
    }

    #[test]
    fn quantise_freq_covers_full_range() {
        // Bin 0 maps to bucket 0; bin 512 (≈ Nyquist - 1 step) ≈ bucket 511.
        assert_eq!(quantise_freq(0), 0);
        assert!(quantise_freq(512) < WANG_FREQ_BUCKETS);
        // Quantisation is monotonic non-decreasing.
        let mut prev = 0;
        for b in 0..513_u16 {
            let q = quantise_freq(b);
            assert!(q >= prev);
            assert!(q < WANG_FREQ_BUCKETS);
            prev = q;
        }
    }

    #[test]
    fn streaming_with_one_sample_chunks_still_matches_offline() {
        let samples = synthetic_audio(0xABCD, 8_000 * 3);
        let mut offline = Wang::default();
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut s = StreamingWang::default();
        let mut online = Vec::new();
        // Push one sample at a time — pathological case for any incremental
        // streaming impl.
        for &sample in &samples {
            online.extend(s.push(&[sample]).into_iter().map(|(_, h)| h));
        }
        online.extend(s.flush().into_iter().map(|(_, h)| h));

        let mut a = off.hashes;
        let mut b = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        assert_eq!(a, b);
    }

    #[test]
    fn streaming_state_stays_bounded_under_long_input() {
        // Push 30 s of audio in 256-sample chunks (~940 pushes) and
        // track peak-observed sizes for each streaming buffer. Tight
        // ceilings document the actual steady-state and catch future
        // regressions that would inflate any of them.
        let secs = 30usize;
        let samples = synthetic_audio(7, WANG_SR as usize * secs);
        let chunk = 256usize;

        let mut s = StreamingWang::default();
        let max_spec_rows = 2 * WANG_PEAK_NEIGHBOURHOOD + 1;

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

            // Hard structural invariants — must hold every push.
            assert!(s.sample_carry.len() < WANG_N_FFT);
            assert!(s.spec_n_rows <= max_spec_rows);
            start = end;
        }

        // Tight ceilings on the peaks observed across the whole run at
        // default config (peaks_per_sec=30, target_zone_t=63 frames ≈
        // 1 s of bucket coverage, fan_out=5).
        assert_eq!(
            peak_spec_rows, max_spec_rows,
            "spec window should fill once the stream is long enough",
        );
        assert!(peak_carry < WANG_N_FFT, "peak_carry {peak_carry}");
        assert!(
            peak_bucket_pending <= 3,
            "bucket_pending peaked at {peak_bucket_pending} (steady state should be ≤ 2)",
        );
        // 1 s of finalised buckets × peaks_per_sec=30 = ~30 anchors;
        // allow modest headroom for the boundary between adjacent buckets.
        assert!(
            peak_anchors <= 40,
            "pending_anchors peaked at {peak_anchors} (expected ≤ 40)",
        );

        // Flush drains everything.
        let _ = s.flush();
        assert_eq!(s.bucket_pending.len(), 0);
        assert_eq!(s.pending_anchors.len(), 0);
    }

    #[test]
    fn target_zone_filters_far_peaks() {
        let peaks = alloc::vec![
            Peak {
                t_frame: 0,
                f_bin: 100,
                _pad: 0,
                mag: 0.0
            },
            // Same time → skipped (Δt < 1).
            Peak {
                t_frame: 0,
                f_bin: 200,
                _pad: 0,
                mag: 0.0
            },
            // Δt = 70 > target_zone_t (63) → skipped.
            Peak {
                t_frame: 70,
                f_bin: 100,
                _pad: 0,
                mag: 0.0
            },
            // Inside zone.
            Peak {
                t_frame: 5,
                f_bin: 110,
                _pad: 0,
                mag: 0.0
            },
            // |Δf| = 200 > 64 → skipped.
            Peak {
                t_frame: 5,
                f_bin: 300,
                _pad: 0,
                mag: 0.0
            },
        ];
        // Note: peaks vec must be sorted by (t_frame, f_bin) for the
        // "break on dt > zone" optimisation to fire correctly.
        let mut sorted = peaks;
        sorted.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));

        let cfg = WangConfig::default();
        let hashes = build_hashes(&sorted, &cfg);
        // Anchor at (0,100) should pair with (5,110) only; anchor at (0,200)
        // can pair with (5,110) (|Δf|=90 — wait that's > 64), or (5,300)
        // (|Δf|=100 > 64). Neither fits → no hash from anchor (0,200).
        // From (5,110) onwards, no later peaks fit any anchor.
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0].t_anchor, 0);
    }
}
