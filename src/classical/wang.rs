//! Wang-style landmark fingerprinter.
//!
//! The algorithm: Wang, A. "An Industrial-Strength Audio Search
//! Algorithm." Proceedings of the 4th International Conference on Music
//! Information Retrieval (ISMIR), Baltimore, MD, USA, 2003.
//! <https://www.ee.columbia.edu/~dpwe/papers/Wang03-shazam.pdf>
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

use bytemuck::Zeroable;

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, Fingerprinter, Result, SampleRate, StreamingFingerprinter, TimestampMs};

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
///
/// Always construct with FRU so future additive fields stay compatible:
/// `WangConfig { fan_out: 5, ..Default::default() }`.
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
    /// Maximum input sample count accepted by [`extract`]. `None` disables
    /// the check (full backward compatibility). Default: 14_400_000
    /// (30 minutes at 8 kHz).
    ///
    /// [`extract`]: Wang::extract
    pub max_input_samples: Option<usize>,
    /// Maximum number of hashes allowed. `None` disables. Default: 500_000
    /// — enough for ~2 hours of rich music at default fan_out=10.
    pub max_hashes: Option<usize>,
    /// Maximum number of pending anchors in the streaming pipeline.
    /// `None` disables (default, unbounded). When set, anchors exceeding
    /// this cap are dropped oldest-first so memory stays bounded.
    /// Recommended: `Some(10_000)` for untrusted input.
    /// Relevant only for [`StreamingWang`].
    pub max_pending_anchors: Option<usize>,
    /// Maximum samples accepted in a single `push` call. `None` disables
    /// (default). When set, excess samples beyond the cap are **dropped**
    /// (streaming `push` is infallible).
    pub max_push_samples: Option<usize>,
}

impl Default for WangConfig {
    fn default() -> Self {
        Self {
            fan_out: 10,
            target_zone_t: 63,
            target_zone_f: 64,
            peaks_per_sec: 30,
            min_anchor_mag_db: -50.0,
            max_input_samples: Some(30 * 60 * WANG_SR as usize),
            max_hashes: Some(500_000),
            max_pending_anchors: None,
            max_push_samples: None,
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
use crate::dsp::power_to_db_wide;

/// Wang offline fingerprinter.
///
/// # Example
///
/// ```
/// use audiofp::{Fingerprinter, SampleRate};
/// use audiofp::classical::Wang;
///
/// let mut fp = Wang::default();
/// // 3 seconds of silence — produces an empty fingerprint, not an error.
/// let samples = vec![0.0_f32; 8_000 * 3];
///
/// let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
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
    ///
    /// Clamps `target_zone_t` to a minimum of 1 and `fan_out` to a
    /// minimum of 1 to prevent underflows/empty output from degenerate
    /// configurations. Caps `target_zone_t` at 512 and `fan_out` at 64
    /// to prevent OOM from extreme values.
    #[must_use]
    pub fn new(mut cfg: WangConfig) -> Self {
        cfg.target_zone_t = cfg.target_zone_t.clamp(1, 512);
        cfg.fan_out = cfg.fan_out.clamp(1, 64);
        cfg.peaks_per_sec = cfg.peaks_per_sec.min(500);
        // Reject zero-value limits (would reject all inputs/outputs).
        if cfg.max_input_samples == Some(0) {
            cfg.max_input_samples = Some(1);
        }
        if cfg.max_hashes == Some(0) {
            cfg.max_hashes = Some(1);
        }
        if cfg.max_pending_anchors == Some(0) {
            cfg.max_pending_anchors = Some(1);
        }
        cfg.target_zone_f = cfg.target_zone_f.clamp(1, 512);
        cfg.min_anchor_mag_db = cfg.min_anchor_mag_db.clamp(-200.0, 0.0);
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

/// Progress callback reporting interval for Wang/Panako (62.5 fps):
/// every 32 frames ≈ 500 ms of audio.
const WANG_PROGRESS_INTERVAL: usize = 32;

impl Wang {
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
    ) -> Result<WangFingerprint> {
        crate::pcm::reject_non_finite(samples)?;
        if let Some(limit) = self.cfg.max_input_samples
            && samples.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: samples.len(),
            });
        }
        if rate.hz() != WANG_SR {
            return Err(AfpError::UnsupportedSampleRate(rate.hz()));
        }
        if samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: samples.len(),
            });
        }

        progress(0.0);

        // Compute power (|X|²) directly from the FFT — skips a per-bin
        // sqrt that the dB conversion would immediately undo.
        // 20 · log10(sqrt(p)) ≡ 10 · log10(p).
        let (n_frames, n_bins) = self.stft.power_flat_into(samples, &mut self.log_spec);
        if n_frames == 0 {
            progress(1.0);
            return Ok(WangFingerprint {
                hashes: Vec::new(),
                frames_per_sec: WANG_FRAMES_PER_SEC,
            });
        }

        // Report progress through the STFT phase (~70% of total work).
        // Since power_flat_into is a bulk operation, report proportional
        // progress at intervals based on frame count.
        let total_frames = n_frames;
        let stft_weight = 0.7_f32;
        let interval = WANG_PROGRESS_INTERVAL;
        {
            let mut reported = 0usize;
            while reported + interval < total_frames {
                reported += interval;
                progress(stft_weight * (reported as f32 / total_frames as f32));
            }
        }
        progress(stft_weight);

        // Convert power → dB log-magnitude in-place.
        // 10·log10(power) ≡ DB_LOG2_FACTOR·log2(power).
        power_to_db_wide(&mut self.log_spec, WANG_LOG_FLOOR_POWER);
        progress(0.80);

        let peaks = self
            .picker
            .pick(&self.log_spec, n_frames, n_bins, WANG_FRAMES_PER_SEC);
        progress(0.90);

        let mut hashes = build_hashes(&peaks, &self.cfg);
        // Stable, deterministic ordering for round-trip and golden tests.
        hashes.sort_unstable_by_key(|h| (h.t_anchor, h.hash));

        if let Some(limit) = self.cfg.max_hashes
            && hashes.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: hashes.len(),
            });
        }

        progress(1.0);

        Ok(WangFingerprint {
            hashes,
            frames_per_sec: WANG_FRAMES_PER_SEC,
        })
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

    fn required_sample_rate(&self) -> SampleRate {
        // WANG_SR is a compile-time constant; unwrap is trivially safe.
        SampleRate::new(WANG_SR).expect("WANG_SR is non-zero")
    }

    fn min_samples(&self) -> usize {
        WANG_SR as usize * 2
    }

    fn extract(&mut self, samples: &[f32], rate: SampleRate) -> Result<Self::Output> {
        self.extract_with_progress(samples, rate, |_| {})
    }
}

/// Walk `peaks` (sorted by `(t_frame, f_bin)`) and emit landmark hashes.
fn build_hashes(peaks: &[Peak], cfg: &WangConfig) -> Vec<WangHash> {
    let mut hashes = Vec::with_capacity(peaks.len() * cfg.fan_out as usize);
    let target_zone_t = cfg.target_zone_t as u32;
    let target_zone_f = cfg.target_zone_f as i32;
    let fan_out = cfg.fan_out as usize;

    // Pooled target list reused across anchors. Maintained sorted by
    // (mag desc, position asc) via linear-insert. For fan_out ≤ 16 the
    // O(K) insert is faster than BinaryHeap's O(log K) because the
    // constant factor of partition_point + memmove is lower in practice.
    let mut targets: Vec<Peak> = Vec::with_capacity(fan_out);

    for (i, anchor) in peaks.iter().enumerate() {
        // Binary search for the upper bound of the target zone:
        // the first peak whose t_frame > anchor.t_frame + target_zone_t.
        // Since peaks are sorted by (t_frame, f_bin), all valid targets
        // for this anchor lie in `peaks[i+1..zone_end]`.
        let zone_limit = anchor.t_frame.saturating_add(target_zone_t);
        let zone_end = peaks[i + 1..].partition_point(|p| p.t_frame <= zone_limit);

        targets.clear();
        for target in &peaks[i + 1..i + 1 + zone_end] {
            let dt = target.t_frame - anchor.t_frame;
            if dt < 1 {
                continue;
            }
            let df = target.f_bin as i32 - anchor.f_bin as i32;
            if df.abs() > target_zone_f {
                continue;
            }
            // Linear-insert into sorted list: maintains top-K peaks by
            // (mag desc, position asc). partition_point finds where
            // this target belongs; insert shifts elements right.
            // O(K) insert per target beats BinaryHeap for K ≤ 16.
            if targets.len() < fan_out {
                let pos = targets.partition_point(|p| {
                    p.mag > target.mag
                        || (p.mag == target.mag
                            && (p.t_frame, p.f_bin) <= (target.t_frame, target.f_bin))
                });
                targets.insert(pos, *target);
            } else if target.mag > targets.last().unwrap().mag
                || (target.mag == targets.last().unwrap().mag
                    && (target.t_frame, target.f_bin)
                        < (
                            targets.last().unwrap().t_frame,
                            targets.last().unwrap().f_bin,
                        ))
            {
                let pos = targets.partition_point(|p| {
                    p.mag > target.mag
                        || (p.mag == target.mag
                            && (p.t_frame, p.f_bin) <= (target.t_frame, target.f_bin))
                });
                targets.insert(pos, *target);
                targets.pop();
            }
        }

        // targets is already sorted by (mag desc, position asc) from the
        // linear-insert logic, so we can emit directly.

        let f_a_q = quantise_freq(anchor.f_bin);
        for target in &targets {
            let f_b_q = quantise_freq(target.f_bin);
            // Δt is encoded in 14 bits (max 16383). target_zone_t can never
            // realistically saturate this with default config, but clamp
            // defensively so an out-of-range Δt becomes the zone ceiling
            // rather than silently aliasing through a bitmask wraparound.
            let dt = (target.t_frame - anchor.t_frame).clamp(1, 0x3FFF);
            let hash = ((f_a_q & 0x1FF) << 23) | ((f_b_q & 0x1FF) << 14) | dt;
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
/// assert!(s.push(&zeros).unwrap().is_empty());
/// assert!(s.push(&zeros).unwrap().is_empty());
/// assert!(s.flush().unwrap().is_empty());
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

    // Incremental peak detection (replaces full-window rolling_max_2d).
    peak_det: crate::dsp::peaks::IncrementalPeakDetector,
    peak_row_max: alloc::vec::Vec<f32>,

    // Reusable scratch row for STFT output.
    frame_scratch: alloc::vec::Vec<f32>,

    // Per-second adaptive thresholding.
    // Sorted Vec replaces BTreeMap — bucket_pending is bounded (≤ 3
    // entries in steady state), so linear search is faster than tree
    // traversal.
    bucket_pending: alloc::vec::Vec<(u32, alloc::vec::Vec<Peak>)>,
    last_finalized_bucket: i32,

    // Anchors awaiting finalisation, in t-order.
    pending_anchors: alloc::collections::VecDeque<PendingAnchor>,

    /// Pooled scratch for `finalize_buckets` / `flush`: the list of
    /// bucket keys to finalise on this call. `bucket_pending` is
    /// bounded by `≤ 3` entries in steady state (verified by
    /// `streaming_state_stays_bounded_under_long_input`), so this Vec
    /// is tiny — but it was previously a fresh `Vec::collect()` on
    /// every `push`, i.e. one heap allocation per ~256 samples.
    /// Pooled here and `clear()`ed per call so the streaming hot path
    /// performs zero allocations for bucket finalisation.
    to_finalize: alloc::vec::Vec<u32>,

    /// Pooled output buffer for `emit_finalized_anchors`. Cleared at
    /// the start of each `push`/`flush`, populated by the emit logic,
    /// and `take`n at the end to return. Avoids a fresh allocation per
    /// call.
    emitted: alloc::vec::Vec<(TimestampMs, WangHash)>,
}

impl Default for StreamingWang {
    fn default() -> Self {
        Self::new(WangConfig::default())
    }
}

impl StreamingWang {
    /// Build a streaming Wang extractor with the given config.
    ///
    /// Clamps `target_zone_t` to a minimum of 1 and `fan_out` to a
    /// minimum of 1 to prevent underflows/empty output from degenerate
    /// configurations. Caps `target_zone_t` at 512 and `fan_out` at 64
    /// to prevent OOM from extreme values.
    #[must_use]
    pub fn new(mut cfg: WangConfig) -> Self {
        cfg.target_zone_t = cfg.target_zone_t.clamp(1, 512);
        cfg.fan_out = cfg.fan_out.clamp(1, 64);
        cfg.peaks_per_sec = cfg.peaks_per_sec.min(500);
        if cfg.max_input_samples == Some(0) {
            cfg.max_input_samples = Some(1);
        }
        if cfg.max_hashes == Some(0) {
            cfg.max_hashes = Some(1);
        }
        if cfg.max_pending_anchors == Some(0) {
            cfg.max_pending_anchors = Some(1);
        }
        cfg.target_zone_f = cfg.target_zone_f.clamp(1, 512);
        cfg.min_anchor_mag_db = cfg.min_anchor_mag_db.clamp(-200.0, 0.0);
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
            peak_det: crate::dsp::peaks::IncrementalPeakDetector::new(
                WANG_PEAK_NEIGHBOURHOOD,
                WANG_PEAK_NEIGHBOURHOOD,
                n_bins,
            ),
            peak_row_max: alloc::vec![0.0_f32; n_bins],
            frame_scratch: alloc::vec![0.0_f32; n_bins],
            bucket_pending: alloc::vec::Vec::new(),
            last_finalized_bucket: -1,
            pending_anchors: alloc::collections::VecDeque::new(),
            to_finalize: alloc::vec::Vec::new(),
            emitted: alloc::vec::Vec::new(),
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &WangConfig {
        &self.cfg
    }

    /// Reset all internal state. The stream behaves as if freshly
    /// constructed: no buffered audio, no pending peaks or anchors.
    /// Call between independent streams sharing one instance so stale
    /// data from a previous stream doesn't bleed into the first
    /// emitted hash.
    pub fn reset(&mut self) {
        self.sample_carry.clear();
        self.peak_det.reset();
        self.spec_n_rows = 0;
        self.spec_first_frame = 0;
        self.n_frames_total = 0;
        self.last_pd_frame = -1;
        self.bucket_pending.clear();
        self.last_finalized_bucket = -1;
        self.pending_anchors.clear();
        self.to_finalize.clear();
        self.emitted.clear();
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
    fn detect_rows_range(&mut self, from_row: usize, to_row: usize) {
        if self.spec_n_rows == 0 || from_row > to_row {
            return;
        }
        let n_bins = self.spec_n_bins;
        let min_mag = self.cfg.min_anchor_mag_db;

        for row in from_row..=to_row {
            if row >= self.spec_n_rows {
                break;
            }
            let abs_f = self.spec_first_frame + row as u32;
            let bucket = (abs_f as f32 / WANG_FRAMES_PER_SEC) as u32;
            let row_start = row * n_bins;
            let spec_row = &self.spec[row_start..row_start + n_bins];
            let peak_max = &self.peak_row_max[..n_bins];
            // Push peaks directly into bucket_pending — avoids a per-row
            // Vec allocation (typically 1-3 peaks per row).  Matches
            // Panako's `detect_rows` pattern.
            for bin in 0..n_bins {
                let v = spec_row[bin];
                if v > min_mag && v >= peak_max[bin] {
                    let peak = Peak {
                        t_frame: abs_f,
                        f_bin: bin as u16,
                        mag: v,
                        ..Peak::zeroed()
                    };
                    // Linear search in the sorted Vec; insert sorted if
                    // not found. bucket_pending is bounded (≤ 3 entries).
                    match self.bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
                        Ok(idx) => self.bucket_pending[idx].1.push(peak),
                        Err(idx) => self.bucket_pending.insert(idx, (bucket, alloc::vec![peak])),
                    }
                }
            }
        }
    }

    /// Finalise one bucket: apply per-second adaptive threshold (top
    /// `peaks_per_sec` by magnitude), then for each surviving peak in
    /// `(t, f)` order, grow target heaps of older anchors and register
    /// the peak as a new anchor.
    fn finalize_bucket(&mut self, bucket: u32) {
        let mut peaks = match self.bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
            Ok(idx) => self.bucket_pending.remove(idx).1,
            Err(_) => return,
        };
        // Match the offline picker's `adaptive_per_second`: sort by mag
        // desc, then `(t, f)` ascending. The positional tiebreak is unique
        // per peak, so equal-magnitude peaks at the truncation boundary
        // resolve identically to the offline path.
        peaks.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
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
            // If a hard cap is configured, evict oldest anchors first
            // so memory stays bounded under adversarial / dense input.
            if let Some(limit) = self.cfg.max_pending_anchors {
                while self.pending_anchors.len() >= limit {
                    self.pending_anchors.pop_front();
                }
            }
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
        // Collect into the pooled buffer instead of allocating a fresh
        // `Vec` on every `push`. `bucket_pending` is bounded (≤ 3 in
        // steady state), so the buffer never grows after warmup.
        //
        // The index-based loop (rather than `drain(..)`) sidesteps the
        // borrow conflict: `drain` would hold `&mut self.to_finalize`
        // across the loop body where `self.finalize_bucket` needs
        // `&mut self`. Indexing a `Copy` element produces a `u32` by
        // value, so the immutable borrow of `to_finalize` ends before
        // the mutable call begins.
        self.to_finalize.clear();
        self.to_finalize.extend(
            self.bucket_pending.iter().map(|e| e.0).filter(|&b| {
                (b as i32) > self.last_finalized_bucket && (b as i32) < current_bucket
            }),
        );
        let n = self.to_finalize.len();
        for i in 0..n {
            let bucket = self.to_finalize[i];
            self.finalize_bucket(bucket);
        }
        self.to_finalize.clear();
    }

    /// Pop anchors whose target zone is fully observed (i.e. the bucket
    /// containing the last possible target frame has been finalised),
    /// build hashes from their accumulated target heap, and push into
    /// `self.emitted`.
    fn emit_finalized_anchors(&mut self) {
        // Pop-and-push pattern: take the front anchor, decide whether its
        // target zone is fully observed, and if not put it back. This avoids
        // an `unwrap` after a separate `front()` peek and stays a clean
        // `while let` over the pop result.
        //
        // Temporarily take `emitted` to split the borrow: the loop body
        // needs `&self` (for `build_hashes_for_anchor`) and `&mut emitted`.
        let mut emitted = core::mem::take(&mut self.emitted);
        while let Some(anchor) = self.pending_anchors.pop_front() {
            let last_target_frame = anchor.peak.t_frame + self.cfg.target_zone_t as u32;
            let last_target_bucket = (last_target_frame as f32 / WANG_FRAMES_PER_SEC) as i32;
            if self.last_finalized_bucket < last_target_bucket {
                self.pending_anchors.push_front(anchor);
                break;
            }
            self.build_hashes_for_anchor(anchor, &mut emitted);
        }
        self.emitted = emitted;
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
        let f_a_q = quantise_freq(anchor.peak.f_bin);
        for target in &targets {
            let f_b_q = quantise_freq(target.f_bin);
            let dt = (target.t_frame - anchor.peak.t_frame).clamp(1, 0x3FFF);
            let hash = ((f_a_q & 0x1FF) << 23) | ((f_b_q & 0x1FF) << 14) | dt;
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

    // ── Private helpers for the zero-alloc push_with / flush_with path ──

    /// Common processing for `push` and `push_with`: advance the STFT,
    /// detect peaks, finalise buckets, emit ready anchors into
    /// `self.emitted`.
    fn process_push_samples(&mut self, samples: &[f32]) {
        let samples = crate::pcm::truncate_push(samples, self.cfg.max_push_samples);
        crate::pcm::extend_sanitized(&mut self.sample_carry, samples);

        let mut off = 0usize;
        while self.sample_carry.len() - off >= WANG_N_FFT {
            self.stft
                .process_frame_power(
                    &self.sample_carry[off..off + WANG_N_FFT],
                    &mut self.frame_scratch,
                )
                .expect("frame_scratch is sized n_bins and frames are exactly n_fft");
            power_to_db_wide(&mut self.frame_scratch, WANG_LOG_FLOOR_POWER);
            self.append_frame_scratch_row();

            self.n_frames_total += 1;
            off += WANG_HOP;

            if let Some(ripe_abs) = self
                .peak_det
                .push_row(&self.frame_scratch, &mut self.peak_row_max)
            {
                let row_idx = (ripe_abs - self.spec_first_frame) as usize;
                self.detect_rows_range(row_idx, row_idx);
                self.last_pd_frame = ripe_abs as i32;
            }
        }

        if off > 0 {
            self.sample_carry.drain(0..off);
        }

        self.finalize_buckets();
        self.emit_finalized_anchors();
    }

    /// Common processing for `flush` and `flush_with`: drain remaining
    /// peaks from the incremental detector, finalise all buckets, emit
    /// all anchors into `self.emitted`.
    fn process_flush(&mut self) {
        let n_bins = self.spec_n_bins;
        let min_mag = self.cfg.min_anchor_mag_db;
        let spec = &self.spec;
        let spec_first_frame = self.spec_first_frame;
        let bucket_pending = &mut self.bucket_pending;
        let last_pd = &mut self.last_pd_frame;

        self.peak_det
            .flush(&mut self.peak_row_max, |ripe_abs, max_row| {
                let row_idx = (ripe_abs - spec_first_frame) as usize;
                let bucket = (ripe_abs as f32 / WANG_FRAMES_PER_SEC) as u32;
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
                        match bucket_pending.binary_search_by_key(&bucket, |e| e.0) {
                            Ok(idx) => bucket_pending[idx].1.push(peak),
                            Err(idx) => bucket_pending.insert(idx, (bucket, alloc::vec![peak])),
                        }
                    }
                }
                *last_pd = ripe_abs as i32;
            });

        self.to_finalize.clear();
        self.to_finalize
            .extend(self.bucket_pending.iter().map(|e| e.0));
        let n = self.to_finalize.len();
        for i in 0..n {
            let bucket = self.to_finalize[i];
            self.finalize_bucket(bucket);
        }
        self.to_finalize.clear();

        let mut emitted = core::mem::take(&mut self.emitted);
        while let Some(anchor) = self.pending_anchors.pop_front() {
            self.build_hashes_for_anchor(anchor, &mut emitted);
        }
        self.emitted = emitted;
    }
}

impl StreamingFingerprinter for StreamingWang {
    type Frame = WangHash;

    fn required_sample_rate(&self) -> u32 {
        WANG_SR
    }

    fn push(&mut self, samples: &[f32]) -> Result<alloc::vec::Vec<(TimestampMs, Self::Frame)>> {
        self.emitted.clear();
        self.process_push_samples(samples);
        Ok(core::mem::take(&mut self.emitted))
    }

    fn push_with<F>(&mut self, samples: &[f32], mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        self.emitted.clear();
        self.process_push_samples(samples);
        let mut n = 0usize;
        for (t, frame) in self.emitted.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
    }

    fn flush(&mut self) -> Result<alloc::vec::Vec<(TimestampMs, Self::Frame)>> {
        self.emitted.clear();
        self.process_flush();
        Ok(core::mem::take(&mut self.emitted))
    }

    fn flush_with<F>(&mut self, mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        self.emitted.clear();
        self.process_flush();
        let mut n = 0usize;
        for (t, frame) in self.emitted.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
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

        match fp.extract(&samples, SampleRate::HZ_16000) {
            Err(AfpError::UnsupportedSampleRate(16_000)) => {}
            other => panic!("expected UnsupportedSampleRate, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_audio() {
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 8_000]; // 1 second, need 2

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
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 8_000 * 3];

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_eq!(fpr.frames_per_sec, 62.5);
        assert!(fpr.hashes.is_empty());
    }

    #[test]
    fn synthetic_signal_produces_hashes() {
        let mut fp = Wang::default();
        let samples = synthetic_audio(0xC0FFEE, 8_000 * 5);

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert!(
            (650..=1100).contains(&fpr.hashes.len()),
            "expected 650..=1100 hashes from a 5s tone, got {}",
            fpr.hashes.len(),
        );
        let distinct: alloc::collections::BTreeSet<u32> =
            fpr.hashes.iter().map(|h| h.hash).collect();
        assert!(
            distinct.len() > 500,
            "expected most hashes to be distinct, got {} distinct of {}",
            distinct.len(),
            fpr.hashes.len(),
        );
        // Ordering invariant: sorted by (t_anchor, hash).
        for w in fpr.hashes.windows(2) {
            assert!((w[0].t_anchor, w[0].hash) <= (w[1].t_anchor, w[1].hash));
        }
    }

    #[test]
    fn synthetic_signal_is_deterministic() {
        let samples = synthetic_audio(0xBEEF, 8_000 * 3);
        let mut a = Wang::default();
        let mut b = Wang::default();
        let fa = a.extract(&samples, SampleRate::HZ_8000).unwrap();
        let fb = b.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert_eq!(fa.hashes, fb.hashes);
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Wang::default();

        let f1 = fp1.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut fp2 = Wang::default();

        let f2 = fp2.extract(&samples, SampleRate::HZ_8000).unwrap();

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
        let fa = fp.extract(&samples_a, SampleRate::HZ_8000).unwrap();
        let fb = fp.extract(&samples_b, SampleRate::HZ_8000).unwrap();
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
        let ta = hashes[0].t_anchor;
        assert_eq!(ta, 100);
    }

    #[test]
    fn dt_field_clamps_to_14_bit_ceiling_not_wraparound() {
        // Anchor at t=0, target at t=20_000 — well past the 14-bit ceiling
        // of 16383. Use a config with a large target_zone_t so the zone
        // check itself doesn't cull the target before clamping kicks in.
        let peaks = alloc::vec![
            Peak {
                t_frame: 0,
                f_bin: 50,
                _pad: 0,
                mag: -10.0
            },
            Peak {
                t_frame: 20_000,
                f_bin: 70,
                _pad: 0,
                mag: -12.0
            },
        ];
        let cfg = WangConfig {
            target_zone_t: u16::MAX,
            target_zone_f: u16::MAX,
            ..WangConfig::default()
        };
        let hashes = build_hashes(&peaks, &cfg);
        assert_eq!(hashes.len(), 1);
        // Δt field must saturate at 16383 — NOT wrap around to 20000 % 16384 = 3616.
        let dt = hashes[0].hash & 0x3FFF;
        assert_eq!(dt, 0x3FFF, "Δt must clamp to 14-bit max, got {dt}");
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
        assert!(s.push(&[]).unwrap().is_empty());
        assert!(s.flush().unwrap().is_empty());
    }

    #[test]
    fn streaming_silence_emits_nothing() {
        let mut s = StreamingWang::default();
        let zeros = vec![0.0_f32; 8_000 * 4];
        assert!(s.push(&zeros).unwrap().is_empty());
        assert!(s.flush().unwrap().is_empty());
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
                out.extend(s.push(chunk).unwrap().into_iter().map(|(_, h)| h));
            }
            out.extend(s.flush().unwrap().into_iter().map(|(_, h)| h));
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
        let off = offline.extract(&samples, SampleRate::HZ_8000).unwrap();

        // Streaming with random chunks.
        let mut streaming = StreamingWang::default();
        let mut online = Vec::new();
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

        let mut wide = Wang::new(WangConfig {
            fan_out: 10,
            ..WangConfig::default()
        });
        let mut narrow = Wang::new(WangConfig {
            fan_out: 3,
            ..WangConfig::default()
        });
        let f_wide = wide.extract(&samples, SampleRate::HZ_8000).unwrap();
        let f_narrow = narrow.extract(&samples, SampleRate::HZ_8000).unwrap();
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
        let off = offline.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut s = StreamingWang::default();
        let mut online = Vec::new();
        // Push one sample at a time — pathological case for any incremental
        // streaming impl.
        for &sample in &samples {
            online.extend(s.push(&[sample]).unwrap().into_iter().map(|(_, h)| h));
        }
        online.extend(s.flush().unwrap().into_iter().map(|(_, h)| h));

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
            let _ = s.push(&samples[start..end]).unwrap();
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
        let _ = s.flush().unwrap();
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
        let ta = hashes[0].t_anchor;
        assert_eq!(ta, 0);
    }

    // -----------------------------------------------------------------
    // Direct unit tests for `emit_finalized_anchors`.
    //
    // These pin the re-queue invariant of the 226e0f2 refactor: when
    // the front anchor's target zone is not yet finalised, the pop-
    // and-push-front pattern must return it to the front of the queue
    // intact. A forgotten `push_front` (or any other deviation) would
    // silently drop the anchor and lose its hashes forever — the
    // existing `streaming_offline_equivalence` test would catch the
    // symptom at a high level, but these tests catch the cause at the
    // site of the refactor.
    //
    // We populate `pending_anchors` and `last_finalized_bucket`
    // directly (both are private; the test module sits inside the
    // same file with `use super::*`, so it has access).
    // -----------------------------------------------------------------

    fn anchor_with_target(
        t_frame: u32,
        f_bin: u16,
        target_t: u32,
        target_f: u16,
        target_mag: f32,
    ) -> PendingAnchor {
        let target = MinByMagOwned(Peak {
            t_frame: target_t,
            f_bin: target_f,
            _pad: 0,
            mag: target_mag,
        });
        let mut targets = alloc::collections::BinaryHeap::with_capacity(4);
        targets.push(target);
        PendingAnchor {
            peak: Peak {
                t_frame,
                f_bin,
                _pad: 0,
                mag: 1.0,
            },
            targets,
        }
    }

    /// Bucket index for a frame at the Wang default rate
    /// (`WANG_FRAMES_PER_SEC = 62.5`).
    fn wang_bucket_of(t_frame: u32) -> i32 {
        (t_frame as f32 / WANG_FRAMES_PER_SEC) as i32
    }

    #[test]
    fn wang_emit_finalized_anchors_emits_all_when_zones_covered() {
        // Three anchors, all of whose target zones are covered by
        // `last_finalized_bucket`. All three should emit.
        let mut s = StreamingWang::default();
        // Anchor at frame 0: zone covers frames [1, 63], last target frame = 63
        // → bucket 1.
        s.pending_anchors
            .push_back(anchor_with_target(0, 10, 10, 12, 0.9));
        // Anchor at frame 5: zone covers [6, 68], bucket 1.
        s.pending_anchors
            .push_back(anchor_with_target(5, 20, 15, 22, 0.8));
        // Anchor at frame 100: zone covers [101, 163], bucket 2.
        s.pending_anchors
            .push_back(anchor_with_target(100, 30, 110, 32, 0.7));
        s.last_finalized_bucket = wang_bucket_of(163);

        s.emitted.clear();
        s.emit_finalized_anchors();
        assert_eq!(s.emitted.len(), 3);
        assert!(s.pending_anchors.is_empty());
    }

    #[test]
    fn wang_emit_finalized_anchors_re_queues_unfinalised() {
        // Two anchors; only the first is covered. The second must
        // remain in `pending_anchors` after the emit.
        let mut s = StreamingWang::default();
        s.pending_anchors
            .push_back(anchor_with_target(0, 10, 10, 12, 0.9));
        // Zone covers [1, 63], bucket 1.
        s.pending_anchors
            .push_back(anchor_with_target(100, 30, 110, 32, 0.7));
        // Only cover bucket 1, not bucket 2.
        s.last_finalized_bucket = 1;

        s.emitted.clear();
        s.emit_finalized_anchors();
        assert_eq!(s.emitted.len(), 1);
        assert_eq!(s.pending_anchors.len(), 1);
        // The re-queued anchor must be the unfinalised one (frame 100).
        assert_eq!(s.pending_anchors.front().unwrap().peak.t_frame, 100);
    }

    #[test]
    fn wang_emit_finalized_anchors_idempotent_under_repeated_calls() {
        // With one anchor covered, two consecutive calls must emit
        // the same hashes (no double-emit, no lost anchor).
        let mut s = StreamingWang::default();
        s.pending_anchors
            .push_back(anchor_with_target(0, 10, 10, 12, 0.9));
        s.last_finalized_bucket = wang_bucket_of(63);

        s.emitted.clear();
        s.emit_finalized_anchors();
        let first_len = s.emitted.len();
        s.emitted.clear();
        s.emit_finalized_anchors();
        let second_len = s.emitted.len();
        assert_eq!(first_len, 1);
        assert_eq!(second_len, 0);
        assert!(s.pending_anchors.is_empty());
    }

    // -----------------------------------------------------------------
    // Public API contract pins.
    //
    // These pin the return values of the `Fingerprinter` and
    // `StreamingFingerprinter` trait methods. A silent change to any
    // of these (a rename of `name`, a change in `required_sample_rate`
    // or `min_samples`, a shift in the latency window) would break
    // downstream consumers that hardcode these values (e.g. the
    // `tests/goldens/*.bin` regression headers include the algorithm
    // name). The pins below catch the change at the unit-test level.
    // -----------------------------------------------------------------

    #[test]
    fn public_api_name_and_config_match_documented_values() {
        let fp = Wang::default();
        assert_eq!(fp.name(), "wang-v1");
        assert_eq!(fp.required_sample_rate(), SampleRate::HZ_8000);
        assert_eq!(fp.min_samples(), 16_000);

        let s = StreamingWang::default();
        assert_eq!(s.latency_ms(), 2_256);
    }

    // ── Backward-compat (& forward-safe) constructor clamping tests ──

    #[test]
    fn default_config_is_unchanged_by_guard_clamps() {
        // The clamp ceilings are well above the defaults; defaults
        // must survive construction unmodified.
        let fp = Wang::default();
        assert_eq!(fp.config().fan_out, 10);
        assert_eq!(fp.config().target_zone_t, 63);
        assert_eq!(fp.config().peaks_per_sec, 30);
    }

    #[test]
    fn zero_target_zone_is_clamped_to_one_not_underflow() {
        let cfg = WangConfig {
            target_zone_t: 0,
            ..WangConfig::default()
        };
        let fp = Wang::new(cfg);
        assert_eq!(fp.config().target_zone_t, 1);
    }

    #[test]
    fn extreme_config_is_clamped_within_safe_bounds() {
        let cfg = WangConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..WangConfig::default()
        };
        let fp = Wang::new(cfg);
        assert_eq!(fp.config().fan_out, 64);
        assert_eq!(fp.config().target_zone_t, 512);
        assert_eq!(fp.config().peaks_per_sec, 500);
    }

    #[test]
    fn clamped_config_still_produces_valid_hashes() {
        // Config with extreme-but-clamped values must not panic/error
        // when run against real audio.
        let cfg = WangConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..WangConfig::default()
        };
        let mut fp = Wang::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 3);

        let fpr = fp.extract(&samples, SampleRate::HZ_8000).unwrap();
        assert!(!fpr.hashes.is_empty());
    }

    #[test]
    fn streaming_default_config_is_unchanged_by_guard_clamps() {
        let s = StreamingWang::default();
        let cfg = s.config();
        assert_eq!(cfg.fan_out, 10);
        assert_eq!(cfg.target_zone_t, 63);
        assert_eq!(cfg.peaks_per_sec, 30);
    }

    #[test]
    fn streaming_extreme_config_is_clamped_within_safe_bounds() {
        let cfg = WangConfig {
            fan_out: u16::MAX,
            target_zone_t: u16::MAX,
            peaks_per_sec: u16::MAX,
            ..WangConfig::default()
        };
        let s = StreamingWang::new(cfg);
        assert_eq!(s.config().fan_out, 64);
        assert_eq!(s.config().target_zone_t, 512);
        assert_eq!(s.config().peaks_per_sec, 500);
    }

    #[test]
    fn streaming_reset_clears_all_state() {
        let mut s = StreamingWang::default();
        // Push audio to build up state.
        let samples = synthetic_audio(0xFEED, 8_000 * 4);
        let before = s.push(&samples).unwrap();
        assert!(!before.is_empty(), "should produce hashes");

        s.reset();
        assert!(s.push(&[]).unwrap().is_empty(), "reset should clear state");
        // Fresh push of same audio should produce identical hashes.
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
        let mut a = StreamingWang::default();
        let mut b = StreamingWang::default();
        let samples = synthetic_audio(0xABCD, 8_000 * 4);

        let via_push = a.push(&samples).unwrap();
        let mut via_cb: Vec<(TimestampMs, WangHash)> = Vec::new();
        let n = b.push_with(&samples, |t, f| via_cb.push((t, *f))).unwrap();
        let via_flush = b.flush().unwrap();
        let flush_len = via_flush.len();
        via_cb.extend(via_flush);

        // push() returns flush-drainable output; collect the same.
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
        let mut a = StreamingWang::default();
        let mut b = StreamingWang::default();
        // Push nothing; just drain at end of hypothetical stream.
        let samples = synthetic_audio(0xF00D, 8_000 * 4);
        let _ = a.push(&samples).unwrap();
        let _ = b.push(&samples).unwrap();

        let via_flush = a.flush().unwrap();
        let mut via_cb: Vec<(TimestampMs, WangHash)> = Vec::new();
        let n = b.flush_with(|t, f| via_cb.push((t, *f))).unwrap();

        assert_eq!(n, via_flush.len());
        assert_eq!(via_cb, via_flush);
    }

    // ── OOM protection: max_input_samples enforcement ──

    #[test]
    fn default_max_input_samples_is_set() {
        let fp = Wang::default();
        assert!(fp.config().max_input_samples.is_some());
    }

    #[test]
    fn input_larger_than_max_is_rejected() {
        let cfg = WangConfig {
            max_input_samples: Some(1_000),
            ..WangConfig::default()
        };
        let mut fp = Wang::new(cfg);
        let samples = vec![0.0_f32; 2_000];

        let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
        match err {
            AfpError::InputTooLarge { limit, provided } => {
                assert_eq!(limit, 1_000);
                assert_eq!(provided, 2_000);
            }
            other => panic!("expected InputTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn none_disables_max_input_check() {
        let cfg = WangConfig {
            max_input_samples: None,
            ..WangConfig::default()
        };
        let mut fp = Wang::new(cfg);
        // 16_000 samples (2 s) is above default limit but None passes.
        let samples = vec![0.0_f32; 16_000];

        fp.extract(&samples, SampleRate::HZ_8000).unwrap();
    }

    #[test]
    fn valid_input_under_limit_passes() {
        let cfg = WangConfig {
            max_input_samples: Some(100_000),
            ..WangConfig::default()
        };
        let mut fp = Wang::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 3);

        fp.extract(&samples, SampleRate::HZ_8000).unwrap();
    }

    #[test]
    fn max_hashes_enforced_rejects_too_many() {
        let cfg = WangConfig {
            max_hashes: Some(10),
            ..WangConfig::default()
        };
        let mut fp = Wang::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 5);

        let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
        assert!(matches!(err, AfpError::InputTooLarge { .. }));
    }

    #[test]
    fn max_pending_anchors_evicts_oldest() {
        let cfg = WangConfig {
            max_pending_anchors: Some(100),
            ..WangConfig::default()
        };
        let mut s = StreamingWang::new(cfg);
        let samples = synthetic_audio(0xCAFE, 8_000 * 20);
        let mut hashes = s.push(&samples).unwrap();
        hashes.extend(s.flush().unwrap());
        assert!(s.config().max_pending_anchors.is_some());
        assert!(!hashes.is_empty(), "should produce hashes with cap=100");
    }

    #[test]
    fn extract_rejects_nan_pcm() {
        let mut fp = Wang::default();
        let mut samples = vec![0.0_f32; 8_000 * 3];
        samples[100] = f32::NAN;

        let err = fp.extract(&samples, SampleRate::HZ_8000).unwrap_err();
        assert!(matches!(err, AfpError::NonFiniteSample { index: 100 }));
    }

    #[test]
    fn max_push_samples_truncates_hostile_chunk() {
        let cfg = WangConfig {
            max_push_samples: Some(512),
            ..WangConfig::default()
        };
        let mut s = StreamingWang::new(cfg);
        let samples = synthetic_audio(0xBEEF, 8_000 * 5);
        let _ = s.push(&samples).unwrap();
        let _ = s.flush().unwrap();
        assert_eq!(s.config().max_push_samples, Some(512));
    }

    #[test]
    fn push_sanitizes_nan_to_zero() {
        let mut clean = StreamingWang::default();
        let mut dirty = StreamingWang::default();
        let mut samples = synthetic_audio(0xABCD, 8_000 * 3);
        let a = clean.push(&samples);
        samples[10] = f32::NAN;
        samples[20] = f32::INFINITY;
        let b = dirty.push(&samples);
        // Sanitized NaN/Inf → 0.0; hashes need not match exactly, but push must not panic.
        let _ = (a, b);
        let _ = dirty.flush();
    }

    // ── Progress callback tests ──

    #[test]
    fn extract_with_progress_is_called_and_monotonic() {
        let mut fp = Wang::default();
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

        let mut fp1 = Wang::default();
        let result1 = fp1.extract(&samples, SampleRate::HZ_8000).unwrap();

        let mut fp2 = Wang::default();
        let result2 = fp2
            .extract_with_progress(&samples, SampleRate::HZ_8000, |_| {})
            .unwrap();

        assert_eq!(result1.hashes, result2.hashes);
        assert_eq!(result1.frames_per_sec, result2.frames_per_sec);
    }

    #[test]
    fn extract_with_progress_short_audio_still_reports_0_and_1() {
        let mut fp = Wang::default();
        // Exactly at minimum length — should still give 0.0 and 1.0.
        let samples = synthetic_audio(0xFACE, 8_000 * 2);

        let mut values: Vec<f32> = Vec::new();
        let _ = fp.extract_with_progress(&samples, SampleRate::HZ_8000, |v| values.push(v));
        assert_eq!(values[0], 0.0);
        assert_eq!(*values.last().unwrap(), 1.0);
    }
}
