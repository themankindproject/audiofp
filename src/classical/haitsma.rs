//! Haitsma–Kalker / Philips robust hash.
//!
//! Reference: Haitsma, J. & Kalker, T. "A Highly Robust Audio
//! Fingerprinting System." Proceedings of the 3rd International
//! Conference on Music Information Retrieval (ISMIR), Paris, France,
//! 2002. <https://www.ismir.net/resources/ismir-conferences/>
//!
//! Algorithm:
//!
//! 1. Resample the input to 5 kHz mono *(caller's responsibility)*.
//! 2. Take a Hann-windowed STFT with `n_fft = 2048`, `hop = 64`
//!    (≈78.125 frames/s).
//! 3. Sum power across **33 logarithmically-spaced bands** from 300 Hz
//!    to 2000 Hz → `E[n][b]`.
//! 4. For each frame `n ≥ 1` and band index `b ∈ {0..=31}` emit one bit:
//!
//!    ```text
//!    F[n][b] = ((E[n][b] − E[n][b+1]) − (E[n−1][b] − E[n−1][b+1])) > 0
//!    ```
//!
//! 5. Pack the 32 bits per frame into a `u32` with band 0 in the most
//!    significant bit (the spec's "MSB-zero" ordering) and band 31 in
//!    the least significant.
//!
//! ## Divergence from the paper
//!
//! The bit-packing order in this implementation (band 0 → bit 31,
//! "MSB-zero") is **not** the natural band-index order described in
//! the Haitsma & Kalker 2002 paper. The paper packs bands in their
//! natural index order. This is a deliberate, stable divergence — the
//! `haitsma-v1` hash layout in [`Haitsma::name`] is part of the crate's
//! versioned contract, and changing it would invalidate every
//! persisted `haitsma-v1` fingerprint. Callers porting an existing
//! Haitsma database must XOR or byte-reverse each 32-bit frame before
//! comparison.

use alloc::vec;
use alloc::vec::Vec;

use libm::powf;

use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, Fingerprinter, Result, SampleRate, StreamingFingerprinter, TimestampMs};

/// All bit-frames produced by [`Haitsma`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct HaitsmaFingerprint {
    /// One `u32` per STFT frame from `n=1` onwards.
    pub frames: Vec<u32>,
    /// Frame rate of the underlying STFT — always 78.125 for `haitsma-v1`
    /// (`5000 / 64`).
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Haitsma`].
///
/// Always construct with FRU so future additive fields stay compatible:
/// `HaitsmaConfig { fmin: 400.0, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct HaitsmaConfig {
    /// Lowest band edge in Hz. Default 300.
    pub fmin: f32,
    /// Highest band edge in Hz. Default 2000.
    pub fmax: f32,
    /// Maximum input sample count accepted by [`extract`]. `None` disables
    /// the check. Default: 9_000_000 (30 minutes at 5 kHz).
    ///
    /// [`extract`]: Haitsma::extract
    pub max_input_samples: Option<usize>,
    /// Maximum samples accepted in a single streaming `push`. `None`
    /// disables (default). Excess samples are dropped.
    pub max_push_samples: Option<usize>,
}

impl Default for HaitsmaConfig {
    fn default() -> Self {
        Self {
            fmin: 300.0,
            fmax: 2_000.0,
            max_input_samples: Some(30 * 60 * HAITSMA_SR as usize),
            max_push_samples: None,
        }
    }
}

const HAITSMA_N_FFT: usize = 2048;
const HAITSMA_HOP: usize = 64;
const HAITSMA_SR: u32 = 5_000;
const HAITSMA_FRAMES_PER_SEC: f32 = HAITSMA_SR as f32 / HAITSMA_HOP as f32;
const HAITSMA_N_BANDS: usize = 33;

/// Sentinel value meaning "this FFT bin falls outside the band range".
const NO_BAND: u8 = u8::MAX;

/// Haitsma–Kalker offline fingerprinter.
///
/// # Example
///
/// ```
/// use audiofp::{Fingerprinter, SampleRate};
/// use audiofp::classical::Haitsma;
///
/// let mut fp = Haitsma::default();
/// let samples = vec![0.0_f32; 5_000 * 3];
///
/// // Wrong rate is rejected immediately.
/// assert!(fp.extract(&samples, SampleRate::HZ_8000).is_err());
///
/// let fpr = fp.extract(&samples, SampleRate::HZ_5000).unwrap();
/// assert_eq!(fpr.frames_per_sec, 78.125);
/// // Silence → all-zero hash frames (no band differences).
/// for &h in &fpr.frames {
///     assert_eq!(h, 0);
/// }
/// ```
pub struct Haitsma {
    cfg: HaitsmaConfig,
    stft: ShortTimeFFT,
    /// Precomputed contiguous bin ranges for each band. `band_ranges[b] =
    /// (start_bin, end_bin)` (exclusive end). Eliminates per-bin branching
    /// in the energy accumulation loop and enables SIMD auto-vectorization.
    band_ranges: Vec<(usize, usize)>,
    /// Reused buffer for per-frame band energies across `extract` calls.
    energies_buf: Vec<[f32; HAITSMA_N_BANDS]>,
    /// Reused buffer for packed frame hashes across `extract` calls.
    frames_buf: Vec<u32>,
    /// Reused buffer for STFT power spectrogram across `extract` calls.
    power_buf: Vec<f32>,
}

impl Default for Haitsma {
    fn default() -> Self {
        Self::new(HaitsmaConfig::default())
    }
}

impl Haitsma {
    /// Build a Haitsma extractor with the given config.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.fmin <= 0`, `cfg.fmax <= cfg.fmin`, or
    /// `cfg.fmax >= HAITSMA_SR / 2` (above Nyquist). Use [`try_new`]
    /// for a fallible alternative.
    ///
    /// [`try_new`]: Haitsma::try_new
    #[must_use]
    pub fn new(cfg: HaitsmaConfig) -> Self {
        Self::try_new(cfg).expect("invalid HaitsmaConfig (see AfpError::Config)")
    }

    /// Fallible constructor — returns [`AfpError::Config`] on invalid
    /// `fmin`/`fmax`/Nyquist instead of panicking.
    pub fn try_new(cfg: HaitsmaConfig) -> crate::Result<Self> {
        if cfg.fmin <= 0.0 || cfg.fmin.is_nan() {
            return Err(crate::AfpError::Config("fmin must be positive".into()));
        }
        if cfg.fmax <= cfg.fmin || cfg.fmax.is_nan() {
            return Err(crate::AfpError::Config("fmax must exceed fmin".into()));
        }
        if cfg.fmax >= HAITSMA_SR as f32 / 2.0 {
            return Err(crate::AfpError::Config(alloc::format!(
                "fmax must be below Nyquist ({} Hz)",
                HAITSMA_SR / 2
            )));
        }

        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: HAITSMA_N_FFT,
            hop: HAITSMA_HOP,
            window: WindowKind::Hann,
            center: false,
        });

        let bin_to_band = build_bin_to_band(&cfg, stft.n_bins());
        let band_ranges = build_band_ranges(&bin_to_band);

        Ok(Self {
            cfg,
            stft,
            band_ranges,
            energies_buf: Vec::new(),
            frames_buf: Vec::new(),
            power_buf: Vec::new(),
        })
    }
}

/// Progress callback reporting interval for Haitsma (78.125 fps):
/// every 39 frames ≈ 500 ms of audio.
const HAITSMA_PROGRESS_INTERVAL: usize = 39;

impl Haitsma {
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
    ) -> Result<HaitsmaFingerprint> {
        crate::pcm::reject_non_finite(samples)?;
        if let Some(limit) = self.cfg.max_input_samples
            && samples.len() > limit
        {
            return Err(AfpError::InputTooLarge {
                limit,
                provided: samples.len(),
            });
        }
        if rate.hz() != HAITSMA_SR {
            return Err(AfpError::UnsupportedSampleRate(rate.hz()));
        }
        if samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: samples.len(),
            });
        }

        progress(0.0);

        // Pull power directly — band energy is `Σ |X|²`, so the previous
        // path's `m * m` after a `sqrt(|X|²)` was redundant.
        let (n_frames, n_bins) = self.stft.power_flat_into(samples, &mut self.power_buf);
        let power_flat = &self.power_buf;
        if n_frames < 2 {
            // Frame 0 has no hash, and every hash at frame n needs frame
            // n−1's band energies — fewer than 2 frames yields zero hashes.
            progress(1.0);
            return Ok(HaitsmaFingerprint {
                frames: Vec::new(),
                frames_per_sec: HAITSMA_FRAMES_PER_SEC,
            });
        }

        // Report STFT phase progress (~50% of total work for Haitsma).
        let total_frames = n_frames;
        let stft_weight = 0.50_f32;
        let interval = HAITSMA_PROGRESS_INTERVAL;
        {
            let mut reported = 0usize;
            while reported + interval < total_frames {
                reported += interval;
                progress(stft_weight * (reported as f32 / total_frames as f32));
            }
        }
        progress(stft_weight);

        self.energies_buf.clear();
        self.energies_buf.reserve(n_frames);
        for f in 0..n_frames {
            let row = &power_flat[f * n_bins..(f + 1) * n_bins];
            let mut e = [0.0_f32; HAITSMA_N_BANDS];
            for (b, &(start, end)) in self.band_ranges.iter().enumerate() {
                e[b] = row[start..end].iter().sum();
            }
            self.energies_buf.push(e);

            // Report progress during band energy computation (~30% of work).
            if (f + 1) % interval == 0 {
                let band_progress = stft_weight + 0.30 * ((f + 1) as f32 / total_frames as f32);
                progress(band_progress);
            }
        }
        progress(0.80);

        self.frames_buf.clear();
        self.frames_buf.reserve(self.energies_buf.len() - 1);
        for n in 1..self.energies_buf.len() {
            self.frames_buf.push(pack_frame_bits(
                &self.energies_buf[n],
                &self.energies_buf[n - 1],
            ));
        }

        progress(1.0);

        // Move ownership into the return value; the struct keeps capacity.
        Ok(HaitsmaFingerprint {
            frames: core::mem::take(&mut self.frames_buf),
            frames_per_sec: HAITSMA_FRAMES_PER_SEC,
        })
    }
}

impl Fingerprinter for Haitsma {
    type Output = HaitsmaFingerprint;
    type Config = HaitsmaConfig;

    fn name(&self) -> &'static str {
        "haitsma-v1"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> SampleRate {
        // HAITSMA_SR is a compile-time constant; unwrap is trivially safe.
        SampleRate::new(HAITSMA_SR).expect("HAITSMA_SR is non-zero")
    }

    fn min_samples(&self) -> usize {
        HAITSMA_SR as usize * 2
    }

    fn extract(&mut self, samples: &[f32], rate: SampleRate) -> Result<Self::Output> {
        self.extract_with_progress(samples, rate, |_| {})
    }
}

/// Pack 32 sign bits comparing band-difference deltas between frame `n`
/// and frame `n−1`.
#[inline]
fn pack_frame_bits(curr: &[f32; HAITSMA_N_BANDS], prev: &[f32; HAITSMA_N_BANDS]) -> u32 {
    // SIMD loop assumes 4 chunks × 8 lanes = 32 bits from 33 bands.
    // If HAITSMA_N_BANDS ever changes, this function must be updated.
    debug_assert_eq!(HAITSMA_N_BANDS, 4 * 8 + 1);

    let mut hash = 0_u32;

    // Vectorize the band-difference computation using f32x8.
    // Compute all 32 diffs in 4 SIMD iterations, then extract sign bits.
    // Each iteration reads curr[off..off+8] and curr[off+1..off+9] to
    // compute 8 adjacent-band differences from the 33-band array.
    use wide::f32x8;

    for chunk in 0..4 {
        let off = chunk * 8;
        let curr_lo = f32x8::new(
            curr[off..off + 8]
                .try_into()
                .expect("curr[off..off+8] is exactly 8 elements: 33-band array with chunk<4"),
        );
        let curr_hi = f32x8::new(
            curr[off + 1..off + 9]
                .try_into()
                .expect("curr[off+1..off+9] is exactly 8 elements: 33-band array with chunk<4"),
        );
        let prev_lo = f32x8::new(
            prev[off..off + 8]
                .try_into()
                .expect("prev[off..off+8] is exactly 8 elements: 33-band array with chunk<4"),
        );
        let prev_hi = f32x8::new(
            prev[off + 1..off + 9]
                .try_into()
                .expect("prev[off+1..off+9] is exactly 8 elements: 33-band array with chunk<4"),
        );

        // diff[i] = (curr[i] - curr[i+1]) - (prev[i] - prev[i+1])
        let diff = (curr_lo - curr_hi) - (prev_lo - prev_hi);
        let arr = diff.to_array();

        // Pack sign bits: band at offset+i maps to bit (31 - offset - i).
        for (i, &d) in arr.iter().enumerate() {
            hash |= ((d > 0.0) as u32) << (31 - off - i);
        }
    }

    hash
}

/// Precompute contiguous bin ranges for each band.
/// `band_ranges[b] = (first_bin_inclusive, last_bin_exclusive)` for band b.
fn build_band_ranges(bin_to_band: &[u8]) -> Vec<(usize, usize)> {
    let mut ranges = vec![(0usize, 0usize); HAITSMA_N_BANDS];
    let mut found_start = [false; HAITSMA_N_BANDS];
    for (i, &b) in bin_to_band.iter().enumerate() {
        if b != NO_BAND {
            let band = b as usize;
            if !found_start[band] {
                ranges[band].0 = i;
                found_start[band] = true;
            }
            ranges[band].1 = i + 1;
        }
    }
    ranges
}

/// Compute the FFT-bin → band-index lookup table.
///
/// 33 bands defined by 34 logarithmically-spaced edges from `fmin` to
/// `fmax`. Bin `i`'s frequency is `i · sr / n_fft`. A bin in
/// `[edge_b, edge_{b+1})` is mapped to band `b`. Bins outside the range
/// are mapped to `NO_BAND` (255).
fn build_bin_to_band(cfg: &HaitsmaConfig, n_bins: usize) -> Vec<u8> {
    let n_edges = HAITSMA_N_BANDS + 1;
    let mut edges = [0.0_f32; HAITSMA_N_BANDS + 1];
    let ratio = cfg.fmax / cfg.fmin;
    for (k, e) in edges.iter_mut().enumerate() {
        let frac = k as f32 / HAITSMA_N_BANDS as f32;
        *e = cfg.fmin * powf(ratio, frac);
    }

    let bin_hz = HAITSMA_SR as f32 / HAITSMA_N_FFT as f32;

    let mut out = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let f = i as f32 * bin_hz;
        if f < edges[0] || f >= edges[n_edges - 1] {
            out.push(NO_BAND);
            continue;
        }
        // Linear scan is fine — only 33 edges.
        let mut found = NO_BAND;
        for b in 0..HAITSMA_N_BANDS {
            if f >= edges[b] && f < edges[b + 1] {
                found = b as u8;
                break;
            }
        }
        out.push(found);
    }
    out
}

/// Streaming Haitsma–Kalker fingerprinter.
///
/// Trivially incremental: each output bit-frame depends only on the
/// current and previous frames' band energies, so we just keep one
/// previous-frame energy vector. No spectrogram window, no peak picker,
/// no per-second adaptive threshold. Per-push CPU cost is proportional
/// to the number of new samples, independent of total stream length,
/// and latency is bounded by the STFT window length (`n_fft / sr ≈
/// 410 ms`) — much lower than the landmark extractors.
///
/// Output is bit-exactly equivalent to [`Haitsma::extract`].
pub struct StreamingHaitsma {
    cfg: HaitsmaConfig,

    stft: ShortTimeFFT,
    sample_carry: Vec<f32>,
    /// Precomputed contiguous bin ranges for each band.
    band_ranges: Vec<(usize, usize)>,

    /// Per-bin scratch for one frame's power spectrum.
    frame_power: Vec<f32>,

    /// Whether we've seen any frame at all (frame 0 has no hash but we
    /// still need its band energies as the "prev" for frame 1).
    has_prev: bool,
    prev_energy: [f32; HAITSMA_N_BANDS],

    /// Next absolute frame index whose hash hasn't been emitted yet
    /// (1-based — frame 0 has no hash).
    next_frame_idx: u32,

    /// Output frames produced but not yet drained (incremental push).
    pending: Vec<(TimestampMs, u32)>,
}

impl Default for StreamingHaitsma {
    fn default() -> Self {
        Self::new(HaitsmaConfig::default())
    }
}

impl StreamingHaitsma {
    /// Build a streaming Haitsma extractor with the given config.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.fmin <= 0`, `cfg.fmax <= cfg.fmin`, or
    /// `cfg.fmax >= HAITSMA_SR / 2` (above Nyquist). Use [`try_new`]
    /// for a fallible alternative.
    ///
    /// [`try_new`]: StreamingHaitsma::try_new
    #[must_use]
    pub fn new(cfg: HaitsmaConfig) -> Self {
        Self::try_new(cfg).expect("invalid HaitsmaConfig (see AfpError::Config)")
    }

    /// Fallible constructor — returns [`AfpError::Config`] on invalid
    /// `fmin`/`fmax`/Nyquist instead of panicking.
    pub fn try_new(cfg: HaitsmaConfig) -> crate::Result<Self> {
        if cfg.fmin <= 0.0 || cfg.fmin.is_nan() {
            return Err(crate::AfpError::Config("fmin must be positive".into()));
        }
        if cfg.fmax <= cfg.fmin || cfg.fmax.is_nan() {
            return Err(crate::AfpError::Config("fmax must exceed fmin".into()));
        }
        if cfg.fmax >= HAITSMA_SR as f32 / 2.0 {
            return Err(crate::AfpError::Config(alloc::format!(
                "fmax must be below Nyquist ({} Hz)",
                HAITSMA_SR / 2
            )));
        }

        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: HAITSMA_N_FFT,
            hop: HAITSMA_HOP,
            window: WindowKind::Hann,
            center: false,
        });
        let bin_to_band = build_bin_to_band(&cfg, stft.n_bins());
        let band_ranges = build_band_ranges(&bin_to_band);
        let n_bins = stft.n_bins();
        Ok(Self {
            cfg,
            stft,
            sample_carry: Vec::new(),
            band_ranges,
            frame_power: alloc::vec![0.0_f32; n_bins],
            has_prev: false,
            prev_energy: [0.0_f32; HAITSMA_N_BANDS],
            next_frame_idx: 1,
            pending: Vec::new(),
        })
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &HaitsmaConfig {
        &self.cfg
    }

    /// Reset all internal state. The stream behaves as if freshly
    /// constructed: no buffered audio, no pending frames, frame
    /// counter restarted. Call between independent input streams
    /// sharing one instance.
    pub fn reset(&mut self) {
        self.sample_carry.clear();
        self.has_prev = false;
        self.next_frame_idx = 1;
        self.pending.clear();
    }

    /// Core processing: advance the STFT, pack hashes, push into
    /// `self.pending`. Shared by `push()` and `push_with()`.
    fn process_push(&mut self, samples: &[f32]) {
        let samples = crate::pcm::truncate_push(samples, self.cfg.max_push_samples);
        crate::pcm::extend_sanitized(&mut self.sample_carry, samples);

        let mut off = 0usize;
        while self.sample_carry.len() - off >= HAITSMA_N_FFT {
            self.stft
                .process_frame_power(
                    &self.sample_carry[off..off + HAITSMA_N_FFT],
                    &mut self.frame_power,
                )
                .expect("frame_power is sized n_bins and frames are exactly n_fft");
            let mut e = [0.0_f32; HAITSMA_N_BANDS];
            for (b, &(start, end)) in self.band_ranges.iter().enumerate() {
                e[b] = self.frame_power[start..end].iter().sum();
            }

            if self.has_prev {
                let hash = pack_frame_bits(&e, &self.prev_energy);
                let abs_frame = self.next_frame_idx;
                let t_ms = (abs_frame as u64 * HAITSMA_HOP as u64 * 1000) / HAITSMA_SR as u64;
                self.pending.push((TimestampMs(t_ms), hash));
                self.next_frame_idx += 1;
            } else {
                self.has_prev = true;
            }
            self.prev_energy = e;
            off += HAITSMA_HOP;
        }

        if off > 0 {
            self.sample_carry.drain(0..off);
        }
    }
}

impl StreamingFingerprinter for StreamingHaitsma {
    type Frame = u32;

    fn required_sample_rate(&self) -> u32 {
        HAITSMA_SR
    }

    fn push(&mut self, samples: &[f32]) -> Result<Vec<(TimestampMs, Self::Frame)>> {
        self.process_push(samples);
        Ok(core::mem::take(&mut self.pending))
    }

    fn push_with<F>(&mut self, samples: &[f32], mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        self.process_push(samples);
        let mut n = 0usize;
        for (t, frame) in self.pending.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
    }

    fn flush(&mut self) -> Result<Vec<(TimestampMs, Self::Frame)>> {
        Ok(core::mem::take(&mut self.pending))
    }

    fn flush_with<F>(&mut self, mut callback: F) -> Result<usize>
    where
        F: FnMut(TimestampMs, &Self::Frame),
    {
        let mut n = 0usize;
        for (t, frame) in self.pending.drain(..) {
            callback(t, &frame);
            n += 1;
        }
        Ok(n)
    }

    fn latency_ms(&self) -> u32 {
        (HAITSMA_N_FFT as u32 * 1000) / HAITSMA_SR
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
            let t = n as f32 / 5_000.0;
            // Use frequencies in the 300-2000 Hz band so they hit our bands.
            let s = 0.5 * libm::sinf(2.0 * PI * 600.0 * t)
                + 0.3 * libm::sinf(2.0 * PI * 1200.0 * t)
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
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 10_000];

        match fp.extract(&samples, SampleRate::HZ_16000) {
            Err(AfpError::UnsupportedSampleRate(16_000)) => {}
            other => panic!("expected UnsupportedSampleRate, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_audio() {
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 5_000];

        match fp.extract(&samples, SampleRate::HZ_5000) {
            Err(AfpError::AudioTooShort {
                needed: 10_000,
                got: 5_000,
            }) => {}
            other => panic!("expected AudioTooShort, got {other:?}"),
        }
    }

    #[test]
    fn silence_gives_all_zero_frames() {
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 5_000 * 3];

        let fpr = fp.extract(&samples, SampleRate::HZ_5000).unwrap();
        assert_eq!(fpr.frames_per_sec, 78.125);
        assert!(!fpr.frames.is_empty());
        for &h in &fpr.frames {
            assert_eq!(h, 0, "silence should produce zero hash");
        }
    }

    #[test]
    fn synthetic_signal_produces_nonzero_hashes() {
        let mut fp = Haitsma::default();
        let samples = synthetic_audio(0xC0FFEE, 5_000 * 4);

        let fpr = fp.extract(&samples, SampleRate::HZ_5000).unwrap();
        assert!(
            (200..=400).contains(&fpr.frames.len()),
            "expected 200..=400 frames from a 4s tone @ 5 kHz, got {}",
            fpr.frames.len(),
        );
        let nonzero = fpr.frames.iter().filter(|&&h| h != 0).count();
        assert!(
            nonzero > fpr.frames.len() * 3 / 4,
            "expected > 75% of frames to have at least one bit set, got {nonzero}/{}",
            fpr.frames.len()
        );
        let distinct: alloc::collections::BTreeSet<u32> = fpr.frames.iter().copied().collect();
        assert!(
            distinct.len() > 200,
            "expected most frames to be distinct, got {} distinct of {}",
            distinct.len(),
            fpr.frames.len(),
        );
    }

    #[test]
    fn synthetic_signal_is_deterministic() {
        let samples = synthetic_audio(0xBEEF, 5_000 * 3);
        let mut a = Haitsma::default();
        let mut b = Haitsma::default();
        let fa = a.extract(&samples, SampleRate::HZ_5000).unwrap();
        let fb = b.extract(&samples, SampleRate::HZ_5000).unwrap();
        assert_eq!(fa.frames, fb.frames);
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 5_000 * 3);

        let mut fp1 = Haitsma::default();
        let f1 = fp1.extract(&samples, SampleRate::HZ_5000).unwrap();

        let mut fp2 = Haitsma::default();
        let f2 = fp2.extract(&samples, SampleRate::HZ_5000).unwrap();

        assert_eq!(f1.frames, f2.frames);
    }

    #[test]
    fn different_signals_diverge() {
        let a = synthetic_audio(0x1111, 5_000 * 3);
        let b = synthetic_audio(0x2222, 5_000 * 3);

        let mut fp = Haitsma::default();
        let fa = fp.extract(&a, SampleRate::HZ_5000).unwrap();
        let fb = fp.extract(&b, SampleRate::HZ_5000).unwrap();
        assert_ne!(fa.frames, fb.frames);
    }

    #[test]
    fn pack_frame_bits_msb_zero_band_layout() {
        // Construct energies so that band 0's delta is positive but bands
        // 1..31 are all zero.
        let mut curr = [0.0_f32; HAITSMA_N_BANDS];
        let prev = [0.0_f32; HAITSMA_N_BANDS];
        // E[curr][0] - E[curr][1] - 0 > 0 → set band 0.
        curr[0] = 1.0;

        let h = pack_frame_bits(&curr, &prev);
        // Band 0 → MSB (bit 31).
        assert_eq!(h, 1 << 31);
    }

    #[test]
    fn band_31_lives_in_the_lsb() {
        // Make band 31's delta positive: E[31] - E[32] > 0 (with all prev zero).
        let mut curr = [0.0_f32; HAITSMA_N_BANDS];
        let prev = [0.0_f32; HAITSMA_N_BANDS];
        curr[31] = 1.0;

        let h = pack_frame_bits(&curr, &prev);
        // Band 31 → bit 0.
        assert_eq!(h, 1);
    }

    #[test]
    fn streaming_latency_matches_n_fft() {
        let s = StreamingHaitsma::default();
        // 2048 samples / 5000 sr * 1000 = 409 ms (integer).
        assert_eq!(s.latency_ms(), 409);
    }

    #[test]
    fn band_lookup_table_covers_in_band_frequencies() {
        let cfg = HaitsmaConfig::default();
        let n_bins = HAITSMA_N_FFT / 2 + 1;
        let lookup = build_bin_to_band(&cfg, n_bins);
        assert_eq!(lookup.len(), n_bins);

        let bin_hz = HAITSMA_SR as f32 / HAITSMA_N_FFT as f32;
        // At least one bin in each band should be tagged.
        let mut hit_per_band = [false; HAITSMA_N_BANDS];
        for &b in &lookup {
            if b != NO_BAND {
                hit_per_band[b as usize] = true;
            }
        }
        for (i, &h) in hit_per_band.iter().enumerate() {
            assert!(h, "band {i} has no FFT bins");
        }

        // Bins outside [fmin, fmax) are NO_BAND.
        let bin_at_100hz = (100.0 / bin_hz) as usize;
        assert_eq!(
            lookup[bin_at_100hz], NO_BAND,
            "100 Hz should be below fmin=300"
        );
    }

    #[test]
    fn custom_band_range() {
        let cfg = HaitsmaConfig {
            fmin: 500.0,
            fmax: 1500.0,
            max_input_samples: None,
            max_push_samples: None,
        };
        let mut h = Haitsma::new(cfg.clone());
        let samples = synthetic_audio(0xC0FFEE, 5_000 * 3);

        let f = h.extract(&samples, SampleRate::HZ_5000).unwrap();
        // Should still produce frames; band edges differ but algorithm runs.
        assert!(!f.frames.is_empty());
    }

    #[test]
    fn invalid_band_range_returns_config_error() {
        let result = Haitsma::try_new(HaitsmaConfig {
            fmin: 1000.0,
            fmax: 1000.0,
            max_input_samples: None,
            max_push_samples: None,
        });
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected Config error"),
        };
        assert!(
            matches!(err, crate::AfpError::Config(ref msg) if msg.contains("fmax must exceed fmin")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn fmax_above_nyquist_returns_config_error() {
        let result = Haitsma::try_new(HaitsmaConfig {
            fmin: 300.0,
            fmax: 3_000.0,
            max_input_samples: None,
            max_push_samples: None,
        });
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected Config error"),
        };
        assert!(
            matches!(err, crate::AfpError::Config(ref msg) if msg.contains("Nyquist")),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn streaming_offline_equivalence() {
        let samples = synthetic_audio(0xBEEF, 5_000 * 5);

        let mut offline = Haitsma::default();
        let off = offline.extract(&samples, SampleRate::HZ_5000).unwrap();

        let mut streaming = StreamingHaitsma::default();
        let mut online: Vec<u32> = Vec::new();
        let mut cursor = 0;
        for n in chunk_sizes(0xCAFE, samples.len(), 3_000) {
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

        assert_eq!(off.frames, online, "streaming != offline frame sequence");
    }

    #[test]
    fn streaming_state_stays_bounded_under_long_input() {
        // 30 s of audio in 256-sample chunks. Haitsma's streaming
        // state is just `sample_carry` (drained per push) and
        // `prev_energy` (fixed array); `pending` is `mem::take`d on
        // every return.
        let secs = 30usize;
        let samples = synthetic_audio(13, HAITSMA_SR as usize * secs);
        let chunk = 256usize;

        let mut s = StreamingHaitsma::default();
        let mut peak_carry = 0usize;

        let mut start = 0usize;
        while start < samples.len() {
            let end = (start + chunk).min(samples.len());
            let _ = s.push(&samples[start..end]).unwrap();
            peak_carry = peak_carry.max(s.sample_carry.len());

            assert!(s.sample_carry.len() < HAITSMA_N_FFT);
            // `pending` is drained by `mem::take` at the end of every
            // push, so it must be empty between calls.
            assert_eq!(s.pending.len(), 0, "pending leaked between pushes");
            start = end;
        }

        // sample_carry should reach close to (but never equal) N_FFT
        // — that's the "almost-a-frame's-worth of leftover" steady state.
        assert!(peak_carry < HAITSMA_N_FFT, "peak_carry {peak_carry}");
        assert!(
            peak_carry >= HAITSMA_N_FFT - HAITSMA_HOP,
            "expected the carry to fill close to N_FFT under continuous input, got {peak_carry}",
        );

        let _ = s.flush().unwrap();
        assert_eq!(s.pending.len(), 0, "pending after flush");
    }

    // -----------------------------------------------------------------
    // Public API contract pins. See wang.rs for motivation.
    // -----------------------------------------------------------------

    #[test]
    fn public_api_name_and_config_match_documented_values() {
        let fp = Haitsma::default();
        assert_eq!(fp.name(), "haitsma-v1");
        assert_eq!(fp.required_sample_rate(), SampleRate::HZ_5000);
        assert_eq!(fp.min_samples(), 10_000);

        let s = StreamingHaitsma::default();
        assert_eq!(s.latency_ms(), 409);
    }

    // ── Backward-compat correctness tests ──

    #[test]
    fn streaming_reset_clears_all_state() {
        let mut s = StreamingHaitsma::default();
        let samples = synthetic_audio(0xFEED, 5_000 * 3);
        let before = s.push(&samples).unwrap();
        assert!(!before.is_empty(), "should produce frames");

        s.reset();
        assert!(s.push(&[]).unwrap().is_empty(), "reset should clear state");
        let after_reset = s.push(&samples).unwrap();
        assert!(!after_reset.is_empty());
        assert_eq!(
            before, after_reset,
            "reset+replay must produce identical frames"
        );
    }

    // -----------------------------------------------------------------
    // Constructor panic coverage.
    //
    // `Haitsma::new` (used by the `Haitsma` extractor) validates
    // `fmin > 0`, `fmax > fmin`, and `fmax < Nyquist`. The latter two
    // are already covered by `invalid_band_range_returns_config_error`
    // and `fmax_above_nyquist_returns_config_error`. The fmin>0 case:
    // -----------------------------------------------------------------

    #[test]
    fn haitsma_new_rejects_zero_fmin() {
        let cfg = HaitsmaConfig {
            fmin: 0.0,
            ..HaitsmaConfig::default()
        };
        let err = match Haitsma::try_new(cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected Config error"),
        };
        assert!(
            matches!(err, crate::AfpError::Config(ref msg) if msg.contains("fmin must be positive"))
        );
    }

    #[test]
    fn haitsma_new_rejects_negative_fmin() {
        let cfg = HaitsmaConfig {
            fmin: -10.0,
            ..HaitsmaConfig::default()
        };
        let err = match Haitsma::try_new(cfg) {
            Err(e) => e,
            Ok(_) => panic!("expected Config error"),
        };
        assert!(
            matches!(err, crate::AfpError::Config(ref msg) if msg.contains("fmin must be positive"))
        );
    }

    // ── Performance regression: zero-alloc push_with contract ──

    #[test]
    fn push_with_matches_push_output_count() {
        let mut a = StreamingHaitsma::default();
        let mut b = StreamingHaitsma::default();
        let samples = synthetic_audio(0xABCD, 5_000 * 4);

        let via_push = a.push(&samples).unwrap();
        let mut via_cb: Vec<(TimestampMs, u32)> = Vec::new();
        let n = b.push_with(&samples, |t, f| via_cb.push((t, *f))).unwrap();

        assert_eq!(n, via_push.len());
        assert_eq!(via_cb, via_push);
    }

    #[test]
    fn flush_with_matches_flush_output() {
        let mut a = StreamingHaitsma::default();
        let mut b = StreamingHaitsma::default();
        let samples = synthetic_audio(0xF00D, 5_000 * 4);
        let _ = a.push(&samples).unwrap();
        let _ = b.push(&samples).unwrap();

        let via_flush = a.flush().unwrap();
        let mut via_cb: Vec<(TimestampMs, u32)> = Vec::new();
        let n = b.flush_with(|t, f| via_cb.push((t, *f))).unwrap();

        assert_eq!(n, via_flush.len());
        assert_eq!(via_cb, via_flush);
    }

    // ── OOM protection: max_input_samples enforcement ──

    #[test]
    fn input_larger_than_max_is_rejected() {
        let cfg = HaitsmaConfig {
            max_input_samples: Some(1_000),
            ..HaitsmaConfig::default()
        };
        let mut fp = Haitsma::new(cfg);
        let samples = vec![0.0_f32; 2_000];

        let err = fp.extract(&samples, SampleRate::HZ_5000).unwrap_err();
        assert!(matches!(err, AfpError::InputTooLarge { .. }));
    }

    #[test]
    fn none_disables_max_input_check() {
        let cfg = HaitsmaConfig {
            max_input_samples: None,
            ..HaitsmaConfig::default()
        };
        let mut fp = Haitsma::new(cfg);
        let samples = vec![0.0_f32; 10_000];

        fp.extract(&samples, SampleRate::HZ_5000).unwrap();
    }

    // ── Progress callback tests ──

    #[test]
    fn extract_with_progress_is_called_and_monotonic() {
        let mut fp = Haitsma::default();
        let samples = synthetic_audio(0xCAFE, 5_000 * 5);

        let mut values: Vec<f32> = Vec::new();
        let result = fp.extract_with_progress(&samples, SampleRate::HZ_5000, |v| values.push(v));
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
        let samples = synthetic_audio(0xDEAD, 5_000 * 4);

        let mut fp1 = Haitsma::default();
        let result1 = fp1.extract(&samples, SampleRate::HZ_5000).unwrap();

        let mut fp2 = Haitsma::default();
        let result2 = fp2
            .extract_with_progress(&samples, SampleRate::HZ_5000, |_| {})
            .unwrap();

        assert_eq!(result1.frames, result2.frames);
        assert_eq!(result1.frames_per_sec, result2.frames_per_sec);
    }

    #[test]
    fn extract_with_progress_short_audio_still_reports_0_and_1() {
        let mut fp = Haitsma::default();
        let samples = synthetic_audio(0xFACE, 5_000 * 2);

        let mut values: Vec<f32> = Vec::new();
        let _ = fp.extract_with_progress(&samples, SampleRate::HZ_5000, |v| values.push(v));
        assert_eq!(values[0], 0.0);
        assert_eq!(*values.last().unwrap(), 1.0);
    }
}
