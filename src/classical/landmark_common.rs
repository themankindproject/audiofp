//! Shared offline-extract helpers for the classical fingerprinters.
//!
//! [`Wang`](super::Wang) and [`Panako`](super::Panako) run the same
//! front-end (8 kHz, 1024-pt Hann STFT, dB peak picking) and differ only
//! in hash emission, so the input-validation preamble and the STFT-phase
//! progress reporting live here once instead of drifting in two copies.

use alloc::vec::Vec;

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, Result, SampleRate};

/// STFT window length of the landmark front-end.
pub(crate) const N_FFT: usize = 1024;
/// STFT hop of the landmark front-end.
pub(crate) const HOP: usize = 128;
/// Sample rate the landmark fingerprinters require.
pub(crate) const SR: u32 = 8_000;
/// Shortest accepted input: two seconds.
pub(crate) const MIN_SAMPLES: usize = SR as usize * 2;
/// STFT frame rate (62.5 fps).
pub(crate) const FRAMES_PER_SEC: f32 = SR as f32 / HOP as f32;
/// Half-width (frames and bins) of the peak-picking neighbourhood.
pub(crate) const PEAK_NEIGHBOURHOOD: usize = 15;
const LOG_FLOOR: f32 = 1e-6;
/// Squared form of the magnitude floor — fed to `log10(power)` instead of
/// `log10(magnitude)`, which lets us skip the per-bin `sqrt` in STFT.
/// Equivalent to `LOG_FLOOR.powi(2)`.
pub(crate) const LOG_FLOOR_POWER: f32 = LOG_FLOOR * LOG_FLOOR;
/// Progress callback reporting interval (62.5 fps): every 32 frames
/// ≈ 500 ms of audio.
pub(crate) const PROGRESS_INTERVAL: usize = 32;

/// Frames an anchor must have *after* it before all of its targets are
/// observed. Used only for `latency_ms` — emission timing in the
/// incremental implementation is driven by anchor finalisation.
pub(crate) fn lookahead_frames(target_zone_t: u16) -> u32 {
    target_zone_t as u32 + PEAK_NEIGHBOURHOOD as u32 + FRAMES_PER_SEC.ceil() as u32
}

/// Build the shared pipeline's [`PeakCfg`](super::stream::PeakCfg) from a
/// [`WangConfig`](super::WangConfig) / [`PanakoConfig`](super::PanakoConfig)
/// (same-named fields, distinct types).
macro_rules! peak_cfg {
    ($cfg:expr) => {
        $crate::classical::stream::PeakCfg {
            min_anchor_mag_db: $cfg.min_anchor_mag_db,
            target_zone_t: $cfg.target_zone_t as i32,
            target_zone_f: $cfg.target_zone_f as i32,
            fan_out: $cfg.fan_out as usize,
            peaks_per_sec: $cfg.peaks_per_sec as usize,
            max_pending_anchors: $cfg.max_pending_anchors,
            max_push_samples: $cfg.max_push_samples,
        }
    };
}
pub(crate) use peak_cfg;

/// Validate offline-extract inputs: finite samples, input-size cap,
/// expected sample rate, minimum length — in that order.
///
/// Error precedence is part of the contract (pinned by each
/// extractor's `rejects_*` tests): size cap before rate before length.
pub(crate) fn check_extract_preamble(
    samples: &[f32],
    rate: SampleRate,
    expected_sr: u32,
    min_samples: usize,
    max_input: Option<usize>,
) -> Result<()> {
    crate::pcm::reject_non_finite(samples)?;
    if let Some(limit) = max_input
        && samples.len() > limit
    {
        return Err(AfpError::InputTooLarge {
            limit,
            provided: samples.len(),
        });
    }
    if rate.hz() != expected_sr {
        return Err(AfpError::UnsupportedSampleRate(rate.hz()));
    }
    if samples.len() < min_samples {
        return Err(AfpError::AudioTooShort {
            needed: min_samples,
            got: samples.len(),
        });
    }
    Ok(())
}

/// Shared offline front-end for the landmark fingerprinters: Hann STFT,
/// per-frame dB conversion, and a pooled row-streaming peak picker.
///
/// [`Wang`](super::Wang) and [`Panako`](super::Panako) embed one and
/// differ only in hash emission, so construction and the
/// STFT→dB→pick pipeline live here once.
///
/// Frames are transformed, converted, and fed to the picker one at a
/// time, so the spectrogram is never materialised: scratch is
/// `O(neighborhood · n_bins)` (≈ 190 KB) instead of three
/// `n_frames × n_bins` buffers (≈ 39 MB per minute of audio). The peaks
/// are bit-identical to picking over the full dB spectrogram — see
/// [`crate::dsp::simd::db_into_split`] for the one rounding detail that
/// has to be reproduced.
pub(crate) struct FrontEnd {
    stft: ShortTimeFFT,
    picker: PeakPicker,
    /// Pooled single-frame power / dB row.
    frame: Vec<f32>,
}

impl FrontEnd {
    pub(crate) fn new(
        n_fft: usize,
        hop: usize,
        neighborhood: usize,
        min_anchor_mag_db: f32,
        peaks_per_sec: usize,
    ) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft,
            hop,
            window: WindowKind::Hann,
            // No reflect-padding: hashes are most stable when the first
            // frame starts at sample 0 of the input buffer.
            center: false,
        });
        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: neighborhood,
            neighborhood_f: neighborhood,
            min_magnitude_db: min_anchor_mag_db,
            min_magnitude_linear: None,
            target_per_sec: peaks_per_sec,
        });
        let frame = alloc::vec![0.0_f32; stft.n_bins()];
        Self {
            stft,
            picker,
            frame,
        }
    }

    /// Offline-extract sequence shared by `Wang` and `Panako`: validate the
    /// input, pick peaks while reporting progress, then build and sort the
    /// hashes.
    ///
    /// `build_hashes` turns the `(t, f)`-sorted peaks into hashes and
    /// `sort_key` fixes their deterministic output order (round-trip and
    /// golden tests). Progress is reported as `0.0`, the front-end's own
    /// frame-by-frame values scaled to `0.7`, `0.80`, `0.90`, then `1.0`; an
    /// input shorter than one frame reports `0.0` then `1.0` and yields no
    /// hashes.
    pub(crate) fn extract_hashes<H, K: Ord>(
        &mut self,
        samples: &[f32],
        rate: SampleRate,
        max_input: Option<usize>,
        mut progress: impl FnMut(f32),
        build_hashes: impl FnOnce(&[Peak]) -> Result<Vec<H>>,
        sort_key: impl FnMut(&H) -> K,
    ) -> Result<Vec<H>> {
        check_extract_preamble(samples, rate, SR, MIN_SAMPLES, max_input)?;

        progress(0.0);

        // Compute power (|X|2) directly from the FFT — skips a per-bin
        // sqrt that the dB conversion would immediately undo.
        // 20 · log10(sqrt(p)) ≡ 10 · log10(p).
        // The STFT → dB → peak-pick phase is ~70 % of the work; it reports
        // its own progress as frames are processed.
        let (n_frames, peaks) = self.pick_peaks(
            samples,
            FRAMES_PER_SEC,
            LOG_FLOOR_POWER,
            0.7,
            PROGRESS_INTERVAL,
            &mut progress,
        );
        if n_frames == 0 {
            progress(1.0);
            return Ok(Vec::new());
        }

        progress(0.80);
        progress(0.90);

        let mut hashes = build_hashes(&peaks)?;
        // Stable, deterministic ordering for round-trip and golden tests.
        hashes.sort_unstable_by_key(sort_key);

        progress(1.0);

        Ok(hashes)
    }

    /// Run STFT → dB → peak pick. Picker config is fixed at construction
    /// (the extractor config is immutable after `new`), so this just
    /// reuses the pooled scratch on every call.
    ///
    /// This phase is `weight` of the extractor's work: every `interval`
    /// frames it reports `weight · done / total`, then `weight` once the
    /// peaks are picked — as the frames are processed, not after (the
    /// values used to be replayed once the whole phase had finished).
    /// Nothing is reported for an input shorter than one frame.
    fn pick_peaks(
        &mut self,
        samples: &[f32],
        frames_per_sec: f32,
        log_floor_power: f32,
        weight: f32,
        interval: usize,
        progress: &mut impl FnMut(f32),
    ) -> (usize, Vec<Peak>) {
        let n_frames = self.stft.n_frames(samples.len());
        let n_bins = self.stft.n_bins();
        if n_frames == 0 {
            return (0, Vec::new());
        }
        let n_fft = self.stft.config().n_fft;
        let hop = self.stft.config().hop;
        // The dB conversion used to run over the whole flat spectrogram,
        // whose last `len % 8` cells took the scalar path. Keep exactly
        // that split so the dB values — and therefore the peaks — are
        // bit-identical.
        let total = n_frames * n_bins;
        let simd_end = total - total % 8;
        self.picker.begin(n_frames, n_bins, frames_per_sec);
        for f in 0..n_frames {
            let start = f * hop;
            self.stft
                .process_frame_power(&samples[start..start + n_fft], &mut self.frame)
                .expect("frame is sized n_bins and every frame is exactly n_fft");
            let scalar_from = simd_end.saturating_sub(f * n_bins);
            crate::dsp::simd::db_into_split(
                &mut self.frame,
                log_floor_power,
                crate::dsp::DB_LOG2_FACTOR,
                scalar_from,
            );
            self.picker.push_row(&self.frame);
            let done = f + 1;
            if done % interval == 0 && done < n_frames {
                progress(weight * (done as f32 / n_frames as f32));
            }
        }
        let peaks = self.picker.finish();
        progress(weight);
        (n_frames, peaks)
    }
}
