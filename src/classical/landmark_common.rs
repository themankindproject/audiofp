//! Shared offline-extract helpers for the landmark fingerprinters.
//!
//! [`Wang`](super::Wang) and [`Panako`](super::Panako) run the same
//! front-end (8 kHz, 1024-pt Hann STFT, dB peak picking) and differ only
//! in hash emission, so the input-validation preamble and the STFT-phase
//! progress reporting live here once instead of drifting in two copies.

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{AfpError, Result, SampleRate};

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

/// Report proportional progress through the bulk-STFT phase (which holds
/// `weight` of total work), then report `weight` itself on completion.
pub(crate) fn report_stft_progress(
    total_frames: usize,
    weight: f32,
    interval: usize,
    progress: &mut impl FnMut(f32),
) {
    let mut reported = 0usize;
    while reported + interval < total_frames {
        reported += interval;
        progress(weight * (reported as f32 / total_frames as f32));
    }
    progress(weight);
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
    frame: alloc::vec::Vec<f32>,
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

    /// Run STFT → dB → peak pick. Picker config is fixed at construction
    /// (the extractor config is immutable after `new`), so this just
    /// reuses the pooled scratch on every call.
    pub(crate) fn pick_peaks(
        &mut self,
        samples: &[f32],
        frames_per_sec: f32,
        log_floor_power: f32,
    ) -> (usize, alloc::vec::Vec<Peak>) {
        let n_frames = self.stft.n_frames(samples.len());
        let n_bins = self.stft.n_bins();
        if n_frames == 0 {
            return (0, alloc::vec::Vec::new());
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
        }
        (n_frames, self.picker.finish(frames_per_sec))
    }
}
