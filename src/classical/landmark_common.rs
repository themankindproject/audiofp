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
/// pooled dB log-magnitude buffer, and a pooled peak picker.
///
/// [`Wang`](super::Wang) and [`Panako`](super::Panako) embed one and
/// differ only in hash emission, so construction and the
/// STFT→dB→pick pipeline live here once.
pub(crate) struct FrontEnd {
    stft: ShortTimeFFT,
    picker: PeakPicker,
    /// Pooled log-magnitude buffer reused between calls.
    log_spec: alloc::vec::Vec<f32>,
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
        Self {
            stft,
            picker,
            log_spec: alloc::vec::Vec::new(),
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
        let (n_frames, n_bins) = self.stft.power_flat_into(samples, &mut self.log_spec);
        crate::dsp::power_to_db_wide(&mut self.log_spec, log_floor_power);
        let peaks = self
            .picker
            .pick(&self.log_spec, n_frames, n_bins, frames_per_sec);
        (n_frames, peaks)
    }
}
