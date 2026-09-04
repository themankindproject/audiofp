//! Shared offline-extract helpers for the landmark fingerprinters.
//!
//! [`Wang`](super::Wang) and [`Panako`](super::Panako) run the same
//! front-end (8 kHz, 1024-pt Hann STFT, dB peak picking) and differ only
//! in hash emission, so the input-validation preamble and the STFT-phase
//! progress reporting live here once instead of drifting in two copies.

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
