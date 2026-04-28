//! Log-mel front-end shared by [`super::NeuralEmbedder`] and
//! [`super::StreamingNeuralEmbedder`].
//!
//! `LogMelFrontend` owns a pre-planned [`ShortTimeFFT`] and
//! [`MelFilterBank`] and consumes one analysis window of PCM at a time
//! to drive a per-frame callback. The callback pattern lets the caller
//! decide whether to copy each frame into a contiguous buffer or write
//! it directly into a strided destination (e.g. the model input
//! tensor) — no intermediate `Vec` is required.
//!
//! `pub(crate)` for now: users who need raw log-mel can compose
//! [`MelFilterBank`] and [`ShortTimeFFT`] (both `pub`) themselves; the
//! callback contract isn't stable enough to expose yet.

use alloc::vec;
use alloc::vec::Vec;

use crate::dsp::mel::MelFilterBank;
use crate::dsp::stft::ShortTimeFFT;

/// Pre-planned STFT → log-mel pipeline.
///
/// Allocation budget: per call, **zero** beyond what the caller
/// supplies in the per-frame callback. Internal scratch (`power`,
/// `mel_per_frame`) is allocated once at construction and reused.
pub(crate) struct LogMelFrontend {
    stft: ShortTimeFFT,
    mel: MelFilterBank,
    n_fft: usize,
    hop: usize,
    n_mels: usize,
    n_frames: usize,
    window_samples: usize,
    /// Per-frame power spectrum scratch (`n_fft / 2 + 1` floats).
    power: Vec<f32>,
    /// Per-frame log-mel scratch (`n_mels` floats).
    mel_per_frame: Vec<f32>,
}

impl LogMelFrontend {
    pub(crate) fn new(stft: ShortTimeFFT, mel: MelFilterBank, window_samples: usize) -> Self {
        let n_fft = stft.config().n_fft;
        let hop = stft.config().hop;
        let n_mels = mel.n_mels;
        let n_bins = stft.n_bins();
        let n_frames = (window_samples - n_fft) / hop + 1;
        Self {
            stft,
            mel,
            n_fft,
            hop,
            n_mels,
            n_frames,
            window_samples,
            power: vec![0.0; n_bins],
            mel_per_frame: vec![0.0; n_mels],
        }
    }

    /// Number of STFT frames produced for one analysis window.
    #[allow(dead_code)] // exposed for tests + future callers
    pub(crate) fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of mel bands.
    pub(crate) fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Required analysis-window length (samples).
    #[allow(dead_code)] // exposed for tests + future callers
    pub(crate) fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Walk frames of `window`. For each frame, the callback receives
    /// `(frame_idx, &log_mel_row[n_mels])`. The slice borrows internal
    /// scratch and is overwritten on the next iteration — copy out if
    /// you need to keep it.
    ///
    /// # Panics
    ///
    /// Panics if `window.len() != self.window_samples`.
    #[inline]
    pub(crate) fn for_each_frame<F: FnMut(usize, &[f32])>(
        &mut self,
        window: &[f32],
        mut callback: F,
    ) {
        assert_eq!(
            window.len(),
            self.window_samples,
            "for_each_frame requires exactly window_samples"
        );

        for f in 0..self.n_frames {
            let frame = &window[f * self.hop..f * self.hop + self.n_fft];
            self.stft.process_frame_power(frame, &mut self.power);
            self.mel
                .log_mel_from_power(&self.power, &mut self.mel_per_frame);
            callback(f, &self.mel_per_frame);
        }
    }

    /// Convenience: fill `out` with `[n_frames, n_mels]` row-major
    /// log-mel data (frame-major). Used by tests and by callers that
    /// want a contiguous matrix instead of a strided write.
    ///
    /// # Panics
    ///
    /// Panics if `window.len() != self.window_samples` or
    /// `out.len() != n_frames * n_mels`.
    #[allow(dead_code)] // retained for tests and future direct consumers
    pub(crate) fn fill_frame_major(&mut self, window: &[f32], out: &mut [f32]) {
        assert_eq!(
            out.len(),
            self.n_frames * self.n_mels,
            "out length must equal n_frames * n_mels",
        );
        let n_mels = self.n_mels;
        self.for_each_frame(window, |f, row| {
            out[f * n_mels..(f + 1) * n_mels].copy_from_slice(row);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::mel::MelScale;
    use crate::dsp::stft::StftConfig;
    use crate::dsp::windows::WindowKind;

    /// Build a default-configured frontend matching `NeuralEmbedderConfig::new`.
    fn default_frontend() -> LogMelFrontend {
        let n_fft = 1024;
        let hop = 320;
        let n_mels = 128;
        let sr = 16_000u32;
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft,
            hop,
            window: WindowKind::Hann,
            center: false,
        });
        let mel = MelFilterBank::new(n_mels, n_fft, sr, 0.0, sr as f32 / 2.0, MelScale::Slaney);
        let window_samples = sr as usize; // 1 s
        LogMelFrontend::new(stft, mel, window_samples)
    }

    #[test]
    fn n_frames_matches_formula() {
        let f = default_frontend();
        // (16000 - 1024) / 320 + 1 = 47
        assert_eq!(f.n_frames(), (16_000 - 1024) / 320 + 1);
        assert_eq!(f.n_mels(), 128);
        assert_eq!(f.window_samples(), 16_000);
    }

    #[test]
    fn callback_receives_n_mels_long_rows() {
        let mut f = default_frontend();
        let window = vec![0.0_f32; f.window_samples()];
        let n_mels = f.n_mels();
        let mut got_calls = 0;
        f.for_each_frame(&window, |_, row| {
            assert_eq!(row.len(), n_mels);
            got_calls += 1;
        });
        assert_eq!(got_calls, f.n_frames());
    }

    #[test]
    fn silence_hits_log_floor() {
        let mut f = default_frontend();
        let window = vec![0.0_f32; f.window_samples()];
        let mut out = vec![0.0_f32; f.n_frames() * f.n_mels()];
        f.fill_frame_major(&window, &mut out);
        // log10(1e-10) = -10 exactly.
        for &v in &out {
            assert!((v + 10.0).abs() < 1e-4, "expected ~ -10, got {v}");
        }
    }

    #[test]
    fn pure_sine_produces_a_clear_peak() {
        // 1 kHz sine at 16 kHz for 1 s.
        let sr = 16_000u32;
        let n = sr as usize;
        let freq = 1000.0_f32;
        let window: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let mut f = default_frontend();
        let mut out = vec![0.0_f32; f.n_frames() * f.n_mels()];
        f.fill_frame_major(&window, &mut out);

        // Average each band over time.
        let n_mels = f.n_mels();
        let mut band_avg = vec![0.0_f32; n_mels];
        for fr in 0..f.n_frames() {
            for m in 0..n_mels {
                band_avg[m] += out[fr * n_mels + m];
            }
        }
        for v in &mut band_avg {
            *v /= f.n_frames() as f32;
        }

        let (peak_band, peak) = band_avg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        // Slaney mel(1 kHz) ≈ 15 of 128 bands at 0–8 kHz. Peak should be in
        // the lower-third of the band axis, well above the silence floor.
        assert!(peak_band < n_mels / 2, "peak at band {peak_band}");
        assert!(*peak > -5.0, "peak magnitude too low: {peak}");
    }

    #[test]
    fn output_is_deterministic_across_calls() {
        let mut f1 = default_frontend();
        let mut f2 = default_frontend();
        let window: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.0123).sin() * 0.5 + (i as f32 * 0.057).cos() * 0.25)
            .collect();

        let mut a = vec![0.0_f32; f1.n_frames() * f1.n_mels()];
        let mut b = vec![0.0_f32; f1.n_frames() * f1.n_mels()];
        f1.fill_frame_major(&window, &mut a);
        f2.fill_frame_major(&window, &mut b);
        assert_eq!(a, b);

        // And: same frontend, called twice, gives the same output.
        let mut c = vec![0.0_f32; f1.n_frames() * f1.n_mels()];
        f1.fill_frame_major(&window, &mut c);
        assert_eq!(a, c);
    }

    #[test]
    fn fill_frame_major_matches_for_each_frame_collected() {
        let mut f = default_frontend();
        let window: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin()).collect();

        let mut via_fill = vec![0.0_f32; f.n_frames() * f.n_mels()];
        f.fill_frame_major(&window, &mut via_fill);

        let n_mels = f.n_mels();
        let mut via_callback = vec![0.0_f32; f.n_frames() * n_mels];
        f.for_each_frame(&window, |fr, row| {
            via_callback[fr * n_mels..(fr + 1) * n_mels].copy_from_slice(row);
        });
        assert_eq!(via_fill, via_callback);
    }
}
