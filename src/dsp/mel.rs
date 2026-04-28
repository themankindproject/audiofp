//! Mel filterbank: triangular filters spaced on the perceptual mel scale.
//!
//! [`MelFilterBank`] holds an `(n_mels, n_fft/2 + 1)` matrix of triangle
//! weights; [`MelFilterBank::log_mel`] dots a magnitude spectrum (squared
//! to power) into one log-mel frame.
//!
//! Filters are slaney-normalised — each triangle has unit area in the
//! linear-frequency domain — so log-mel output magnitudes are stable
//! across `n_mels` choices and match `librosa.feature.melspectrogram`'s
//! defaults.

use alloc::vec;
use alloc::vec::Vec;

use libm::{expf, log10f, logf, powf};

/// Selects how hertz are mapped to the mel scale.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MelScale {
    /// HTK formula `mel = 2595 · log10(1 + hz/700)`. Single closed-form
    /// expression, slightly different from Slaney above 1 kHz.
    Htk,

    /// Slaney's auditory-toolbox mapping: linear below 1 kHz, log above.
    /// This is `librosa`'s default and the right choice for most music
    /// applications.
    Slaney,
}

const SLANEY_F_SP: f32 = 200.0 / 3.0;
const SLANEY_MIN_LOG_HZ: f32 = 1000.0;
const SLANEY_LOGSTEP_DENOM: f32 = 27.0;

impl MelScale {
    fn hz_to_mel(self, hz: f32) -> f32 {
        match self {
            MelScale::Htk => 2595.0 * log10f(1.0 + hz / 700.0),
            MelScale::Slaney => {
                let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
                let logstep = logf(6.4) / SLANEY_LOGSTEP_DENOM;
                if hz < SLANEY_MIN_LOG_HZ {
                    hz / SLANEY_F_SP
                } else {
                    min_log_mel + logf(hz / SLANEY_MIN_LOG_HZ) / logstep
                }
            }
        }
    }

    fn mel_to_hz(self, mel: f32) -> f32 {
        match self {
            MelScale::Htk => 700.0 * (powf(10.0, mel / 2595.0) - 1.0),
            MelScale::Slaney => {
                let min_log_mel = SLANEY_MIN_LOG_HZ / SLANEY_F_SP;
                let logstep = logf(6.4) / SLANEY_LOGSTEP_DENOM;
                if mel < min_log_mel {
                    SLANEY_F_SP * mel
                } else {
                    SLANEY_MIN_LOG_HZ * expf(logstep * (mel - min_log_mel))
                }
            }
        }
    }
}

/// A precomputed triangular mel filterbank.
///
/// # Example
///
/// ```
/// use audiofp::dsp::mel::{MelFilterBank, MelScale};
///
/// // 128 mels covering 0–11025 Hz at sr=22050, n_fft=2048.
/// let fb = MelFilterBank::new(128, 2048, 22_050, 0.0, 11_025.0, MelScale::Slaney);
/// assert_eq!(fb.n_mels, 128);
/// assert_eq!(fb.n_bins(), 1025);
/// ```
#[derive(Clone, Debug)]
pub struct MelFilterBank {
    /// Number of mel bands (rows of the matrix).
    pub n_mels: usize,
    /// FFT length the upstream STFT uses; bin count is `n_fft / 2 + 1`.
    pub n_fft: usize,
    /// Sample rate of the audio fed to the upstream STFT.
    pub sr: u32,
    /// Lowest frequency (Hz) covered by the filterbank.
    pub fmin: f32,
    /// Highest frequency (Hz) covered by the filterbank.
    pub fmax: f32,
    /// Mel scale convention used to lay out filter centres.
    pub scale: MelScale,

    /// Row-major `(n_mels, n_fft/2 + 1)` weight matrix.
    matrix: Vec<f32>,
}

impl MelFilterBank {
    /// Build a filterbank.
    ///
    /// # Panics
    ///
    /// Panics if `n_mels == 0`, `n_fft < 2`, `n_fft` is not even, or
    /// `fmin >= fmax`.
    #[must_use]
    pub fn new(
        n_mels: usize,
        n_fft: usize,
        sr: u32,
        fmin: f32,
        fmax: f32,
        scale: MelScale,
    ) -> Self {
        assert!(n_mels > 0, "n_mels must be > 0");
        assert!(n_fft >= 2 && n_fft % 2 == 0, "n_fft must be even and >= 2");
        assert!(fmin < fmax, "fmin must be strictly less than fmax");

        let n_bins = n_fft / 2 + 1;
        let mut matrix = vec![0.0_f32; n_mels * n_bins];

        // Mel-spaced centre points, including the left and right "skirts".
        let mel_min = scale.hz_to_mel(fmin);
        let mel_max = scale.hz_to_mel(fmax);
        let n_points = n_mels + 2;
        let mut hz_points = Vec::with_capacity(n_points);
        for k in 0..n_points {
            let mel = mel_min + (mel_max - mel_min) * k as f32 / (n_points - 1) as f32;
            hz_points.push(scale.mel_to_hz(mel));
        }

        // FFT bin frequencies in Hz: bin b corresponds to b * sr / n_fft.
        let bin_hz = sr as f32 / n_fft as f32;

        for k in 0..n_mels {
            let left = hz_points[k];
            let centre = hz_points[k + 1];
            let right = hz_points[k + 2];
            // Slaney normalisation: unit area in linear frequency.
            let norm = 2.0 / (right - left).max(1e-10);

            let row = &mut matrix[k * n_bins..(k + 1) * n_bins];
            for (b, w) in row.iter_mut().enumerate() {
                let f = b as f32 * bin_hz;
                *w = if f <= left || f >= right {
                    0.0
                } else if f <= centre {
                    norm * (f - left) / (centre - left).max(1e-10)
                } else {
                    norm * (right - f) / (right - centre).max(1e-10)
                };
            }
        }

        Self {
            n_mels,
            n_fft,
            sr,
            fmin,
            fmax,
            scale,
            matrix,
        }
    }

    /// Number of FFT bins each filter spans (`n_fft / 2 + 1`).
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Borrow the row-major weight matrix.
    #[must_use]
    pub fn matrix(&self) -> &[f32] {
        &self.matrix
    }

    /// Compute one log-mel frame from a magnitude spectrum.
    ///
    /// Computes `log10(M · |X|² + 1e-10)` per librosa: the magnitude is
    /// squared to power before the matrix-vector product, and a small
    /// floor avoids `log10(0)`.
    ///
    /// # Panics
    ///
    /// Panics if `magnitude.len() != n_bins()` or `out.len() != n_mels`.
    pub fn log_mel(&self, magnitude: &[f32], out: &mut [f32]) {
        assert_eq!(
            magnitude.len(),
            self.n_bins(),
            "magnitude length must equal n_bins"
        );
        assert_eq!(out.len(), self.n_mels, "out length must equal n_mels");

        let n_bins = self.n_bins();
        for (k, slot) in out.iter_mut().enumerate() {
            let row = &self.matrix[k * n_bins..(k + 1) * n_bins];
            let mut acc = 0.0_f32;
            for (w, m) in row.iter().zip(magnitude.iter()) {
                acc += w * (m * m);
            }
            *slot = log10f(acc + 1e-10);
        }
    }

    /// Compute one log-mel frame from a **power** spectrum
    /// (`re² + im²` per bin, e.g. one row of
    /// [`ShortTimeFFT::power_flat`]).
    ///
    /// Equivalent to [`log_mel`] but skips the per-bin square — feed the
    /// output of `power_flat` / `process_frame_power` directly to avoid
    /// doing the work twice.
    ///
    /// [`log_mel`]: MelFilterBank::log_mel
    /// [`ShortTimeFFT::power_flat`]: crate::dsp::stft::ShortTimeFFT::power_flat
    ///
    /// # Panics
    ///
    /// Panics if `power.len() != n_bins()` or `out.len() != n_mels`.
    pub fn log_mel_from_power(&self, power: &[f32], out: &mut [f32]) {
        assert_eq!(power.len(), self.n_bins(), "power length must equal n_bins");
        assert_eq!(out.len(), self.n_mels, "out length must equal n_mels");

        let n_bins = self.n_bins();
        for (k, slot) in out.iter_mut().enumerate() {
            let row = &self.matrix[k * n_bins..(k + 1) * n_bins];
            let mut acc = 0.0_f32;
            for (w, p) in row.iter().zip(power.iter()) {
                acc += w * p;
            }
            *slot = log10f(acc + 1e-10);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn htk_round_trip() {
        for &hz in &[0.0_f32, 100.0, 440.0, 1_000.0, 5_000.0, 11_025.0] {
            let m = MelScale::Htk.hz_to_mel(hz);
            assert_relative_eq!(MelScale::Htk.mel_to_hz(m), hz, max_relative = 1e-5);
        }
    }

    #[test]
    fn slaney_round_trip() {
        for &hz in &[
            0.0_f32, 100.0, 440.0, 999.0, 1_000.0, 1_001.0, 5_000.0, 11_025.0,
        ] {
            let m = MelScale::Slaney.hz_to_mel(hz);
            assert_relative_eq!(MelScale::Slaney.mel_to_hz(m), hz, max_relative = 1e-4);
        }
    }

    #[test]
    fn matrix_dimensions() {
        let fb = MelFilterBank::new(64, 1024, 16_000, 0.0, 8_000.0, MelScale::Htk);
        assert_eq!(fb.n_bins(), 513);
        assert_eq!(fb.matrix().len(), 64 * 513);
    }

    #[test]
    fn each_filter_has_a_peak_in_band() {
        let fb = MelFilterBank::new(40, 2048, 22_050, 0.0, 11_025.0, MelScale::Slaney);
        let n_bins = fb.n_bins();
        for k in 0..fb.n_mels {
            let row = &fb.matrix[k * n_bins..(k + 1) * n_bins];
            let max = row.iter().cloned().fold(0.0_f32, f32::max);
            assert!(max > 0.0, "filter {k} is all-zero");
        }
    }

    #[test]
    fn log_mel_floor_at_silence() {
        let fb = MelFilterBank::new(16, 512, 16_000, 0.0, 8_000.0, MelScale::Htk);
        let zeros = vec![0.0_f32; fb.n_bins()];
        let mut out = vec![0.0_f32; fb.n_mels];
        fb.log_mel(&zeros, &mut out);
        // log10(1e-10) = -10.0 exactly.
        for v in out {
            assert_relative_eq!(v, -10.0, max_relative = 1e-5);
        }
    }

    #[test]
    fn htk_and_slaney_diverge_above_1khz() {
        // Below 1 kHz the two scales should agree to within ~5 mel.
        // Above 1 kHz Slaney is logarithmic with a different slope, so the
        // converted mel values diverge.
        let lo = 500.0_f32;
        let hi = 4_000.0_f32;
        let m_htk_lo = MelScale::Htk.hz_to_mel(lo);
        let m_sla_lo = MelScale::Slaney.hz_to_mel(lo);
        let m_htk_hi = MelScale::Htk.hz_to_mel(hi);
        let m_sla_hi = MelScale::Slaney.hz_to_mel(hi);

        let diff_lo = (m_htk_lo - m_sla_lo).abs();
        let diff_hi = (m_htk_hi - m_sla_hi).abs();
        assert!(
            diff_hi > diff_lo,
            "expected divergence to grow above 1 kHz: lo={diff_lo} hi={diff_hi}",
        );
    }

    #[test]
    fn matrix_rows_are_non_negative() {
        let fb = MelFilterBank::new(64, 2048, 22_050, 0.0, 11_025.0, MelScale::Slaney);
        for &w in fb.matrix() {
            assert!(w >= 0.0, "negative weight in mel matrix: {w}");
        }
    }

    #[test]
    fn log_mel_from_power_matches_log_mel_on_squared_input() {
        let fb = MelFilterBank::new(32, 1024, 16_000, 0.0, 8_000.0, MelScale::Slaney);
        let n_bins = fb.n_bins();

        // Synthetic spiky magnitude spectrum.
        let mag: Vec<f32> = (0..n_bins)
            .map(|b| ((b as f32 * 0.073).sin().abs() + 0.001) * (1 + b % 7) as f32)
            .collect();
        let pow: Vec<f32> = mag.iter().map(|m| m * m).collect();

        let mut out_mag = vec![0.0_f32; fb.n_mels];
        let mut out_pow = vec![0.0_f32; fb.n_mels];
        fb.log_mel(&mag, &mut out_mag);
        fb.log_mel_from_power(&pow, &mut out_pow);

        for (a, b) in out_mag.iter().zip(out_pow.iter()) {
            assert_relative_eq!(*a, *b, max_relative = 1e-6);
        }
    }

    #[test]
    fn log_mel_picks_up_dirac_in_band() {
        let fb = MelFilterBank::new(40, 2048, 22_050, 0.0, 11_025.0, MelScale::Slaney);
        // Dirac at bin 200 ≈ 200 * 22050/2048 ≈ 2154 Hz.
        let mut mag = vec![0.0_f32; fb.n_bins()];
        mag[200] = 1.0;
        let mut out = vec![0.0_f32; fb.n_mels];
        fb.log_mel(&mag, &mut out);

        // Some band must respond above the silence floor.
        let max = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > -9.0, "no band responded: max={max}");
    }
}
