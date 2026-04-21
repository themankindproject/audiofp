//! Sample-rate conversion.
//!
//! Two implementations:
//!
//! - [`linear`] — straight linear interpolation. Cheap but introduces audible
//!   aliasing on downsamples; only suitable for non-critical paths or as a
//!   sanity baseline.
//! - [`SincResampler`] — windowed-sinc convolution with a Kaiser window.
//!   Higher quality at the cost of `O(N · taps)` work per output sample.
//!
//! Both treat any out-of-range input as silence (zero-pad). Mono `f32` only;
//! callers downmix multi-channel inputs upstream.

use alloc::vec::Vec;

use libm::{sinf, sqrtf};

/// Linear resampler. Output length is `ceil(input.len() * to_sr / from_sr)`,
/// or `0` if the input is empty.
///
/// Pass-through when `from_sr == to_sr` (clones the input).
///
/// # Example
///
/// ```
/// use afp::dsp::resample::linear;
///
/// // Halve the rate: every other sample is kept.
/// let x = vec![0.0_f32, 1.0, 0.0, 1.0, 0.0, 1.0];
/// let y = linear(&x, 2, 1);
/// assert_eq!(y.len(), 3);
/// ```
#[must_use]
pub fn linear(input: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    assert!(from_sr > 0 && to_sr > 0, "sample rates must be non-zero");
    if input.is_empty() {
        return Vec::new();
    }
    if from_sr == to_sr {
        return input.to_vec();
    }

    let n_in = input.len();
    let n_out = ((n_in as u64 * to_sr as u64).div_ceil(from_sr as u64)) as usize;
    let ratio = from_sr as f64 / to_sr as f64;

    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let pos = n as f64 * ratio;
        let i = pos.floor() as usize;
        let frac = (pos - pos.floor()) as f32;
        let a = input[i.min(n_in - 1)];
        let b = input[(i + 1).min(n_in - 1)];
        out.push(a * (1.0 - frac) + b * frac);
    }
    out
}

/// Quality knobs for [`SincResampler`].
#[derive(Copy, Clone, Debug)]
pub struct SincQuality {
    /// Half the filter width in *input* samples. The filter spans
    /// `2 * half_taps` input samples around each output position.
    /// Typical: 16 (low-latency), 32 (good), 64 (excellent).
    pub half_taps: usize,
    /// Kaiser β parameter. Larger values trade transition-band sharpness
    /// for stopband attenuation. Typical: 8.6 (≈ -80 dB stopband), 12.0
    /// (≈ -120 dB).
    pub kaiser_beta: f32,
}

impl Default for SincQuality {
    fn default() -> Self {
        Self { half_taps: 32, kaiser_beta: 8.6 }
    }
}

/// Windowed-sinc Kaiser resampler.
///
/// Each output sample is computed by convolving the input with a sinc
/// kernel multiplied by a Kaiser window. The cutoff is automatically set
/// to `min(from_sr, to_sr) / 2` (in the input's sample-rate frame), which
/// suppresses aliasing on downsamples and limits image content on
/// upsamples.
///
/// # Example
///
/// ```
/// use afp::dsp::resample::SincResampler;
///
/// // 1.0 s of 1 kHz tone at 16 kHz → resample to 8 kHz.
/// let sr_in = 16_000u32;
/// let sr_out = 8_000u32;
/// let n = sr_in as usize;
/// let x: Vec<f32> = (0..n)
///     .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr_in as f32).sin())
///     .collect();
///
/// let r = SincResampler::new(sr_in, sr_out);
/// let y = r.process(&x);
/// // Output length is roughly half of input.
/// assert!((y.len() as i64 - sr_out as i64).abs() < 16);
/// ```
pub struct SincResampler {
    from_sr: u32,
    to_sr: u32,
    quality: SincQuality,
    /// `min(from_sr, to_sr) / from_sr / 2`: anti-aliasing cutoff,
    /// normalised to the input's Nyquist.
    cutoff: f32,
    /// Reciprocal of `modified_bessel_i0(beta)`, precomputed for the
    /// Kaiser window denominator.
    inv_i0_beta: f32,
    /// DC gain of the kernel; applied as a divisor to keep constant
    /// signals constant.
    dc_gain: f32,
}

impl SincResampler {
    /// Build a resampler with default quality (`half_taps=32`, `β=8.6`).
    #[must_use]
    pub fn new(from_sr: u32, to_sr: u32) -> Self {
        Self::with_quality(from_sr, to_sr, SincQuality::default())
    }

    /// Build a resampler with custom quality.
    ///
    /// # Panics
    ///
    /// Panics if `from_sr == 0`, `to_sr == 0`, or `quality.half_taps == 0`.
    #[must_use]
    pub fn with_quality(from_sr: u32, to_sr: u32, quality: SincQuality) -> Self {
        assert!(from_sr > 0 && to_sr > 0, "sample rates must be non-zero");
        assert!(quality.half_taps > 0, "half_taps must be > 0");

        let cutoff = from_sr.min(to_sr) as f32 / from_sr as f32 / 2.0;
        let inv_i0_beta = 1.0 / modified_bessel_i0(quality.kaiser_beta);

        // Compute DC gain by summing a unit-DC kernel at the centre.
        let mut dc_gain = 0.0_f32;
        let half = quality.half_taps as isize;
        for k in -half..=half {
            let x = k as f32;
            let w = kaiser_window_input(x, quality.half_taps as f32, quality.kaiser_beta, inv_i0_beta);
            dc_gain += sinc(x * 2.0 * cutoff) * w * (2.0 * cutoff);
        }
        if dc_gain.abs() < 1e-10 {
            dc_gain = 1.0;
        }

        Self { from_sr, to_sr, quality, cutoff, inv_i0_beta, dc_gain }
    }

    /// Borrow the quality knobs this resampler was built with.
    #[must_use]
    pub fn quality(&self) -> &SincQuality {
        &self.quality
    }

    /// Resample `input` into the configured target rate.
    ///
    /// Output length is `ceil(input.len() * to_sr / from_sr)`. Out-of-range
    /// neighbours are treated as zero (zero-pad boundary).
    #[must_use]
    pub fn process(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        if self.from_sr == self.to_sr {
            return input.to_vec();
        }

        let n_in = input.len();
        let n_out = ((n_in as u64 * self.to_sr as u64).div_ceil(self.from_sr as u64)) as usize;
        let ratio = self.from_sr as f64 / self.to_sr as f64;
        let half = self.quality.half_taps as isize;
        let half_f = self.quality.half_taps as f32;
        let two_cutoff = 2.0 * self.cutoff;

        let mut out = Vec::with_capacity(n_out);
        for n in 0..n_out {
            let pos = n as f64 * ratio;
            let i_centre = pos.floor() as isize;
            let frac = (pos - pos.floor()) as f32;

            let mut acc = 0.0_f32;
            for k in -half..=half {
                let idx = i_centre + k;
                if idx < 0 || (idx as usize) >= n_in {
                    continue;
                }
                let x = k as f32 - frac;
                let w = kaiser_window_input(x, half_f, self.quality.kaiser_beta, self.inv_i0_beta);
                let s = sinc(x * two_cutoff);
                acc += input[idx as usize] * s * w * two_cutoff;
            }
            out.push(acc / self.dc_gain);
        }
        out
    }
}

/// Normalised sinc: `sin(π·x) / (π·x)`, with `sinc(0) = 1`.
#[inline]
fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    let pi_x = core::f32::consts::PI * x;
    sinf(pi_x) / pi_x
}

/// Kaiser window evaluated at `x` measured in input samples from the centre,
/// with width ±`half`. Returns 0 outside the window.
#[inline]
fn kaiser_window_input(x: f32, half: f32, beta: f32, inv_i0_beta: f32) -> f32 {
    let t = x / half;
    if t.abs() > 1.0 {
        return 0.0;
    }
    let arg = beta * sqrtf((1.0 - t * t).max(0.0));
    modified_bessel_i0(arg) * inv_i0_beta
}

/// Modified Bessel function of the first kind, order 0, via series
/// expansion. Converges fast for `|x| ≤ 30` (well past audio Kaiser β).
#[inline]
fn modified_bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0_f32;
    let mut term = 1.0_f32;
    let half_x_sq = (x * x) / 4.0;
    for k in 1..=30 {
        let kf = k as f32;
        term *= half_x_sq / (kf * kf);
        sum += term;
        if term / sum < 1e-9 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;
    use core::f32::consts::PI;

    #[test]
    fn linear_passthrough_when_rates_match() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let y = linear(&x, 16_000, 16_000);
        assert_eq!(x, y);
    }

    #[test]
    fn linear_halves_length_on_2_to_1() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = linear(&x, 2, 1);
        assert_eq!(y.len(), 3);
        // Linear at t=0,2,4 → exact samples 1, 3, 5.
        assert_relative_eq!(y[0], 1.0, max_relative = 1e-5);
        assert_relative_eq!(y[1], 3.0, max_relative = 1e-5);
        assert_relative_eq!(y[2], 5.0, max_relative = 1e-5);
    }

    #[test]
    fn linear_doubles_length_on_1_to_2() {
        let x = vec![0.0, 2.0, 4.0];
        let y = linear(&x, 1, 2);
        assert_eq!(y.len(), 6);
        // Halfway between 0 and 2 → 1.
        assert_relative_eq!(y[1], 1.0, max_relative = 1e-5);
    }

    #[test]
    fn sinc_resampler_passthrough_when_rates_match() {
        let r = SincResampler::new(16_000, 16_000);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.05).collect();
        assert_eq!(r.process(&x), x);
    }

    #[test]
    fn sinc_resampler_preserves_dc() {
        let r = SincResampler::new(44_100, 16_000);
        let x = vec![1.0_f32; 4096];
        let y = r.process(&x);
        // Skip the boundary region where zero-padding pulls the DC down.
        let mid_lo = y.len() / 4;
        let mid_hi = 3 * y.len() / 4;
        for &s in &y[mid_lo..mid_hi] {
            assert_relative_eq!(s, 1.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn sinc_resampler_output_length_matches_ratio() {
        let r = SincResampler::new(48_000, 8_000);
        let x = vec![0.0_f32; 48_000];
        let y = r.process(&x);
        // 48k → 8k means 6× downsample.
        assert_eq!(y.len(), 8_000);
    }

    #[test]
    fn sinc_resampler_preserves_pure_tone() {
        // 1 kHz tone at 16 kHz, downsampled to 8 kHz. The tone is well below
        // 4 kHz Nyquist, so it should survive.
        let sr_in = 16_000u32;
        let sr_out = 8_000u32;
        let freq = 1000.0_f32;
        let n = (sr_in as usize) * 2;
        let x: Vec<f32> = (0..n)
            .map(|i| libm::sinf(2.0 * PI * freq * i as f32 / sr_in as f32))
            .collect();

        let r = SincResampler::new(sr_in, sr_out);
        let y = r.process(&x);

        // Compute peak frequency via crude DFT bin scan around 1000 Hz.
        // Use centre region to skip boundary roll-off.
        let lo = y.len() / 4;
        let hi = 3 * y.len() / 4;
        let seg = &y[lo..hi];
        let m = seg.len();

        // Probe a handful of frequencies and pick the strongest.
        let mut best = (0.0_f32, 0.0_f32);
        for f in (500..=1500).step_by(10) {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for (k, &s) in seg.iter().enumerate() {
                let theta = 2.0 * PI * f as f32 * k as f32 / sr_out as f32;
                re += s * libm::cosf(theta);
                im -= s * libm::sinf(theta);
            }
            let mag = sqrtf(re * re + im * im) / m as f32;
            if mag > best.1 {
                best = (f as f32, mag);
            }
        }
        assert!(
            (best.0 - 1000.0).abs() <= 20.0,
            "peak at {} Hz, expected near 1000 Hz",
            best.0
        );
    }

    #[test]
    fn empty_input_produces_empty_output() {
        assert!(linear(&[], 44_100, 16_000).is_empty());
        let r = SincResampler::new(44_100, 16_000);
        assert!(r.process(&[]).is_empty());
    }

    #[test]
    fn modified_bessel_i0_matches_known_values() {
        // I0(0) = 1, I0(1) ≈ 1.2660658, I0(5) ≈ 27.2398718
        assert_relative_eq!(modified_bessel_i0(0.0), 1.0, max_relative = 1e-6);
        assert_relative_eq!(modified_bessel_i0(1.0), 1.266_065_8, max_relative = 1e-5);
        assert_relative_eq!(modified_bessel_i0(5.0), 27.239_872, max_relative = 1e-5);
    }
}
