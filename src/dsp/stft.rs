//! Short-Time Fourier Transform.
//!
//! [`ShortTimeFFT`] holds the FFT plan, the window, and reusable scratch
//! buffers; it can be invoked many times for buffers of arbitrary length
//! without allocating again.
//!
//! When [`StftConfig::center`] is `true` (the default), the input is
//! reflect-padded by `n_fft / 2` samples on each side before framing —
//! matching the behaviour of `librosa.stft(..., center=True)`.

use alloc::sync::Arc;
use alloc::vec::Vec;

use libm::sqrtf;
use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::dsp::windows::{WindowKind, make_window};

/// Parameters controlling an [`ShortTimeFFT`] instance.
#[derive(Clone, Debug)]
pub struct StftConfig {
    /// Length of each FFT in samples. Must be a non-zero power of two.
    pub n_fft: usize,
    /// Step between successive frames in samples. `0 < hop ≤ n_fft`.
    pub hop: usize,
    /// Window function applied to each frame before transformation.
    pub window: WindowKind,
    /// When `true`, reflect-pad the input so frame `i` is centred at
    /// sample `i * hop` (librosa default). When `false`, frame `i`
    /// starts at sample `i * hop`.
    pub center: bool,
}

impl StftConfig {
    /// Build a config with `hop = n_fft / 4`, Hann window, centred framing.
    ///
    /// # Example
    ///
    /// ```
    /// use afp::dsp::stft::StftConfig;
    /// let cfg = StftConfig::new(2048);
    /// assert_eq!(cfg.n_fft, 2048);
    /// assert_eq!(cfg.hop, 512);
    /// assert!(cfg.center);
    /// ```
    #[must_use]
    pub fn new(n_fft: usize) -> Self {
        Self {
            n_fft,
            hop: n_fft / 4,
            window: WindowKind::Hann,
            center: true,
        }
    }
}

/// Pre-planned short-time Fourier transform.
///
/// Construct once with [`ShortTimeFFT::new`], then call [`magnitude`] for a
/// whole buffer or [`process_frame`] for streaming use. Both methods reuse
/// internal scratch — no per-call allocation beyond the output container
/// in [`magnitude`].
///
/// [`magnitude`]: ShortTimeFFT::magnitude
/// [`process_frame`]: ShortTimeFFT::process_frame
///
/// # Example
///
/// ```
/// use afp::dsp::stft::{ShortTimeFFT, StftConfig};
///
/// let mut stft = ShortTimeFFT::new(StftConfig::new(1024));
/// let samples = vec![0.0_f32; 16_000];
/// let spec = stft.magnitude(&samples);
/// // (n_frames, n_bins) shape; default config gives n_bins = 1024/2 + 1.
/// assert_eq!(spec[0].len(), 513);
/// ```
pub struct ShortTimeFFT {
    cfg: StftConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<Complex<f32>>,
}

impl ShortTimeFFT {
    /// Plan an STFT.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.n_fft` is zero or not a power of two, or if
    /// `cfg.hop` is zero or larger than `cfg.n_fft`.
    #[must_use]
    pub fn new(cfg: StftConfig) -> Self {
        assert!(
            cfg.n_fft > 0 && cfg.n_fft.is_power_of_two(),
            "n_fft must be a non-zero power of two, got {}",
            cfg.n_fft
        );
        assert!(
            cfg.hop > 0 && cfg.hop <= cfg.n_fft,
            "hop must be in (0, n_fft], got hop={} n_fft={}",
            cfg.hop,
            cfg.n_fft
        );

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        let window = make_window(cfg.window, cfg.n_fft);
        let scratch_in = fft.make_input_vec();
        let scratch_out = fft.make_output_vec();

        Self {
            cfg,
            fft,
            window,
            scratch_in,
            scratch_out,
        }
    }

    /// Borrow the configuration this instance was built with.
    #[must_use]
    pub fn config(&self) -> &StftConfig {
        &self.cfg
    }

    /// Number of frequency bins emitted per frame: `n_fft / 2 + 1`.
    #[must_use]
    pub const fn n_bins(&self) -> usize {
        self.cfg.n_fft / 2 + 1
    }

    /// Number of frames [`magnitude`] would emit for an input of
    /// `n_samples` samples.
    ///
    /// [`magnitude`]: ShortTimeFFT::magnitude
    #[must_use]
    pub const fn n_frames(&self, n_samples: usize) -> usize {
        if self.cfg.center {
            1 + n_samples / self.cfg.hop
        } else if n_samples < self.cfg.n_fft {
            0
        } else {
            1 + (n_samples - self.cfg.n_fft) / self.cfg.hop
        }
    }

    /// Compute the magnitude spectrogram of `samples`.
    ///
    /// Result shape is `(n_frames, n_bins)` with `n_bins = n_fft/2 + 1`.
    /// Returns an empty `Vec` for empty input.
    #[must_use]
    pub fn magnitude(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        if samples.is_empty() {
            return Vec::new();
        }

        let n_fft = self.cfg.n_fft;
        let hop = self.cfg.hop;
        let n_frames = self.n_frames(samples.len());
        let n_bins = self.n_bins();

        let center_off = if self.cfg.center {
            (n_fft / 2) as isize
        } else {
            0
        };

        let mut out = Vec::with_capacity(n_frames);

        for f in 0..n_frames {
            let start = (f * hop) as isize - center_off;
            self.fill_windowed(samples, start);

            self.fft
                .process(&mut self.scratch_in, &mut self.scratch_out)
                .expect("FFT process: input/output length mismatch");

            let mut mag = Vec::with_capacity(n_bins);
            for c in &self.scratch_out {
                mag.push(sqrtf(c.norm_sqr()));
            }
            out.push(mag);
        }

        out
    }

    /// Streaming variant: window one `n_fft`-sized frame and emit its
    /// magnitude spectrum into `out` (`n_bins` long).
    ///
    /// # Panics
    ///
    /// Panics if `frame.len() != n_fft` or `out.len() != n_bins`.
    pub fn process_frame(&mut self, frame: &[f32], out: &mut [f32]) {
        assert_eq!(frame.len(), self.cfg.n_fft, "frame length must equal n_fft");
        assert_eq!(out.len(), self.n_bins(), "out length must equal n_bins");

        for (i, (s, w)) in frame.iter().zip(self.window.iter()).enumerate() {
            self.scratch_in[i] = s * w;
        }

        self.fft
            .process(&mut self.scratch_in, &mut self.scratch_out)
            .expect("FFT process: input/output length mismatch");

        for (c, o) in self.scratch_out.iter().zip(out.iter_mut()) {
            *o = sqrtf(c.norm_sqr());
        }
    }

    /// Fill `scratch_in` with `samples[start..start+n_fft] * window`,
    /// reflecting indices that fall outside `samples` when the config
    /// uses centred framing.
    fn fill_windowed(&mut self, samples: &[f32], start: isize) {
        let n_fft = self.cfg.n_fft;
        let len = samples.len();

        for k in 0..n_fft {
            let idx = start + k as isize;
            let s = if (0..len as isize).contains(&idx) {
                samples[idx as usize]
            } else if self.cfg.center {
                samples[reflect(idx, len)]
            } else {
                0.0
            };
            self.scratch_in[k] = s * self.window[k];
        }
    }
}

/// Reflect `i` into `[0, len)` using the convention `numpy.pad(mode="reflect")`
/// uses: edges are not repeated. Pattern for `len = 5`: `…3 2 1 2 3 4 5 4 3…`.
fn reflect(i: isize, len: usize) -> usize {
    let n = len as isize;
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut j = i.rem_euclid(period);
    if j >= n {
        j = period - j;
    }
    j as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f32::consts::PI;

    #[test]
    fn reflect_matches_numpy() {
        // np.pad([0,1,2,3,4], 3, mode='reflect') == [3,2,1,0,1,2,3,4,3,2,1]
        let want = [3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1];
        for (i, w) in (-3..8).zip(want) {
            assert_eq!(reflect(i, 5), w, "i={i}");
        }
    }

    #[test]
    fn n_bins_and_frames() {
        let s = ShortTimeFFT::new(StftConfig::new(1024));
        assert_eq!(s.n_bins(), 513);
        // center=true, hop=256: 16000 / 256 + 1 = 63
        assert_eq!(s.n_frames(16_000), 63);
    }

    #[test]
    fn empty_input_produces_no_frames() {
        let mut s = ShortTimeFFT::new(StftConfig::new(1024));
        assert!(s.magnitude(&[]).is_empty());
    }

    #[test]
    fn dc_signal_concentrates_energy_in_bin_zero() {
        // For a DC input, the windowed frame is just the window, whose DFT
        // has support only on bins {0, 1, N-1} for Hann. Bin 1 carries half
        // the DC energy, but bins ≥ 2 are numerically zero.
        let mut s = ShortTimeFFT::new(StftConfig::new(1024));
        let samples = alloc::vec![1.0_f32; 4096];
        let spec = s.magnitude(&samples);
        let mid = spec.len() / 2;
        let f = &spec[mid];
        assert!(f[0] > 0.0);
        for (k, &v) in f.iter().enumerate().skip(2) {
            assert!(
                f[0] > v * 1000.0,
                "bin {k} ({v}) not negligible vs DC ({})",
                f[0]
            );
        }
    }

    #[test]
    fn pure_sine_peaks_at_expected_bin() {
        let n_fft = 1024;
        let sr = 16_000.0_f32;
        let freq = 1000.0_f32;
        let mut s = ShortTimeFFT::new(StftConfig::new(n_fft));

        // 4096 samples of a 1 kHz tone at sr=16 kHz.
        let samples: alloc::vec::Vec<f32> = (0..4096)
            .map(|n| libm::sinf(2.0 * PI * freq * n as f32 / sr))
            .collect();
        let spec = s.magnitude(&samples);

        // Expected bin = freq / (sr / n_fft) = 1000 / (16000/1024) = 64.
        let expected_bin = (freq * n_fft as f32 / sr) as usize;
        let mid = spec.len() / 2;
        let f = &spec[mid];

        let (peak_bin, _) = f
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert_eq!(peak_bin, expected_bin);
    }

    #[test]
    fn process_frame_matches_magnitude() {
        let cfg = StftConfig {
            n_fft: 256,
            hop: 256,
            window: WindowKind::Hann,
            center: false,
        };
        let mut s = ShortTimeFFT::new(cfg.clone());

        let samples: alloc::vec::Vec<f32> = (0..256)
            .map(|n| libm::sinf(2.0 * PI * n as f32 / 32.0))
            .collect();

        let mut frame_out = alloc::vec![0.0_f32; s.n_bins()];
        s.process_frame(&samples, &mut frame_out);

        let mut s2 = ShortTimeFFT::new(cfg);
        let buf_out = s2.magnitude(&samples);

        assert_eq!(buf_out.len(), 1);
        for (a, b) in frame_out.iter().zip(buf_out[0].iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-5);
        }
    }
}
