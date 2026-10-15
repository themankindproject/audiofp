//! Complex STFT used by log-magnitude DSSS.
//!
//! This is not [`crate::dsp::stft::ShortTimeFFT`]. That type is a packed
//! real FFT with reflect padding and a magnitude-only API. The measured
//! log-host mark uses a full complex FFT, a periodic Hann window, and
//! centred zero-padding, then weighted overlap-add on the way back.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use rustfft::{Fft, FftPlannerScalar};

use crate::dsp::windows::{WindowKind, make_window};

/// Modulus of a complex bin. `libm` so the value does not depend on the host libm.
#[inline]
pub(crate) fn modulus(c: Complex<f32>) -> f32 {
    libm::sqrtf(c.re * c.re + c.im * c.im)
}

pub(crate) struct ComplexStft {
    n_fft: usize,
    hop: usize,
    fwd: Arc<dyn Fft<f32>>,
    inv: Arc<dyn Fft<f32>>,
    win: Vec<f32>,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl ComplexStft {
    pub(crate) fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = FftPlannerScalar::new();
        let fwd = planner.plan_fft_forward(n_fft);
        let inv = planner.plan_fft_inverse(n_fft);
        let scratch_len = fwd
            .get_inplace_scratch_len()
            .max(inv.get_inplace_scratch_len());
        Self {
            n_fft,
            hop,
            fwd,
            inv,
            win: make_window(WindowKind::Hann, n_fft),
            buf: vec![Complex::new(0.0, 0.0); n_fft],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
        }
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frame count for `n_samples` with centred padding.
    pub(crate) fn n_frames(&self, n_samples: usize) -> usize {
        if n_samples == 0 {
            0
        } else {
            1 + n_samples / self.hop
        }
    }

    /// Analyse into a flat `(n_frames, n_bins)` complex spectrogram.
    pub(crate) fn analyze(&mut self, samples: &[f32]) -> (Vec<Complex<f32>>, usize) {
        let n_frames = self.n_frames(samples.len());
        let nb = self.n_bins();
        let n = samples.len() as isize;
        let center = (self.n_fft / 2) as isize;
        let mut out = vec![Complex::new(0.0, 0.0); n_frames * nb];
        for f in 0..n_frames {
            let start = f as isize * self.hop as isize - center;
            for k in 0..self.n_fft {
                let idx = start + k as isize;
                let s = if idx < 0 || idx >= n {
                    0.0
                } else {
                    samples[idx as usize]
                };
                self.buf[k] = Complex::new(s * self.win[k], 0.0);
            }
            self.fwd
                .process_with_scratch(&mut self.buf, &mut self.scratch);
            out[f * nb..(f + 1) * nb].copy_from_slice(&self.buf[..nb]);
        }
        (out, n_frames)
    }

    /// Reconstruct `n_samples` from a spectrogram produced by [`Self::analyze`].
    pub(crate) fn synthesize(
        &mut self,
        spec: &[Complex<f32>],
        n_frames: usize,
        n_samples: usize,
    ) -> Vec<f32> {
        let nb = self.n_bins();
        let center = (self.n_fft / 2) as isize;
        let total = n_samples as isize + 2 * center;
        let mut acc = vec![0.0f64; total as usize];
        let mut wsum = vec![0.0f64; total as usize];
        let scale = 1.0 / self.n_fft as f32;

        for f in 0..n_frames {
            for k in 0..nb {
                self.buf[k] = spec[f * nb + k];
            }
            for k in nb..self.n_fft {
                self.buf[k] = spec[f * nb + (self.n_fft - k)].conj();
            }
            self.inv
                .process_with_scratch(&mut self.buf, &mut self.scratch);
            let start = f as isize * self.hop as isize;
            for k in 0..self.n_fft {
                let p = start + k as isize;
                if p < 0 || p >= total {
                    continue;
                }
                let w = f64::from(self.win[k]);
                acc[p as usize] += f64::from(self.buf[k].re) * f64::from(scale) * w;
                wsum[p as usize] += w * w;
            }
        }
        let mut out = vec![0.0f32; n_samples];
        for (i, sample) in out.iter_mut().enumerate() {
            let p = (i as isize + center) as usize;
            if wsum[p] > 1e-8 {
                *sample = (acc[p] / wsum[p]) as f32;
            }
        }
        out
    }
}
