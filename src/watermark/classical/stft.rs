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

    /// Analyse one centred frame into caller-owned spectral scratch.
    pub(crate) fn analyze_frame(
        &mut self,
        samples: &[f32],
        frame: usize,
        out: &mut [Complex<f32>],
    ) {
        debug_assert_eq!(out.len(), self.n_bins());
        self.analyze_frame_in_place(samples, frame);
        out.copy_from_slice(&self.buf[..self.n_bins()]);
    }

    fn analyze_frame_in_place(&mut self, samples: &[f32], frame: usize) {
        let n = samples.len() as isize;
        let start = frame as isize * self.hop as isize - (self.n_fft / 2) as isize;
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
    }

    /// Map each half-spectrum and reconstruct with bounded overlap scratch.
    /// Frame order and per-sample summation order follow the original
    /// complete-spectrogram overlap-add path.
    pub(crate) fn map_overlap_add(
        &mut self,
        samples: &[f32],
        mut map: impl FnMut(usize, &mut [Complex<f32>]),
    ) -> Vec<f32> {
        let n_frames = self.n_frames(samples.len());
        let nb = self.n_bins();
        let center = self.n_fft / 2;
        let end = center + samples.len();
        let mut acc = vec![0.0f64; self.n_fft];
        let mut wsum = vec![0.0f64; self.n_fft];
        let mut out = vec![0.0f32; samples.len()];
        let scale = 1.0 / self.n_fft as f32;
        let mut finalized = 0usize;

        for f in 0..n_frames {
            self.analyze_frame_in_place(samples, f);
            map(f, &mut self.buf[..nb]);
            for k in nb..self.n_fft {
                self.buf[k] = self.buf[self.n_fft - k].conj();
            }
            self.inv
                .process_with_scratch(&mut self.buf, &mut self.scratch);
            let start = f * self.hop;
            for k in 0..self.n_fft {
                let p = start + k;
                let slot = p % self.n_fft;
                let w = f64::from(self.win[k]);
                acc[slot] += f64::from(self.buf[k].re) * f64::from(scale) * w;
                wsum[slot] += w * w;
            }
            // The next frame starts at `(f + 1) * hop`; every earlier
            // padded position has received its last contribution.
            let ready = (start + self.hop).min(end);
            while finalized < ready {
                let slot = finalized % self.n_fft;
                if finalized >= center && wsum[slot] > 1e-8 {
                    out[finalized - center] = (acc[slot] / wsum[slot]) as f32;
                }
                acc[slot] = 0.0;
                wsum[slot] = 0.0;
                finalized += 1;
            }
        }
        while finalized < end {
            let slot = finalized % self.n_fft;
            if finalized >= center && wsum[slot] > 1e-8 {
                out[finalized - center] = (acc[slot] / wsum[slot]) as f32;
            }
            finalized += 1;
        }
        out
    }
}
