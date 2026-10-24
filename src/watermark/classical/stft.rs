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
///
/// `re² + im²` overflows `f32` once a component passes ~1.8e19, which
/// extreme but finite samples reach; the f64 fallback keeps such bins
/// finite instead of turning them into `inf` (and later `NaN`). Ordinary
/// bins never take it, so their value is unchanged.
#[inline]
pub(crate) fn modulus(c: Complex<f32>) -> f32 {
    let m = libm::sqrtf(c.re * c.re + c.im * c.im);
    if m.is_finite() {
        m
    } else {
        let (re, im) = (f64::from(c.re), f64::from(c.im));
        libm::sqrt(re * re + im * im) as f32
    }
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

    /// Forward-transform frame `f` of `samples` (Hann window, centred
    /// zero-padding) and return its `n_bins` bins.
    ///
    /// Frames are produced one at a time so neither embed nor detect holds
    /// the whole complex spectrogram (16 bytes per input sample).
    pub(crate) fn analyze_frame(&mut self, samples: &[f32], f: usize) -> &[Complex<f32>] {
        let n = samples.len() as isize;
        let center = (self.n_fft / 2) as isize;
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
        &self.buf[..self.n_bins()]
    }

    /// Start weighted overlap-add synthesis of `n_samples` output samples.
    pub(crate) fn overlap_add(&self, n_samples: usize) -> OverlapAdd {
        OverlapAdd {
            acc: vec![0.0; self.n_fft],
            wsum: vec![0.0; self.n_fft],
            out: vec![0.0; n_samples],
            total: n_samples + self.n_fft,
            center: self.n_fft / 2,
            finalized: 0,
            covered: 0,
        }
    }

    /// Inverse-transform frame `f` (its `n_bins` bins, the upper half taken
    /// as their conjugates) and overlap-add it into `ola`. Frames must
    /// arrive in order `0, 1, 2, …`.
    ///
    /// Each output position accumulates the same terms in the same frame
    /// order as a whole-spectrogram synthesis, so the result is
    /// bit-identical; positions no later frame can reach are finalised as
    /// soon as this frame is in, so only an `n_fft`-long ring of
    /// accumulators is alive.
    pub(crate) fn synthesize_frame(
        &mut self,
        bins: &[Complex<f32>],
        f: usize,
        ola: &mut OverlapAdd,
    ) {
        let nb = self.n_bins();
        self.buf[..nb].copy_from_slice(&bins[..nb]);
        for k in nb..self.n_fft {
            self.buf[k] = bins[self.n_fft - k].conj();
        }
        self.inv
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        let scale = 1.0 / self.n_fft as f32;
        let start = f * self.hop;
        for k in 0..self.n_fft {
            let p = start + k;
            if p >= ola.total {
                continue;
            }
            let w = f64::from(self.win[k]);
            let slot = p % self.n_fft;
            ola.acc[slot] += f64::from(self.buf[k].re) * f64::from(scale) * w;
            ola.wsum[slot] += w * w;
        }
        ola.covered = (start + self.n_fft).min(ola.total);
        // No later frame starts before `start + hop`.
        ola.finalize_until((start + self.hop).min(ola.total), self.n_fft);
    }
}

/// Streaming weighted overlap-add state for [`ComplexStft::synthesize_frame`].
pub(crate) struct OverlapAdd {
    /// Ring of per-position accumulators, indexed by `position % n_fft`.
    acc: Vec<f64>,
    wsum: Vec<f64>,
    out: Vec<f32>,
    /// Padded length: `n_samples + n_fft` (centred padding on both sides).
    total: usize,
    center: usize,
    /// Positions below this are written to `out`.
    finalized: usize,
    /// Positions below this have been reached by some frame.
    covered: usize,
}

impl OverlapAdd {
    /// Finalise positions `[finalized, end)`: write each covered one to
    /// `out` (`acc / wsum`, or 0 where no window reached it) and clear its
    /// ring slot for the position `n_fft` later.
    fn finalize_until(&mut self, end: usize, n_fft: usize) {
        for p in self.finalized..end {
            let slot = p % n_fft;
            let (acc, wsum) = if p < self.covered {
                (self.acc[slot], self.wsum[slot])
            } else {
                (0.0, 0.0)
            };
            if let Some(i) = p.checked_sub(self.center)
                && i < self.out.len()
                && wsum > 1e-8
            {
                self.out[i] = (acc / wsum) as f32;
            }
            self.acc[slot] = 0.0;
            self.wsum[slot] = 0.0;
        }
        self.finalized = self.finalized.max(end);
    }

    /// Finalise everything and return the samples.
    pub(crate) fn finish(mut self, n_fft: usize) -> Vec<f32> {
        let total = self.total;
        self.finalize_until(total, n_fft);
        self.out
    }
}
