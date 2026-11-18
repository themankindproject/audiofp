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
        // The frame is centred on sample `f * hop`, so it starts half a
        // frame earlier. `before` of its slots lie ahead of the clip's first
        // sample, and it reads the clip from `from` on, as far as the clip
        // goes. Every slot outside the clip is zero.
        let center = self.n_fft / 2;
        let before = center.saturating_sub(f * self.hop);
        let from = (f * self.hop).saturating_sub(center).min(samples.len());
        let inside = &samples[from..samples.len().min(from + self.n_fft - before)];
        let after = before + inside.len();
        let zero = Complex::new(0.0, 0.0);
        self.buf[..before].fill(zero);
        for ((cell, &s), &w) in self.buf[before..after]
            .iter_mut()
            .zip(inside)
            .zip(&self.win[before..after])
        {
            *cell = Complex::new(s * w, 0.0);
        }
        self.buf[after..].fill(zero);
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
        // Bin `n_fft - k` mirrors bin `k`: the upper half is bins
        // `nb - 2` down to 1, conjugated.
        for (cell, bin) in self.buf[nb..].iter_mut().zip(bins[1..nb - 1].iter().rev()) {
            *cell = bin.conj();
        }
        self.inv
            .process_with_scratch(&mut self.buf, &mut self.scratch);
        let scale = f64::from(1.0 / self.n_fft as f32);
        let start = f * self.hop;
        // Positions from `total` on belong to no output sample.
        let len = self.n_fft.min(ola.total.saturating_sub(start));
        // The ring is as long as a frame, so a frame wraps around its end
        // at most once: one run up to the end, one from the start.
        let slot = start % self.n_fft;
        let head = len.min(self.n_fft - slot);
        ola.add(slot, &self.buf[..head], &self.win[..head], scale);
        ola.add(0, &self.buf[head..len], &self.win[head..len], scale);
        ola.covered = (start + self.n_fft).min(ola.total);
        // No later frame starts before `start + hop`.
        ola.finalize_until((start + self.hop).min(ola.total));
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
    /// Add the real part of `frame`, scaled and windowed, to the ring
    /// slots from `slot` on, which the run must not run off the end of.
    fn add(&mut self, slot: usize, frame: &[Complex<f32>], win: &[f32], scale: f64) {
        let acc = &mut self.acc[slot..slot + frame.len()];
        let wsum = &mut self.wsum[slot..slot + frame.len()];
        for (((acc, wsum), bin), &w) in acc.iter_mut().zip(wsum).zip(frame).zip(win) {
            let w = f64::from(w);
            *acc += f64::from(bin.re) * scale * w;
            *wsum += w * w;
        }
    }

    /// Finalise positions `[finalized, end)`: write each covered one to
    /// `out` (`acc / wsum`, or 0 where no window reached it) and clear its
    /// ring slot for the position `n_fft` later.
    fn finalize_until(&mut self, end: usize) {
        let ring = self.acc.len();
        let mut slot = self.finalized % ring;
        for p in self.finalized..end {
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
            slot = if slot + 1 == ring { 0 } else { slot + 1 };
        }
        self.finalized = self.finalized.max(end);
    }

    /// Finalise everything and return the samples.
    pub(crate) fn finish(mut self) -> Vec<f32> {
        let total = self.total;
        self.finalize_until(total);
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic samples in `[-1, 1)`.
    fn samples(len: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn bits(bins: &[Complex<f32>]) -> Vec<(u32, u32)> {
        bins.iter()
            .map(|bin| (bin.re.to_bits(), bin.im.to_bits()))
            .collect()
    }

    /// The analysis as first written: every slot looked up on its own, and
    /// a slot outside the clip read as a zero sample times the window.
    fn analyze_slot_by_slot(
        stft: &mut ComplexStft,
        samples: &[f32],
        f: usize,
    ) -> Vec<Complex<f32>> {
        let start = (f * stft.hop) as isize - (stft.n_fft / 2) as isize;
        let mut buf: Vec<Complex<f32>> = (0..stft.n_fft)
            .map(|k| {
                let idx = start + k as isize;
                let s = if idx < 0 || idx >= samples.len() as isize {
                    0.0
                } else {
                    samples[idx as usize]
                };
                Complex::new(s * stft.win[k], 0.0)
            })
            .collect();
        stft.fwd.process_with_scratch(&mut buf, &mut stft.scratch);
        buf.truncate(stft.n_bins());
        buf
    }

    #[test]
    fn analysis_matches_the_slot_by_slot_reference_bit_for_bit() {
        // Clips shorter than half a frame, shorter than a frame, and longer;
        // every frame, so both edges of the clip are crossed.
        for (n_fft, hop) in [(16, 4), (32, 8), (2048, 512)] {
            let mut stft = ComplexStft::new(n_fft, hop);
            for len in [
                1,
                3,
                n_fft / 2 - 1,
                n_fft / 2,
                n_fft - 1,
                n_fft,
                3 * n_fft + 5,
            ] {
                let clip = samples(len);
                for f in 0..stft.n_frames(len) {
                    let expected = analyze_slot_by_slot(&mut stft, &clip, f);
                    assert_eq!(
                        bits(stft.analyze_frame(&clip, f)),
                        bits(&expected),
                        "n_fft {n_fft}, {len} samples, frame {f}"
                    );
                }
            }
        }
    }

    /// Weighted overlap-add over whole-signal accumulators: every frame is
    /// transformed back and added at its position, then each position is
    /// divided by the window energy that reached it.
    fn synthesize_whole(
        stft: &mut ComplexStft,
        frames: &[Vec<Complex<f32>>],
        len: usize,
    ) -> Vec<f32> {
        let (n_fft, total) = (stft.n_fft, len + stft.n_fft);
        let mut acc = vec![0.0f64; total];
        let mut wsum = vec![0.0f64; total];
        let scale = 1.0 / n_fft as f32;
        for (f, bins) in frames.iter().enumerate() {
            let mut buf: Vec<Complex<f32>> = (0..n_fft)
                .map(|k| {
                    if k < stft.n_bins() {
                        bins[k]
                    } else {
                        bins[n_fft - k].conj()
                    }
                })
                .collect();
            stft.inv.process_with_scratch(&mut buf, &mut stft.scratch);
            for (k, (bin, &w)) in buf.iter().zip(&stft.win).enumerate() {
                let p = f * stft.hop + k;
                if p < total {
                    let w = f64::from(w);
                    acc[p] += f64::from(bin.re) * f64::from(scale) * w;
                    wsum[p] += w * w;
                }
            }
        }
        (0..len)
            .map(|i| {
                let p = i + n_fft / 2;
                if wsum[p] > 1e-8 {
                    (acc[p] / wsum[p]) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn frame_by_frame_synthesis_matches_whole_signal_overlap_add_bit_for_bit() {
        for (n_fft, hop) in [(16, 4), (32, 8), (2048, 512)] {
            let mut stft = ComplexStft::new(n_fft, hop);
            for len in [
                1,
                hop - 1,
                hop,
                n_fft - 1,
                n_fft,
                2 * n_fft + 3,
                7 * n_fft + hop / 2,
            ] {
                let clip = samples(len);
                let frames: Vec<Vec<Complex<f32>>> = (0..stft.n_frames(len))
                    .map(|f| stft.analyze_frame(&clip, f).to_vec())
                    .collect();
                let expected = synthesize_whole(&mut stft, &frames, len);
                let mut ola = stft.overlap_add(len);
                for (f, bins) in frames.iter().enumerate() {
                    stft.synthesize_frame(bins, f, &mut ola);
                }
                let found = ola.finish();
                let as_bits = |out: &[f32]| out.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
                assert_eq!(
                    as_bits(&found),
                    as_bits(&expected),
                    "n_fft {n_fft}, {len} samples"
                );
                // Unmodified frames give the clip back.
                for (i, (&out, &s)) in found.iter().zip(&clip).enumerate() {
                    assert!((out - s).abs() < 1e-4, "sample {i}: {out} vs {s}");
                }
            }
        }
    }
}
