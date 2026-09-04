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
use alloc::vec;
use alloc::vec::Vec;

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
    /// use audiofp::dsp::stft::StftConfig;
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
/// use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
///
/// let mut stft = ShortTimeFFT::new(StftConfig::new(1024));
/// let samples = vec![0.0_f32; 16_000];
/// let (spec, n_frames, n_bins) = stft.magnitude_flat(&samples);
/// // n_bins = n_fft/2 + 1 = 513 for n_fft=1024.
/// assert_eq!(n_bins, 513);
/// assert_eq!(spec.len(), n_frames * n_bins);
/// ```
pub struct ShortTimeFFT {
    cfg: StftConfig,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    scratch_in: Vec<f32>,
    scratch_out: Vec<Complex<f32>>,
    fft_scratch: Vec<Complex<f32>>,
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
        Self::try_new(cfg).expect("invalid StftConfig")
    }

    /// Fallible constructor — returns [`AfpError::Config`](crate::AfpError::Config) on invalid
    /// parameters instead of panicking.
    ///
    /// # Errors
    ///
    /// - `n_fft` is zero or not a power of two
    /// - `hop` is zero or larger than `n_fft`
    pub fn try_new(cfg: StftConfig) -> crate::Result<Self> {
        if cfg.n_fft == 0 || !cfg.n_fft.is_power_of_two() {
            return Err(crate::AfpError::Config(alloc::format!(
                "n_fft must be a non-zero power of two, got {}",
                cfg.n_fft
            )));
        }
        if cfg.hop == 0 || cfg.hop > cfg.n_fft {
            return Err(crate::AfpError::Config(alloc::format!(
                "hop must be in (0, n_fft], got hop={} n_fft={}",
                cfg.hop,
                cfg.n_fft
            )));
        }

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(cfg.n_fft);
        let window = make_window(cfg.window, cfg.n_fft);
        let scratch_in = fft.make_input_vec();
        let scratch_out = fft.make_output_vec();
        let fft_scratch = fft.make_scratch_vec();

        Ok(Self {
            cfg,
            fft,
            window,
            scratch_in,
            scratch_out,
            fft_scratch,
        })
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
    #[deprecated(
        since = "0.3.9",
        note = "use magnitude_flat() instead — same data, single allocation, better cache locality"
    )]
    pub fn magnitude(&mut self, samples: &[f32]) -> Vec<Vec<f32>> {
        let (flat, n_frames, n_bins) = self.magnitude_flat(samples);
        if n_frames == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            out.push(flat[f * n_bins..(f + 1) * n_bins].to_vec());
        }
        out
    }

    /// Compute the **power** spectrogram of `samples` into a caller-owned
    /// `out` buffer, returning `(n_frames, n_bins)`. Each cell is
    /// `re² + im²`. The buffer is resized to `n_frames * n_bins` if
    /// smaller; excess capacity is reused.
    ///
    /// Avoids the allocation of [`power_flat`] when the caller already
    /// owns a suitably-sized buffer.
    ///
    /// [`power_flat`]: ShortTimeFFT::power_flat
    pub fn power_flat_into(&mut self, samples: &[f32], out: &mut Vec<f32>) -> (usize, usize) {
        if samples.is_empty() {
            out.clear();
            return (0, 0);
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

        out.resize(n_frames * n_bins, 0.0);

        for f in 0..n_frames {
            let start = (f * hop) as isize - center_off;
            self.fill_windowed(samples, start);

            self.fft
                .process_with_scratch(
                    &mut self.scratch_in,
                    &mut self.scratch_out,
                    &mut self.fft_scratch,
                )
                .expect("FFT process: input/output length mismatch");

            let row = &mut out[f * n_bins..(f + 1) * n_bins];
            compute_power_wide(&self.scratch_out, row);
        }

        (n_frames, n_bins)
    }

    /// Compute the **power** spectrogram of `samples` into a single
    /// contiguous `Vec<f32>` of shape `(n_frames, n_bins)`. Each cell is
    /// `re² + im²` — equivalent to `magnitude_flat`'s output squared,
    /// but without the per-bin `sqrt`.
    ///
    /// Useful when the next stage applies `log10` (which combines
    /// algebraically with the missing `sqrt`: `20·log10(sqrt(p)) ==
    /// 10·log10(p)`) or any other operation that doesn't need the
    /// magnitude itself. The classical fingerprinters all consume
    /// `power_flat` for this reason.
    ///
    /// # Example
    ///
    /// ```
    /// use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
    ///
    /// let mut stft = ShortTimeFFT::new(StftConfig::new(1024));
    /// let samples = vec![1.0_f32; 4096];
    /// let (power, n_frames, n_bins) = stft.power_flat(&samples);
    /// assert_eq!(power.len(), n_frames * n_bins);
    /// // Centre frame's DC bin dominates by orders of magnitude.
    /// let mid = (n_frames / 2) * n_bins;
    /// assert!(power[mid] > power[mid + 2] * 1_000.0);
    /// ```
    #[must_use]
    pub fn power_flat(&mut self, samples: &[f32]) -> (Vec<f32>, usize, usize) {
        let mut out = Vec::new();
        let (n_frames, n_bins) = self.power_flat_into(samples, &mut out);
        (out, n_frames, n_bins)
    }

    /// Compute the magnitude spectrogram of `samples` into a single
    /// contiguous `Vec<f32>` of shape `(n_frames, n_bins)` (row-major).
    ///
    /// Returns `(data, n_frames, n_bins)`. Far cheaper than [`magnitude`]
    /// for large inputs because it does a single allocation instead of
    /// one per frame, and it lets downstream consumers slice the
    /// spectrogram directly without indirection.
    ///
    /// [`magnitude`]: ShortTimeFFT::magnitude
    ///
    /// # Example
    ///
    /// ```
    /// use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
    ///
    /// let mut stft = ShortTimeFFT::new(StftConfig::new(1024));
    /// let samples = vec![0.0_f32; 16_000];
    /// let (mag, n_frames, n_bins) = stft.magnitude_flat(&samples);
    /// assert_eq!(mag.len(), n_frames * n_bins);
    /// assert_eq!(n_bins, 513);
    /// ```
    #[must_use]
    pub fn magnitude_flat(&mut self, samples: &[f32]) -> (Vec<f32>, usize, usize) {
        if samples.is_empty() {
            return (Vec::new(), 0, 0);
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

        let mut out = vec![0.0_f32; n_frames * n_bins];

        for f in 0..n_frames {
            let start = (f * hop) as isize - center_off;
            self.fill_windowed(samples, start);

            self.fft
                .process_with_scratch(
                    &mut self.scratch_in,
                    &mut self.scratch_out,
                    &mut self.fft_scratch,
                )
                .expect("FFT process: input/output length mismatch");

            let row = &mut out[f * n_bins..(f + 1) * n_bins];
            compute_magnitude_wide(&self.scratch_out, row);
        }

        (out, n_frames, n_bins)
    }

    /// Streaming variant: window one `n_fft`-sized frame and emit its
    /// **power** spectrum (`re² + im²`) into `out` (`n_bins` long).
    ///
    /// Same as [`process_frame`] but skips the per-bin `sqrt`. Useful in
    /// the streaming fingerprinter front-ends, where every step downstream
    /// applies `log10` (or band-summing) and absorbing the `sqrt` is a
    /// simple constant adjustment.
    ///
    /// [`process_frame`]: ShortTimeFFT::process_frame
    ///
    /// # Errors
    ///
    /// Returns [`AfpError::Config`](crate::AfpError::Config) if
    /// `frame.len() != n_fft` or `out.len() != n_bins`.
    ///
    /// # Example
    ///
    /// ```
    /// use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
    ///
    /// let mut stft = ShortTimeFFT::new(StftConfig::new(256));
    /// let frame = vec![0.0_f32; 256];
    /// let mut out = vec![0.0_f32; 129]; // n_fft/2 + 1
    /// stft.process_frame_power(&frame, &mut out).unwrap();
    /// assert!(out.iter().all(|&p| p == 0.0)); // silent input → zero power
    /// ```
    pub fn process_frame_power(&mut self, frame: &[f32], out: &mut [f32]) -> crate::Result<()> {
        self.run_frame(frame, out, compute_power_wide)
    }

    /// Streaming variant: window one `n_fft`-sized frame and emit its
    /// magnitude spectrum into `out` (`n_bins` long).
    ///
    /// # Errors
    ///
    /// Returns [`AfpError::Config`](crate::AfpError::Config) if
    /// `frame.len() != n_fft` or `out.len() != n_bins`.
    pub fn process_frame(&mut self, frame: &[f32], out: &mut [f32]) -> crate::Result<()> {
        self.run_frame(frame, out, compute_magnitude_wide)
    }

    /// Shared single-frame pipeline: validate lengths, apply the window,
    /// run the FFT, then reduce the complex spectrum with `compute`
    /// ([`compute_power_wide`] or [`compute_magnitude_wide`]).
    fn run_frame(
        &mut self,
        frame: &[f32],
        out: &mut [f32],
        compute: fn(&[Complex<f32>], &mut [f32]),
    ) -> crate::Result<()> {
        if frame.len() != self.cfg.n_fft {
            return Err(crate::AfpError::Config(alloc::format!(
                "frame length must equal n_fft: got {}, expected {}",
                frame.len(),
                self.cfg.n_fft
            )));
        }
        if out.len() != self.n_bins() {
            return Err(crate::AfpError::Config(alloc::format!(
                "out length must equal n_bins: got {}, expected {}",
                out.len(),
                self.n_bins()
            )));
        }

        apply_window_wide(frame, &self.window, &mut self.scratch_in);

        self.fft
            .process_with_scratch(
                &mut self.scratch_in,
                &mut self.scratch_out,
                &mut self.fft_scratch,
            )
            .expect("FFT process: input/output length mismatch");

        compute(&self.scratch_out, out);

        Ok(())
    }

    /// Fill `scratch_in` with `samples[start..start+n_fft] * window`,
    /// reflecting indices that fall outside `samples` when the config
    /// uses centred framing.
    ///
    /// Hot-path optimised: when the window slot lives entirely inside
    /// the input buffer (which is true for almost every frame in any
    /// non-edge audio), we take a fast path with no per-sample bounds
    /// or reflect check.
    fn fill_windowed(&mut self, samples: &[f32], start: isize) {
        let n_fft = self.cfg.n_fft;
        let len = samples.len();

        // Fast inner path — window slot fully inside `samples`.
        if start >= 0 && (start as usize).saturating_add(n_fft) <= len {
            let s_off = start as usize;
            let src = &samples[s_off..s_off + n_fft];
            let win = &self.window[..n_fft];
            let dst = &mut self.scratch_in[..n_fft];

            apply_window_wide(src, win, dst);
            return;
        }

        // Slow path — at the buffer edges, with bounds + reflect check.
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

/// SIMD-accelerated window application: `dst[i] = src[i] * win[i]` using `wide`.
///
/// Processes 8 elements at a time via `f32x8` (AVX2/SSE/NEON depending on
/// target), with a scalar tail for the remainder. Entirely safe code.
fn apply_window_wide(src: &[f32], win: &[f32], dst: &mut [f32]) {
    crate::dsp::simd::mul_into(src, win, dst);
}

/// SIMD-accelerated magnitude computation: `dst[i] = sqrt(re² + im²)` using
/// `wide`.
///
/// Vectorises the sqrt that the scalar path must take through `libm::sqrtf`,
/// which cannot be auto-vectorised. `f32x8::sqrt` is the hardware IEEE sqrt
/// (or the same musl-derived software sqrt in `wide`'s no-SIMD fallback), so
/// on default builds results are bit-identical to the scalar loop. On FMA
/// builds the `mul_add` fuses one rounding, matching the existing
/// `compute_power_wide` behaviour.
fn compute_magnitude_wide(complex: &[Complex<f32>], dst: &mut [f32]) {
    crate::dsp::simd::complex_magnitude_into(complex, dst);
}

/// SIMD-accelerated power computation: `dst[i] = complex[i].re² + complex[i].im²`
/// using `wide`.
///
/// Processes 8 power values at a time via `f32x8` by separately loading
/// the real and imaginary parts, then computing `re * re + im * im`.
fn compute_power_wide(complex: &[Complex<f32>], dst: &mut [f32]) {
    crate::dsp::simd::complex_power_into(complex, dst);
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
#[allow(deprecated)]
mod tests {
    use super::*;
    use alloc::string::ToString;
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
    #[should_panic(expected = "n_fft must be a non-zero power of two")]
    fn new_panics_on_zero_n_fft() {
        let _ = ShortTimeFFT::new(StftConfig {
            n_fft: 0,
            hop: 256,
            window: WindowKind::Hann,
            center: true,
        });
    }

    #[test]
    #[should_panic(expected = "n_fft must be a non-zero power of two")]
    fn new_panics_on_non_power_of_two_n_fft() {
        let _ = ShortTimeFFT::new(StftConfig {
            n_fft: 1000,
            hop: 250,
            window: WindowKind::Hann,
            center: true,
        });
    }

    #[test]
    #[should_panic(expected = "hop must be in (0, n_fft]")]
    fn new_panics_on_zero_hop() {
        let _ = ShortTimeFFT::new(StftConfig {
            n_fft: 1024,
            hop: 0,
            window: WindowKind::Hann,
            center: true,
        });
    }

    #[test]
    #[should_panic(expected = "hop must be in (0, n_fft]")]
    fn new_panics_on_hop_greater_than_n_fft() {
        let _ = ShortTimeFFT::new(StftConfig {
            n_fft: 1024,
            hop: 2048,
            window: WindowKind::Hann,
            center: true,
        });
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
        s.process_frame(&samples, &mut frame_out).unwrap();

        let mut s2 = ShortTimeFFT::new(cfg);
        let buf_out = s2.magnitude(&samples);

        assert_eq!(buf_out.len(), 1);
        for (a, b) in frame_out.iter().zip(buf_out[0].iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-5);
        }
    }

    #[test]
    fn process_frame_rejects_wrong_frame_and_out_lengths() {
        let mut s = ShortTimeFFT::new(StftConfig::new(256));
        let frame = alloc::vec![0.0_f32; 256];
        let mut out = alloc::vec![0.0_f32; s.n_bins()];

        // Wrong frame length.
        let err = s.process_frame(&frame[..128], &mut out).unwrap_err();
        assert!(matches!(err, crate::AfpError::Config(_)));
        assert!(err.to_string().contains("frame length"));

        // Wrong out length.
        let mut short_out = alloc::vec![0.0_f32; s.n_bins() - 1];
        let err = s.process_frame(&frame, &mut short_out).unwrap_err();
        assert!(matches!(err, crate::AfpError::Config(_)));
        assert!(err.to_string().contains("out length"));

        // Power variant behaves identically.
        let err = s.process_frame_power(&frame[..128], &mut out).unwrap_err();
        assert!(matches!(err, crate::AfpError::Config(_)));
        let err = s.process_frame_power(&frame, &mut short_out).unwrap_err();
        assert!(matches!(err, crate::AfpError::Config(_)));

        // And the happy path still works after failed calls (no state corruption).
        s.process_frame(&frame, &mut out).unwrap();
        s.process_frame_power(&frame, &mut out).unwrap();
    }

    // `power_flat` / `power_flat_into` direct coverage.
    //
    // These two functions are the inputs to the Wang/Panako/Haitsma
    // hash builders and are exercised transitively by every classical
    // test, but had no direct unit test. They also encode the
    // identity `power = |magnitude|²` (modulo float-rounding), which
    // is the contract that lets the hash builders skip the redundant
    // `sqrt`.

    #[test]
    fn power_flat_matches_magnitude_squared() {
        // 1 kHz tone at 16 kHz — same input as `pure_sine_peaks_at_expected_bin`.
        let n_fft = 1024;
        let sr = 16_000.0_f32;
        let freq = 1_000.0_f32;
        let mut s = ShortTimeFFT::new(StftConfig::new(n_fft));

        let samples: alloc::vec::Vec<f32> = (0..4_096)
            .map(|n| libm::sinf(2.0 * core::f32::consts::PI * freq * n as f32 / sr))
            .collect();

        let (power, n_frames, n_bins) = s.power_flat(&samples);
        let magnitude = s.magnitude(&samples);

        assert_eq!(n_frames, magnitude.len());
        assert_eq!(n_bins, magnitude[0].len());
        assert_eq!(power.len(), n_frames * n_bins);
        for (f, mag_row) in magnitude.iter().enumerate() {
            for (b, &m) in mag_row.iter().enumerate() {
                let p = power[f * n_bins + b];
                // power == |magnitude|². Use a relative epsilon for
                // large magnitudes, absolute for small ones.
                let want = m * m;
                if want.abs() > 1e-3 {
                    assert_relative_eq!(p, want, max_relative = 1e-5);
                } else {
                    assert!((p - want).abs() < 1e-6, "frame={f} bin={b}: {p} vs {want}");
                }
            }
        }
    }

    #[test]
    fn power_flat_into_writes_into_caller_vec_without_realloc() {
        // Reuse a `Vec` across calls; `power_flat_into` must `resize`
        // to the right size without throwing away the existing
        // capacity (this is what makes it zero-alloc on the hot path).
        let n_fft = 1024;
        let sr = 16_000.0_f32;
        let freq = 1_000.0_f32;
        let mut s = ShortTimeFFT::new(StftConfig::new(n_fft));

        let samples: alloc::vec::Vec<f32> = (0..4_096)
            .map(|n| libm::sinf(2.0 * core::f32::consts::PI * freq * n as f32 / sr))
            .collect();

        let mut buf: alloc::vec::Vec<f32> = alloc::vec::Vec::new();
        let initial_cap = buf.capacity();
        let (n_frames, n_bins) = s.power_flat_into(&samples, &mut buf);
        let after_first = buf.capacity();

        assert_eq!(n_frames * n_bins, buf.len());
        // Capacity should be ≥ len, and on the second call should
        // not grow (the Vec already has room for this size).
        assert!(after_first >= n_frames * n_bins);

        let (n_frames2, _) = s.power_flat_into(&samples, &mut buf);
        assert_eq!(n_frames2, n_frames);
        // Capacity must be preserved (no realloc, no shrink).
        assert_eq!(buf.capacity(), after_first);

        // And: empty input clears without growing.
        let mut empty_buf: alloc::vec::Vec<f32> = alloc::vec![1.0; 64];
        let (nf, nb) = s.power_flat_into(&[], &mut empty_buf);
        assert_eq!((nf, nb), (0, 0));
        assert!(empty_buf.is_empty());
        // (Don't assert capacity here — clear() is allowed to shrink.)
        let _ = initial_cap;
    }
}
