//! Haitsma–Kalker / Philips robust hash.
//!
//! Reference: Jaap Haitsma & Ton Kalker, "A Highly Robust Audio
//! Fingerprinting System" (ISMIR 2002).
//!
//! Algorithm:
//!
//! 1. Resample the input to 5 kHz mono *(caller's responsibility)*.
//! 2. Take a Hann-windowed STFT with `n_fft = 2048`, `hop = 64`
//!    (≈78.125 frames/s).
//! 3. Sum power across **33 logarithmically-spaced bands** from 300 Hz
//!    to 2000 Hz → `E[n][b]`.
//! 4. For each frame `n ≥ 1` and band index `b ∈ {0..=31}` emit one bit:
//!
//!    ```text
//!    F[n][b] = ((E[n][b] − E[n][b+1]) − (E[n−1][b] − E[n−1][b+1])) > 0
//!    ```
//!
//! 5. Pack the 32 bits per frame into a `u32` with band 0 in the most
//!    significant bit (the spec's "MSB-zero" ordering) and band 31 in
//!    the least significant.

use alloc::vec::Vec;

use libm::powf;

use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{
    AfpError, AudioBuffer, Fingerprinter, Result, SampleRate, StreamingFingerprinter, TimestampMs,
};

/// All bit-frames produced by [`Haitsma`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct HaitsmaFingerprint {
    /// One `u32` per STFT frame from `n=1` onwards.
    pub frames: Vec<u32>,
    /// Frame rate of the underlying STFT — always 78.125 for `haitsma-v1`
    /// (`5000 / 64`).
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Haitsma`].
#[derive(Clone, Debug)]
pub struct HaitsmaConfig {
    /// Lowest band edge in Hz. Default 300.
    pub fmin: f32,
    /// Highest band edge in Hz. Default 2000.
    pub fmax: f32,
}

impl Default for HaitsmaConfig {
    fn default() -> Self {
        Self {
            fmin: 300.0,
            fmax: 2_000.0,
        }
    }
}

const HAITSMA_N_FFT: usize = 2048;
const HAITSMA_HOP: usize = 64;
const HAITSMA_SR: u32 = 5_000;
const HAITSMA_FRAMES_PER_SEC: f32 = HAITSMA_SR as f32 / HAITSMA_HOP as f32;
const HAITSMA_N_BANDS: usize = 33;

/// Haitsma–Kalker offline fingerprinter.
///
/// # Example
///
/// ```
/// use afp::{AudioBuffer, Fingerprinter, SampleRate};
/// use afp::classical::Haitsma;
///
/// let mut fp = Haitsma::default();
/// let samples = vec![0.0_f32; 5_000 * 3];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
/// // Wrong rate is rejected immediately.
/// assert!(fp.extract(buf).is_err());
///
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::new(5_000).unwrap() };
/// let fpr = fp.extract(buf).unwrap();
/// assert_eq!(fpr.frames_per_sec, 78.125);
/// // Silence → all-zero hash frames (no band differences).
/// for &h in &fpr.frames {
///     assert_eq!(h, 0);
/// }
/// ```
pub struct Haitsma {
    cfg: HaitsmaConfig,
    stft: ShortTimeFFT,
    /// `bin_to_band[i]` = `Some(b)` if FFT bin `i` falls inside band `b`,
    /// `None` if it lies outside `[fmin, fmax]`.
    bin_to_band: Vec<Option<u8>>,
}

impl Default for Haitsma {
    fn default() -> Self {
        Self::new(HaitsmaConfig::default())
    }
}

impl Haitsma {
    /// Build a Haitsma extractor with the given config.
    ///
    /// # Panics
    ///
    /// Panics if `cfg.fmin <= 0`, `cfg.fmax <= cfg.fmin`, or
    /// `cfg.fmax >= HAITSMA_SR / 2` (above Nyquist).
    #[must_use]
    pub fn new(cfg: HaitsmaConfig) -> Self {
        assert!(cfg.fmin > 0.0, "fmin must be positive");
        assert!(cfg.fmax > cfg.fmin, "fmax must exceed fmin");
        assert!(
            cfg.fmax < HAITSMA_SR as f32 / 2.0,
            "fmax must be below Nyquist ({} Hz)",
            HAITSMA_SR / 2
        );

        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: HAITSMA_N_FFT,
            hop: HAITSMA_HOP,
            window: WindowKind::Hann,
            center: false,
        });

        let bin_to_band = build_bin_to_band(&cfg, stft.n_bins());

        Self {
            cfg,
            stft,
            bin_to_band,
        }
    }
}

impl Fingerprinter for Haitsma {
    type Output = HaitsmaFingerprint;
    type Config = HaitsmaConfig;

    fn name(&self) -> &'static str {
        "haitsma-v1"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> u32 {
        HAITSMA_SR
    }

    fn min_samples(&self) -> usize {
        HAITSMA_SR as usize * 2
    }

    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
        if audio.rate.hz() != HAITSMA_SR {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        if audio.samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: audio.samples.len(),
            });
        }

        let spec = self.stft.magnitude(audio.samples);
        if spec.len() < 2 {
            return Ok(HaitsmaFingerprint {
                frames: Vec::new(),
                frames_per_sec: HAITSMA_FRAMES_PER_SEC,
            });
        }

        // Compute per-frame band energies (power, not magnitude).
        let mut energies: Vec<[f32; HAITSMA_N_BANDS]> = Vec::with_capacity(spec.len());
        for frame in &spec {
            let mut e = [0.0_f32; HAITSMA_N_BANDS];
            for (bin, &m) in frame.iter().enumerate() {
                if let Some(b) = self.bin_to_band[bin] {
                    e[b as usize] += m * m;
                }
            }
            energies.push(e);
        }

        // For each frame n >= 1, compute the 32-bit hash.
        let mut frames = Vec::with_capacity(energies.len() - 1);
        for n in 1..energies.len() {
            frames.push(pack_frame_bits(&energies[n], &energies[n - 1]));
        }

        Ok(HaitsmaFingerprint {
            frames,
            frames_per_sec: HAITSMA_FRAMES_PER_SEC,
        })
    }
}

/// Pack 32 sign bits comparing band-difference deltas between frame `n`
/// and frame `n−1`.
fn pack_frame_bits(curr: &[f32; HAITSMA_N_BANDS], prev: &[f32; HAITSMA_N_BANDS]) -> u32 {
    let mut hash = 0_u32;
    for b in 0..32 {
        let lhs = curr[b] - curr[b + 1];
        let rhs = prev[b] - prev[b + 1];
        if lhs - rhs > 0.0 {
            // "MSB-zero": band 0 lands in the most significant bit.
            hash |= 1_u32 << (31 - b);
        }
    }
    hash
}

/// Compute the FFT-bin → band-index lookup table.
///
/// 33 bands defined by 34 logarithmically-spaced edges from `fmin` to
/// `fmax`. Bin `i`'s frequency is `i · sr / n_fft`. A bin in
/// `[edge_b, edge_{b+1})` is mapped to band `b`.
fn build_bin_to_band(cfg: &HaitsmaConfig, n_bins: usize) -> Vec<Option<u8>> {
    let n_edges = HAITSMA_N_BANDS + 1;
    let mut edges = [0.0_f32; HAITSMA_N_BANDS + 1];
    let ratio = cfg.fmax / cfg.fmin;
    for (k, e) in edges.iter_mut().enumerate() {
        let frac = k as f32 / HAITSMA_N_BANDS as f32;
        *e = cfg.fmin * powf(ratio, frac);
    }

    let bin_hz = HAITSMA_SR as f32 / HAITSMA_N_FFT as f32;

    let mut out = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        let f = i as f32 * bin_hz;
        if f < edges[0] || f >= edges[n_edges - 1] {
            out.push(None);
            continue;
        }
        // Linear scan is fine — only 33 edges.
        let mut found = None;
        for b in 0..HAITSMA_N_BANDS {
            if f >= edges[b] && f < edges[b + 1] {
                found = Some(b as u8);
                break;
            }
        }
        out.push(found);
    }
    out
}

/// Streaming Haitsma–Kalker fingerprinter.
///
/// Each new frame's hash depends only on the current and previous
/// frames' band energies, so latency is bounded by the STFT window
/// length (`n_fft / sr ≈ 410 ms`) — much lower than the landmark
/// extractors.
///
/// **Implementation note:** like `StreamingWang`, the current version
/// reruns the offline pipeline on each push. Because Haitsma has no
/// peak picker or per-second adaptive thresholding, an incremental
/// implementation is straightforward and a future optimization.
pub struct StreamingHaitsma {
    cfg: HaitsmaConfig,
    accumulated: Vec<f32>,
    /// First frame index (1-based, since frame 0 has no hash) whose hash
    /// has not yet been emitted.
    next_frame_idx: u32,
}

impl Default for StreamingHaitsma {
    fn default() -> Self {
        Self::new(HaitsmaConfig::default())
    }
}

impl StreamingHaitsma {
    /// Build a streaming Haitsma extractor with the given config.
    #[must_use]
    pub fn new(cfg: HaitsmaConfig) -> Self {
        Self {
            cfg,
            accumulated: Vec::new(),
            next_frame_idx: 1, // hash index 0 is for frame 1 (first hashable frame)
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &HaitsmaConfig {
        &self.cfg
    }

    fn drain_all(&mut self) -> Vec<(TimestampMs, u32)> {
        let mut h = Haitsma::new(self.cfg.clone());
        let audio = AudioBuffer {
            samples: &self.accumulated,
            // SAFETY: 5000 is non-zero.
            rate: SampleRate::new(HAITSMA_SR).unwrap(),
        };
        let result = match h.extract(audio) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        let mut emitted = Vec::new();
        for (i, &hash) in result.frames.iter().enumerate() {
            // Frame index in spectrogram terms is i+1 (we skip frame 0).
            let frame_idx = (i as u32) + 1;
            if frame_idx >= self.next_frame_idx {
                let t_ms = (frame_idx as u64 * HAITSMA_HOP as u64 * 1000) / HAITSMA_SR as u64;
                emitted.push((TimestampMs(t_ms), hash));
            }
        }
        if let Some((_, _)) = emitted.last() {
            self.next_frame_idx = (result.frames.len() as u32) + 1;
        }
        emitted
    }
}

impl StreamingFingerprinter for StreamingHaitsma {
    type Frame = u32;

    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)> {
        self.accumulated.extend_from_slice(samples);
        self.drain_all()
    }

    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)> {
        self.drain_all()
    }

    fn latency_ms(&self) -> u32 {
        // A sample at the start of frame n is only covered fully once
        // frame n's STFT is ready, i.e. n_fft samples later.
        (HAITSMA_N_FFT as u32 * 1000) / HAITSMA_SR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::f32::consts::PI;

    fn synthetic_audio(seed: u32, len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        let mut x: u32 = seed.max(1);
        for n in 0..len {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
            let t = n as f32 / 5_000.0;
            // Use frequencies in the 300-2000 Hz band so they hit our bands.
            let s = 0.5 * libm::sinf(2.0 * PI * 600.0 * t)
                + 0.3 * libm::sinf(2.0 * PI * 1200.0 * t)
                + noise;
            out.push(s);
        }
        out
    }

    fn chunk_sizes(seed: u32, total: usize, max_chunk: usize) -> Vec<usize> {
        let mut x = seed.max(1);
        let mut out = Vec::new();
        let mut remaining = total;
        while remaining > 0 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let n = ((x as usize) % max_chunk).max(1).min(remaining);
            out.push(n);
            remaining -= n;
        }
        out
    }

    fn sr_5khz() -> SampleRate {
        SampleRate::new(5_000).unwrap()
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 10_000];
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_16000,
        };
        match fp.extract(buf) {
            Err(AfpError::UnsupportedSampleRate(16_000)) => {}
            other => panic!("expected UnsupportedSampleRate(16000), got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_audio() {
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 5_000];
        let buf = AudioBuffer {
            samples: &samples,
            rate: sr_5khz(),
        };
        match fp.extract(buf) {
            Err(AfpError::AudioTooShort {
                needed: 10_000,
                got: 5_000,
            }) => {}
            other => panic!("expected AudioTooShort, got {other:?}"),
        }
    }

    #[test]
    fn silence_gives_all_zero_frames() {
        let mut fp = Haitsma::default();
        let samples = vec![0.0_f32; 5_000 * 3];
        let buf = AudioBuffer {
            samples: &samples,
            rate: sr_5khz(),
        };
        let fpr = fp.extract(buf).unwrap();
        assert_eq!(fpr.frames_per_sec, 78.125);
        assert!(!fpr.frames.is_empty());
        for &h in &fpr.frames {
            assert_eq!(h, 0, "silence should produce zero hash");
        }
    }

    #[test]
    fn synthetic_signal_produces_nonzero_hashes() {
        let mut fp = Haitsma::default();
        let samples = synthetic_audio(0xC0FFEE, 5_000 * 4);
        let buf = AudioBuffer {
            samples: &samples,
            rate: sr_5khz(),
        };
        let fpr = fp.extract(buf).unwrap();
        assert!(!fpr.frames.is_empty());
        let nonzero = fpr.frames.iter().filter(|&&h| h != 0).count();
        assert!(
            nonzero > fpr.frames.len() / 4,
            "expected most frames to have at least one bit set, got {nonzero}/{}",
            fpr.frames.len()
        );
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 5_000 * 3);

        let mut fp1 = Haitsma::default();
        let f1 = fp1
            .extract(AudioBuffer {
                samples: &samples,
                rate: sr_5khz(),
            })
            .unwrap();

        let mut fp2 = Haitsma::default();
        let f2 = fp2
            .extract(AudioBuffer {
                samples: &samples,
                rate: sr_5khz(),
            })
            .unwrap();

        assert_eq!(f1.frames, f2.frames);
    }

    #[test]
    fn different_signals_diverge() {
        let a = synthetic_audio(0x1111, 5_000 * 3);
        let b = synthetic_audio(0x2222, 5_000 * 3);

        let mut fp = Haitsma::default();
        let fa = fp
            .extract(AudioBuffer {
                samples: &a,
                rate: sr_5khz(),
            })
            .unwrap();
        let fb = fp
            .extract(AudioBuffer {
                samples: &b,
                rate: sr_5khz(),
            })
            .unwrap();
        assert_ne!(fa.frames, fb.frames);
    }

    #[test]
    fn pack_frame_bits_msb_zero_band_layout() {
        // Construct energies so that band 0's delta is positive but bands
        // 1..31 are all zero.
        let mut curr = [0.0_f32; HAITSMA_N_BANDS];
        let prev = [0.0_f32; HAITSMA_N_BANDS];
        // E[curr][0] - E[curr][1] - 0 > 0 → set band 0.
        curr[0] = 1.0;

        let h = pack_frame_bits(&curr, &prev);
        // Band 0 → MSB (bit 31).
        assert_eq!(h, 1 << 31);
    }

    #[test]
    fn band_31_lives_in_the_lsb() {
        // Make band 31's delta positive: E[31] - E[32] > 0 (with all prev zero).
        let mut curr = [0.0_f32; HAITSMA_N_BANDS];
        let prev = [0.0_f32; HAITSMA_N_BANDS];
        curr[31] = 1.0;

        let h = pack_frame_bits(&curr, &prev);
        // Band 31 → bit 0.
        assert_eq!(h, 1);
    }

    #[test]
    fn streaming_latency_matches_n_fft() {
        let s = StreamingHaitsma::default();
        // 2048 samples / 5000 sr * 1000 = 409 ms (integer).
        assert_eq!(s.latency_ms(), 409);
    }

    #[test]
    fn band_lookup_table_covers_in_band_frequencies() {
        let cfg = HaitsmaConfig::default();
        let n_bins = HAITSMA_N_FFT / 2 + 1;
        let lookup = build_bin_to_band(&cfg, n_bins);
        assert_eq!(lookup.len(), n_bins);

        let bin_hz = HAITSMA_SR as f32 / HAITSMA_N_FFT as f32;
        // At least one bin in each band should be tagged.
        let mut hit_per_band = [false; HAITSMA_N_BANDS];
        for &b in &lookup {
            if let Some(b) = b {
                hit_per_band[b as usize] = true;
            }
        }
        for (i, &h) in hit_per_band.iter().enumerate() {
            assert!(h, "band {i} has no FFT bins");
        }

        // Bins outside [fmin, fmax) are None.
        let bin_at_100hz = (100.0 / bin_hz) as usize;
        assert!(
            lookup[bin_at_100hz].is_none(),
            "100 Hz should be below fmin=300"
        );
    }

    #[test]
    fn custom_band_range() {
        let cfg = HaitsmaConfig {
            fmin: 500.0,
            fmax: 1500.0,
        };
        let mut h = Haitsma::new(cfg.clone());
        let samples = synthetic_audio(0xC0FFEE, 5_000 * 3);
        let buf = AudioBuffer {
            samples: &samples,
            rate: sr_5khz(),
        };
        let f = h.extract(buf).unwrap();
        // Should still produce frames; band edges differ but algorithm runs.
        assert!(!f.frames.is_empty());
    }

    #[test]
    #[should_panic(expected = "fmax must exceed fmin")]
    fn invalid_band_range_panics() {
        let _ = Haitsma::new(HaitsmaConfig {
            fmin: 1000.0,
            fmax: 1000.0,
        });
    }

    #[test]
    #[should_panic(expected = "below Nyquist")]
    fn fmax_above_nyquist_panics() {
        let _ = Haitsma::new(HaitsmaConfig {
            fmin: 300.0,
            fmax: 3_000.0,
        });
    }

    #[test]
    fn streaming_offline_equivalence() {
        let samples = synthetic_audio(0xBEEF, 5_000 * 5);

        let mut offline = Haitsma::default();
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: sr_5khz(),
            })
            .unwrap();

        let mut streaming = StreamingHaitsma::default();
        let mut online: Vec<u32> = Vec::new();
        let mut cursor = 0;
        for n in chunk_sizes(0xCAFE, samples.len(), 3_000) {
            let end = cursor + n;
            online.extend(
                streaming
                    .push(&samples[cursor..end])
                    .into_iter()
                    .map(|(_, h)| h),
            );
            cursor = end;
        }
        online.extend(streaming.flush().into_iter().map(|(_, h)| h));

        assert_eq!(off.frames, online, "streaming != offline frame sequence");
    }
}
