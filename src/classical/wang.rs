//! Wang-style landmark fingerprinter.
//!
//! The algorithm (Avery Wang, "An Industrial-Strength Audio Search
//! Algorithm", 2003 — the "Shazam paper"):
//!
//! 1. Resample the input to 8 kHz mono *(caller's responsibility)*.
//! 2. Take a Hann-windowed STFT with `n_fft = 1024`, `hop = 128` →
//!    62.5 frames/s, 513 frequency bins.
//! 3. Convert the magnitude spectrogram to dB log-magnitude.
//! 4. Pick spectral peaks in a 31×31 neighbourhood, capped at 30/s.
//! 5. For each anchor peak, take the strongest `fan_out` peaks within
//!    `Δt ∈ [1, target_zone_t]` and `|Δf| ≤ target_zone_f`; pack each
//!    `(anchor, target)` pair into a 32-bit hash.
//!
//! Hash layout (high to low bit):
//! ```text
//! [31..23]  f_a_q  (9 bits, anchor frequency, quantised to 512 buckets)
//! [22..14]  f_b_q  (9 bits, target frequency, same quantisation)
//! [13.. 0]  Δt     (14 bits, frames between anchor and target, clamped 1..=16383)
//! ```

use alloc::vec::Vec;

use libm::log10f;

use crate::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;
use crate::{
    AfpError, AudioBuffer, Fingerprinter, Result, SampleRate, StreamingFingerprinter, TimestampMs,
};

/// One anchor-target landmark pair packed into a 32-bit hash.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WangHash {
    /// 32-bit hash: `f_a_q (9) | f_b_q (9) | Δt (14)`, MSB first.
    pub hash: u32,
    /// STFT frame index of the anchor peak.
    pub t_anchor: u32,
}

/// All hashes produced by [`Wang`] over an audio buffer.
#[derive(Clone, Debug)]
pub struct WangFingerprint {
    /// Hashes sorted by `(t_anchor, hash)`.
    pub hashes: Vec<WangHash>,
    /// Frame rate of the underlying STFT — always 62.5 for `wang-v1`
    /// (`8000 / 128`).
    pub frames_per_sec: f32,
}

/// Tunable parameters for [`Wang`].
#[derive(Clone, Debug)]
pub struct WangConfig {
    /// `F`: target peaks paired with each anchor. Default 10; embedded
    /// builds typically lower this to 5.
    pub fan_out: u16,
    /// Maximum `Δt` (frames) between anchor and target. Default 63.
    pub target_zone_t: u16,
    /// Maximum `|Δf|` (FFT bins) between anchor and target. Default 64.
    pub target_zone_f: u16,
    /// Per-second cap on peak count. Default 30.
    pub peaks_per_sec: u16,
    /// Magnitude floor (dB) below which peaks are ignored. Default −50.
    pub min_anchor_mag_db: f32,
}

impl Default for WangConfig {
    fn default() -> Self {
        Self {
            fan_out: 10,
            target_zone_t: 63,
            target_zone_f: 64,
            peaks_per_sec: 30,
            min_anchor_mag_db: -50.0,
        }
    }
}

const WANG_N_FFT: usize = 1024;
const WANG_HOP: usize = 128;
const WANG_SR: u32 = 8_000;
const WANG_FRAMES_PER_SEC: f32 = WANG_SR as f32 / WANG_HOP as f32;
/// Quantisation buckets for the 9-bit frequency field.
const WANG_FREQ_BUCKETS: u32 = 512;
const WANG_PEAK_NEIGHBOURHOOD: usize = 15;
const WANG_LOG_FLOOR: f32 = 1e-6;

/// Wang offline fingerprinter.
///
/// # Example
///
/// ```
/// use afp::{AudioBuffer, Fingerprinter, SampleRate};
/// use afp::classical::Wang;
///
/// let mut fp = Wang::default();
/// // 3 seconds of silence — produces an empty fingerprint, not an error.
/// let samples = vec![0.0_f32; 8_000 * 3];
/// let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
/// let fpr = fp.extract(buf).unwrap();
/// assert_eq!(fpr.frames_per_sec, 62.5);
/// assert!(fpr.hashes.is_empty());
/// ```
pub struct Wang {
    cfg: WangConfig,
    stft: ShortTimeFFT,
}

impl Default for Wang {
    fn default() -> Self {
        Self::new(WangConfig::default())
    }
}

impl Wang {
    /// Build a Wang extractor with the given config.
    #[must_use]
    pub fn new(cfg: WangConfig) -> Self {
        let stft = ShortTimeFFT::new(StftConfig {
            n_fft: WANG_N_FFT,
            hop: WANG_HOP,
            window: WindowKind::Hann,
            // No reflect-padding: hashes are most stable when the first
            // frame starts at sample 0 of the input buffer.
            center: false,
        });
        Self { cfg, stft }
    }
}

impl Fingerprinter for Wang {
    type Output = WangFingerprint;
    type Config = WangConfig;

    fn name(&self) -> &'static str {
        "wang-v1"
    }

    fn config(&self) -> &Self::Config {
        &self.cfg
    }

    fn required_sample_rate(&self) -> u32 {
        WANG_SR
    }

    fn min_samples(&self) -> usize {
        WANG_SR as usize * 2
    }

    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output> {
        if audio.rate.hz() != WANG_SR {
            return Err(AfpError::UnsupportedSampleRate(audio.rate.hz()));
        }
        if audio.samples.len() < self.min_samples() {
            return Err(AfpError::AudioTooShort {
                needed: self.min_samples(),
                got: audio.samples.len(),
            });
        }

        let spec = self.stft.magnitude(audio.samples);
        let n_frames = spec.len();
        if n_frames == 0 {
            return Ok(WangFingerprint {
                hashes: Vec::new(),
                frames_per_sec: WANG_FRAMES_PER_SEC,
            });
        }
        let n_bins = self.stft.n_bins();

        // Flatten to a row-major dB log-magnitude grid for the peak picker.
        let mut log_spec = Vec::with_capacity(n_frames * n_bins);
        for frame in &spec {
            for &m in frame {
                log_spec.push(20.0 * log10f(m.max(WANG_LOG_FLOOR)));
            }
        }

        let picker = PeakPicker::new(PeakPickerConfig {
            neighborhood_t: WANG_PEAK_NEIGHBOURHOOD,
            neighborhood_f: WANG_PEAK_NEIGHBOURHOOD,
            // The "magnitude" field of `Peak` is whatever we feed in here,
            // so a dB floor goes through unchanged.
            min_magnitude: self.cfg.min_anchor_mag_db,
            target_per_sec: self.cfg.peaks_per_sec as usize,
        });
        let peaks = picker.pick(&log_spec, n_frames, n_bins, WANG_FRAMES_PER_SEC);

        let mut hashes = build_hashes(&peaks, &self.cfg);
        // Stable, deterministic ordering for round-trip and golden tests.
        hashes.sort_unstable_by_key(|h| (h.t_anchor, h.hash));

        Ok(WangFingerprint {
            hashes,
            frames_per_sec: WANG_FRAMES_PER_SEC,
        })
    }
}

/// Walk `peaks` (sorted by `(t_frame, f_bin)`) and emit landmark hashes.
fn build_hashes(peaks: &[Peak], cfg: &WangConfig) -> Vec<WangHash> {
    let mut hashes = Vec::with_capacity(peaks.len() * cfg.fan_out as usize);
    let target_zone_t = cfg.target_zone_t as i32;
    let target_zone_f = cfg.target_zone_f as i32;
    let fan_out = cfg.fan_out as usize;

    let mut targets: Vec<&Peak> = Vec::with_capacity(64);

    for (i, anchor) in peaks.iter().enumerate() {
        targets.clear();
        for target in &peaks[i + 1..] {
            let dt = target.t_frame as i32 - anchor.t_frame as i32;
            if dt < 1 {
                continue;
            }
            if dt > target_zone_t {
                // Peaks are sorted by t_frame, so once we exceed the zone
                // for this anchor, no later peak can fit either.
                break;
            }
            let df = target.f_bin as i32 - anchor.f_bin as i32;
            if df.abs() > target_zone_f {
                continue;
            }
            targets.push(target);
        }

        // Strongest first; tiebreak on (t, f) for determinism.
        targets.sort_unstable_by(|a, b| {
            b.mag
                .partial_cmp(&a.mag)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| (a.t_frame, a.f_bin).cmp(&(b.t_frame, b.f_bin)))
        });
        targets.truncate(fan_out);

        for target in &targets {
            let f_a_q = quantise_freq(anchor.f_bin);
            let f_b_q = quantise_freq(target.f_bin);
            let dt = ((target.t_frame - anchor.t_frame) & 0x3FFF).max(1);
            let hash = ((f_a_q & 0x1FF) << 23) | ((f_b_q & 0x1FF) << 14) | (dt & 0x3FFF);
            hashes.push(WangHash {
                hash,
                t_anchor: anchor.t_frame,
            });
        }
    }
    hashes
}

/// FFT gives 513 bins; pack into 9 bits (512 buckets) per spec.
#[inline]
fn quantise_freq(bin: u16) -> u32 {
    (bin as u32 * WANG_FREQ_BUCKETS) / 513
}

/// Streaming Wang fingerprinter.
///
/// Buffers audio internally and emits each [`WangHash`] only once its
/// anchor's full target zone has been observed. The output hash multiset
/// matches what [`Wang::extract`] would produce for the same total input.
///
/// **Implementation note:** the current version reruns the offline
/// pipeline on each push to guarantee bit-exact offline parity. This is
/// quadratic in stream length; an incremental implementation is on the
/// roadmap. For now, prefer [`Wang`] for batch jobs and reserve this
/// variant for short-running captures.
///
/// # Example
///
/// ```
/// use afp::{SampleRate, StreamingFingerprinter};
/// use afp::classical::StreamingWang;
///
/// let mut s = StreamingWang::default();
/// // Feed 4 seconds of silence in two chunks; nothing should be emitted.
/// let zeros = vec![0.0_f32; 8_000 * 2];
/// assert!(s.push(&zeros).is_empty());
/// assert!(s.push(&zeros).is_empty());
/// assert!(s.flush().is_empty());
/// ```
pub struct StreamingWang {
    cfg: WangConfig,
    accumulated: Vec<f32>,
    /// First frame index whose anchors have *not* yet been emitted.
    next_anchor_frame: u32,
}

impl Default for StreamingWang {
    fn default() -> Self {
        Self::new(WangConfig::default())
    }
}

impl StreamingWang {
    /// Build a streaming Wang extractor with the given config.
    #[must_use]
    pub fn new(cfg: WangConfig) -> Self {
        Self {
            cfg,
            accumulated: Vec::new(),
            next_anchor_frame: 0,
        }
    }

    /// Borrow the configuration this stream was built with.
    #[must_use]
    pub fn config(&self) -> &WangConfig {
        &self.cfg
    }

    /// Number of complete STFT frames currently buffered.
    fn frames_buffered(&self) -> u32 {
        if self.accumulated.len() < WANG_N_FFT {
            0
        } else {
            ((self.accumulated.len() - WANG_N_FFT) / WANG_HOP + 1) as u32
        }
    }

    /// Frames an anchor must have *after* it before its hashes are stable:
    ///
    /// - `target_zone_t` so all candidate targets are visible,
    /// - `WANG_PEAK_NEIGHBOURHOOD` so the latest target is past the peak
    ///   picker's confirmation latency,
    /// - one full second so the per-second adaptive threshold has seen
    ///   every peak that competes for the bucket. Without this, peaks at
    ///   the tail briefly survive only to be culled when later peaks
    ///   arrive in the same bucket.
    fn lookahead_frames(&self) -> u32 {
        self.cfg.target_zone_t as u32
            + WANG_PEAK_NEIGHBOURHOOD as u32
            + WANG_FRAMES_PER_SEC.ceil() as u32
    }

    /// Run the offline pipeline and return hashes whose anchor frame is
    /// in `[next_anchor_frame, cutoff)`, advancing `next_anchor_frame`.
    fn drain_up_to(&mut self, cutoff: u32) -> Vec<(TimestampMs, WangHash)> {
        if cutoff <= self.next_anchor_frame {
            return Vec::new();
        }

        let mut wang = Wang::new(self.cfg.clone());
        let audio = AudioBuffer {
            samples: &self.accumulated,
            rate: SampleRate::HZ_8000,
        };
        let result = match wang.extract(audio) {
            Ok(r) => r,
            // Not enough audio yet — wait for the next push.
            Err(_) => return Vec::new(),
        };

        let mut emitted = Vec::with_capacity(result.hashes.len());
        for h in result.hashes {
            if h.t_anchor >= self.next_anchor_frame && h.t_anchor < cutoff {
                let t_ms = (h.t_anchor as u64 * WANG_HOP as u64 * 1000) / WANG_SR as u64;
                emitted.push((TimestampMs(t_ms), h));
            }
        }
        self.next_anchor_frame = cutoff;
        emitted
    }
}

impl StreamingFingerprinter for StreamingWang {
    type Frame = WangHash;

    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)> {
        self.accumulated.extend_from_slice(samples);
        let frames = self.frames_buffered();
        let cutoff = frames.saturating_sub(self.lookahead_frames());
        self.drain_up_to(cutoff)
    }

    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)> {
        // Emit everything still pending. Using u32::MAX as the cutoff
        // guarantees we drain to the end of the buffered audio.
        self.drain_up_to(u32::MAX)
    }

    fn latency_ms(&self) -> u32 {
        (self.lookahead_frames() * WANG_HOP as u32 * 1000) / WANG_SR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleRate;
    use alloc::vec;
    use core::f32::consts::PI;

    fn synthetic_audio(seed: u32, len: usize) -> Vec<f32> {
        // Two-tone with low-amplitude noise: stable across runs (no rng),
        // but rich enough to produce many peaks.
        let mut out = Vec::with_capacity(len);
        let mut x: u32 = seed.max(1);
        for n in 0..len {
            // xorshift32 — deterministic noise.
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
            let t = n as f32 / 8_000.0;
            let s = 0.5 * libm::sinf(2.0 * PI * 880.0 * t)
                + 0.3 * libm::sinf(2.0 * PI * 1320.0 * t)
                + noise;
            out.push(s);
        }
        out
    }

    #[test]
    fn rejects_wrong_sample_rate() {
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 16_000];
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
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 8_000]; // 1 second, need 2
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        match fp.extract(buf) {
            Err(AfpError::AudioTooShort {
                needed: 16_000,
                got: 8_000,
            }) => {}
            other => panic!("expected AudioTooShort, got {other:?}"),
        }
    }

    #[test]
    fn silence_gives_empty_fingerprint() {
        let mut fp = Wang::default();
        let samples = vec![0.0_f32; 8_000 * 3];
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let fpr = fp.extract(buf).unwrap();
        assert_eq!(fpr.frames_per_sec, 62.5);
        assert!(fpr.hashes.is_empty());
    }

    #[test]
    fn synthetic_signal_produces_hashes() {
        let mut fp = Wang::default();
        let samples = synthetic_audio(0xC0FFEE, 8_000 * 5);
        let buf = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let fpr = fp.extract(buf).unwrap();
        assert!(!fpr.hashes.is_empty(), "expected hashes from a 5s tone");
        // Ordering invariant.
        for w in fpr.hashes.windows(2) {
            assert!((w[0].t_anchor, w[0].hash) <= (w[1].t_anchor, w[1].hash));
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let samples = synthetic_audio(0xDEAD, 8_000 * 4);

        let mut fp1 = Wang::default();
        let buf1 = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let f1 = fp1.extract(buf1).unwrap();

        let mut fp2 = Wang::default();
        let buf2 = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let f2 = fp2.extract(buf2).unwrap();

        assert_eq!(f1.hashes.len(), f2.hashes.len());
        for (a, b) in f1.hashes.iter().zip(f2.hashes.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn different_signals_diverge() {
        let samples_a = synthetic_audio(0x1111, 8_000 * 3);
        let samples_b = synthetic_audio(0x2222, 8_000 * 3);

        let mut fp = Wang::default();
        let fa = fp
            .extract(AudioBuffer {
                samples: &samples_a,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        let fb = fp
            .extract(AudioBuffer {
                samples: &samples_b,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();
        // Different noise streams must yield non-identical hash sequences.
        assert_ne!(fa.hashes, fb.hashes);
    }

    #[test]
    fn hash_packing_round_trips() {
        // Smoke: feed a known peak set and verify hash-field decode.
        // Build fake peaks: one anchor, one target inside zone.
        let peaks = alloc::vec![
            Peak {
                t_frame: 100,
                f_bin: 50,
                _pad: 0,
                mag: -10.0
            },
            Peak {
                t_frame: 110,
                f_bin: 70,
                _pad: 0,
                mag: -12.0
            },
        ];
        let cfg = WangConfig::default();
        let hashes = build_hashes(&peaks, &cfg);
        assert_eq!(hashes.len(), 1);
        let h = hashes[0].hash;
        // Decode
        let f_a_q = (h >> 23) & 0x1FF;
        let f_b_q = (h >> 14) & 0x1FF;
        let dt = h & 0x3FFF;
        assert_eq!(f_a_q, quantise_freq(50));
        assert_eq!(f_b_q, quantise_freq(70));
        assert_eq!(dt, 10);
        assert_eq!(hashes[0].t_anchor, 100);
    }

    #[test]
    fn streaming_latency_matches_lookahead() {
        let s = StreamingWang::default();
        // (63 target_zone + 15 picker + 63 adaptive bucket) * 128 / 8000 ≈ 2256 ms.
        assert_eq!(s.latency_ms(), 2_256);
    }

    #[test]
    fn streaming_empty_push_is_empty() {
        let mut s = StreamingWang::default();
        assert!(s.push(&[]).is_empty());
        assert!(s.flush().is_empty());
    }

    #[test]
    fn streaming_silence_emits_nothing() {
        let mut s = StreamingWang::default();
        let zeros = vec![0.0_f32; 8_000 * 4];
        assert!(s.push(&zeros).is_empty());
        assert!(s.flush().is_empty());
    }

    /// xorshift32 → split into deterministic pseudo-random chunk sizes.
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

    #[test]
    fn streaming_offline_equivalence() {
        let samples = synthetic_audio(0xBEEF, 8_000 * 6);

        // Offline reference.
        let mut offline = Wang::default();
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        // Streaming with random chunks.
        let mut streaming = StreamingWang::default();
        let mut online = Vec::new();
        let mut cursor = 0;
        for n in chunk_sizes(0xCAFE, samples.len(), 4_000) {
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

        // Same multiset of hashes.
        let mut a: Vec<WangHash> = off.hashes;
        let mut b: Vec<WangHash> = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        assert_eq!(a.len(), b.len(), "hash count mismatch");
        assert_eq!(a, b, "hash sequences differ");
    }

    #[test]
    fn smaller_fan_out_yields_fewer_hashes() {
        let samples = synthetic_audio(0xFEED, 8_000 * 4);
        let buf_a = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };
        let buf_b = AudioBuffer {
            samples: &samples,
            rate: SampleRate::HZ_8000,
        };

        let mut wide = Wang::new(WangConfig {
            fan_out: 10,
            ..WangConfig::default()
        });
        let mut narrow = Wang::new(WangConfig {
            fan_out: 3,
            ..WangConfig::default()
        });
        let f_wide = wide.extract(buf_a).unwrap();
        let f_narrow = narrow.extract(buf_b).unwrap();
        assert!(
            f_narrow.hashes.len() < f_wide.hashes.len(),
            "narrow={} wide={}",
            f_narrow.hashes.len(),
            f_wide.hashes.len(),
        );
    }

    #[test]
    fn quantise_freq_covers_full_range() {
        // Bin 0 maps to bucket 0; bin 512 (≈ Nyquist - 1 step) ≈ bucket 511.
        assert_eq!(quantise_freq(0), 0);
        assert!(quantise_freq(512) < WANG_FREQ_BUCKETS);
        // Quantisation is monotonic non-decreasing.
        let mut prev = 0;
        for b in 0..513_u16 {
            let q = quantise_freq(b);
            assert!(q >= prev);
            assert!(q < WANG_FREQ_BUCKETS);
            prev = q;
        }
    }

    #[test]
    fn streaming_with_one_sample_chunks_still_matches_offline() {
        let samples = synthetic_audio(0xABCD, 8_000 * 3);
        let mut offline = Wang::default();
        let off = offline
            .extract(AudioBuffer {
                samples: &samples,
                rate: SampleRate::HZ_8000,
            })
            .unwrap();

        let mut s = StreamingWang::default();
        let mut online = Vec::new();
        // Push one sample at a time — pathological case for any incremental
        // streaming impl.
        for &sample in &samples {
            online.extend(s.push(&[sample]).into_iter().map(|(_, h)| h));
        }
        online.extend(s.flush().into_iter().map(|(_, h)| h));

        let mut a = off.hashes;
        let mut b = online;
        a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
        assert_eq!(a, b);
    }

    #[test]
    fn target_zone_filters_far_peaks() {
        let peaks = alloc::vec![
            Peak {
                t_frame: 0,
                f_bin: 100,
                _pad: 0,
                mag: 0.0
            },
            // Same time → skipped (Δt < 1).
            Peak {
                t_frame: 0,
                f_bin: 200,
                _pad: 0,
                mag: 0.0
            },
            // Δt = 70 > target_zone_t (63) → skipped.
            Peak {
                t_frame: 70,
                f_bin: 100,
                _pad: 0,
                mag: 0.0
            },
            // Inside zone.
            Peak {
                t_frame: 5,
                f_bin: 110,
                _pad: 0,
                mag: 0.0
            },
            // |Δf| = 200 > 64 → skipped.
            Peak {
                t_frame: 5,
                f_bin: 300,
                _pad: 0,
                mag: 0.0
            },
        ];
        // Note: peaks vec must be sorted by (t_frame, f_bin) for the
        // "break on dt > zone" optimisation to fire correctly.
        let mut sorted = peaks;
        sorted.sort_unstable_by_key(|p| (p.t_frame, p.f_bin));

        let cfg = WangConfig::default();
        let hashes = build_hashes(&sorted, &cfg);
        // Anchor at (0,100) should pair with (5,110) only; anchor at (0,200)
        // can pair with (5,110) (|Δf|=90 — wait that's > 64), or (5,300)
        // (|Δf|=100 > 64). Neither fits → no hash from anchor (0,200).
        // From (5,110) onwards, no later peaks fit any anchor.
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes[0].t_anchor, 0);
    }
}
