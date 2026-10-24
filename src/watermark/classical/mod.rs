//! Keyed classical watermark embedder and detector.
//!
//! Two algorithms, selected by [`Algorithm`]:
//!
//! - [`Algorithm::QimBlock`] (the default) quantises log-magnitude on a
//!   non-overlapping 2048-point DFT with a 6 dB step and four bins per bit.
//! - [`Algorithm::LogHost`] adds a ±3 dB chip to an overlapped log-magnitude
//!   STFT and subtracts each bin's local mean before correlating.
//!
//! Both run at 16 kHz on mono `f32` samples. The embedded bit string is the
//! payload bytes followed by an IEEE CRC-32. [`ClassicalResult::payload`] is
//! `Some` only when that CRC matches, and [`ClassicalResult::detected`]
//! reports that fact. The algorithms do not share a score scale, and no
//! numeric threshold is shipped.
//!
//! Time-scale modification and pitch shift are outside this contract. The
//! Round 2 attack matrix measured raw 64-bit payloads; this API adds a
//! 32-bit CRC, so those counts are not a claim about framed payloads.
//!
//! # What the detector will not guess
//!
//! A bit the detector has no evidence for is an erasure, never a 1, and one
//! erasure means no payload. Digital silence (`0.0` or `-0.0`), a constant
//! offset, and audio so quiet that no carrier bin reaches the embedder's
//! `1e-6` floor therefore decode to `None` for every payload length and key.
//! Resolving such bits to 1 would turn them into an all-ones frame, and
//! `crc32([0xFF; 4]) == 0xFFFF_FFFF` makes that a valid 4-byte payload.
//!
//! Evidence that says the same thing everywhere is no evidence either. A
//! spectrum that is flat across the carrier bins, which a single click in
//! silence has, puts every carrier at one level, and the QIM detector reads
//! one level as one bit value. A bit string of identical bits is therefore
//! never a payload. The one payload that frames to such a string,
//! `[0xFF; 4]`, is refused by [`ClassicalEmbedder::try_new`] rather than
//! embedded and never found.
//!
//! # What it tolerates
//!
//! The embedded format has not changed; the detectors read it more robustly.
//!
//! - **QIM** follows a constant or slowly changing gain of any size,
//!   including the 3 dB steps that swap every bit: the two bit lattices are
//!   3 dB apart, so every marked carrier is on a 3 dB lattice, and the
//!   detector tracks the level modulo 3 dB and tries both polarities against
//!   the CRC. It finds the block alignment to the sample, so a clip may
//!   start anywhere; the alignment is [`ClassicalResult::sync_offset`]. It
//!   skips silent blocks. It does not follow a level that differs between
//!   frequency bands (equalisation) or one that jumps part-way through, and
//!   lossy codecs and added noise cost recovery.
//! - **LogHost** is level-invariant by construction. It subtracts a per-bin
//!   mean over 17 frames around each frame, so a host whose spectrum drifts
//!   no longer drowns the chips, but it needs far more audio than QIM
//!   (below) and still misses some clean round trips on real audio.
//!
//! # Capacity
//!
//! [`Algorithm::QimBlock`] gives every framed bit four of the 492 bins in
//! `20..512`, so it carries at most 11 payload bytes (`8 * 11 + 32 = 120`
//! framed bits). A longer payload is rejected with [`AfpError::Config`] by
//! [`ClassicalEmbedder::try_new`] and [`ClassicalDetector::try_new`].
//! [`Algorithm::LogHost`] has no bin budget; it needs more audio the longer
//! the payload is (below).
//!
//! # Minimum length
//!
//! [`ClassicalEmbedder::embed`] and [`ClassicalDetector::detect`] accept any
//! clip that holds the framed bits once: 2048 samples for QIM, one STFT frame
//! per framed bit for LogHost (3.04 s for 8 bytes). That structural minimum
//! is not a usable one; LogHost recovered none of 30 clean round trips from
//! 3.04 s of real audio. [`ClassicalConfig::min_samples`] is the length to
//! size inputs by: about 4 s for QIM, and about 16 s for an 8-byte LogHost
//! payload, growing with the payload.
//!
//! # What the mark is not
//!
//! The CRC-32 catches accidental damage. It does not authenticate the
//! payload. The key chooses the carrier bins and their order, but it does not
//! hide a QIM mark: the quantisation lattice is public, so the block grid and
//! nearly all of the carrier bins can be picked out of one marked clip
//! without the key. Marks do not stack either. A second embed, whether of
//! another payload or of the same one under another key, quantises bins the
//! first mark also uses, and the first mark did not survive in any of the
//! cases measured. Use the payload as an identifier that survives ordinary
//! processing, not as a secret and not as proof of origin.
//!
//! # Example
//!
//! The host needs spectral content in every block, so a constant level or
//! silence cannot carry a mark. Deterministic noise stands in for a real
//! 16 kHz mono clip here.
//!
//! ```
//! use audiofp::watermark::classical::{ClassicalConfig, ClassicalDetector, ClassicalEmbedder};
//! use audiofp::SampleRate;
//!
//! let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
//! let key = b"secret-key";
//! let config = ClassicalConfig::new(payload.len());
//!
//! let mut x = 0x1234_5678u32;
//! let samples: Vec<f32> = (0..config.min_samples())
//!     .map(|_| {
//!         x ^= x << 13;
//!         x ^= x >> 17;
//!         x ^= x << 5;
//!         (x as i32 as f32 / i32::MAX as f32) * 0.2
//!     })
//!     .collect();
//!
//! let embedder = ClassicalEmbedder::try_new(config, key, &payload)?;
//! let marked = embedder.embed(&samples, SampleRate::HZ_16000)?;
//! assert_ne!(marked, samples, "the embedder changed the audio");
//!
//! let detector = ClassicalDetector::try_new(config, key)?;
//! let found = detector.detect(&marked, SampleRate::HZ_16000)?;
//! assert_eq!(found.payload.as_deref(), Some(payload.as_slice()));
//!
//! // Audio that was never marked decodes to no payload at all.
//! let clean = detector.detect(&samples, SampleRate::HZ_16000)?;
//! assert!(!clean.detected());
//! # Ok::<(), audiofp::AfpError>(())
//! ```

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::{AfpError, Result, SampleRate};

mod coding;
mod keying;
mod log_host;
mod qim;
mod stft;

/// FFT length shared by both algorithms.
pub(crate) const N_FFT: usize = 2048;
/// Hop of the overlapped log-host STFT. Block QIM does not hop.
pub(crate) const HOP: usize = 512;
const SAMPLE_RATE_HZ: u32 = 16_000;

/// What one algorithm's detector found in a clip.
pub(crate) struct Detection {
    /// The payload, when the CRC-32 of the recovered frame matched.
    pub(crate) payload: Option<Vec<u8>>,
    /// The algorithm's statistic at `sync_offset`.
    pub(crate) score: f64,
    /// The alignment the detector kept.
    pub(crate) sync_offset: usize,
}

impl Detection {
    /// No payload and no evidence for one.
    pub(crate) fn none() -> Self {
        Self {
            payload: None,
            score: 0.0,
            sync_offset: 0,
        }
    }
}

/// Which measured watermark to embed and detect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Algorithm {
    /// Non-overlapping 6 dB block QIM. Exact on the Round 2 clean tracks
    /// when the payload was a raw 64-bit string.
    #[default]
    QimBlock,
    /// Overlapped log-magnitude DSSS with per-bin host subtraction.
    LogHost,
}

/// Payload length and algorithm. Sample rate is fixed at 16 kHz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub struct ClassicalConfig {
    algorithm: Algorithm,
    payload_len: usize,
}

impl ClassicalConfig {
    /// Configuration for `payload_len` bytes using [`Algorithm::QimBlock`].
    ///
    /// `payload_len` is checked in [`ClassicalEmbedder::try_new`] and
    /// [`ClassicalDetector::try_new`], not here.
    pub const fn new(payload_len: usize) -> Self {
        Self {
            algorithm: Algorithm::QimBlock,
            payload_len,
        }
    }

    /// Select the algorithm. The default is [`Algorithm::QimBlock`].
    pub const fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Algorithm this config will embed.
    pub const fn algorithm(self) -> Algorithm {
        self.algorithm
    }

    /// Payload length in bytes, not counting the CRC-32 trailer.
    pub const fn payload_len(self) -> usize {
        self.payload_len
    }

    /// Shortest clip, in samples at 16 kHz, this configuration can be
    /// expected to recover its payload from.
    ///
    /// This is not the shortest clip [`ClassicalEmbedder::embed`] and
    /// [`ClassicalDetector::detect`] accept. They take any clip that holds the
    /// framed bits once: 2048 samples for [`Algorithm::QimBlock`], and
    /// `8 * payload_len + 31` STFT hops (3.04 s for 8 bytes) for
    /// [`Algorithm::LogHost`]. That structural minimum is not a usable one.
    /// LogHost then reads each bit from a single frame, and recovered none of
    /// 30 clean round trips from 3.04 s of real audio (13 of 24 from 12 s,
    /// 19 of 21 from 16 s). QIM recovered 27 of 30 from a single clean block,
    /// and 27 of 30 from 4.1 s once the level was moved by 3 dB and the clip
    /// cut 17 samples in.
    ///
    /// The value returned is about 4.2 s for QIM, whatever the payload, and
    /// five STFT frames per framed bit for LogHost (15.4 s for 8 bytes). It
    /// is guidance, not a check: lossy codecs, noise and filtering need more,
    /// and a clip that is mostly silence counts only for the audio in it.
    ///
    /// The count is meaningful for configurations that pass `try_new`; it
    /// saturates rather than overflowing for absurd payload lengths.
    pub const fn min_samples(self) -> usize {
        let n_bits = self
            .payload_len
            .saturating_mul(8)
            .saturating_add(coding::CRC_BITS);
        match self.algorithm {
            Algorithm::QimBlock => qim::usable_min_samples(),
            Algorithm::LogHost => log_host::usable_min_samples(n_bits),
        }
    }
}

/// Embeds one payload under one key.
///
/// `Debug` prints the configuration and redacts the key and the payload.
#[derive(Clone)]
pub struct ClassicalEmbedder {
    config: ClassicalConfig,
    key: Vec<u8>,
    payload: Vec<u8>,
}

impl fmt::Debug for ClassicalEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassicalEmbedder")
            .field("config", &self.config)
            .field("key", &format_args!("<redacted>"))
            .field("payload", &format_args!("<redacted>"))
            .finish()
    }
}

impl ClassicalEmbedder {
    /// Build an embedder.
    ///
    /// # Errors
    ///
    /// - [`AfpError::Config`] — `key` is empty, `payload.len()` is not
    ///   `config.payload_len`, the payload length is zero, a
    ///   [`Algorithm::QimBlock`] payload does not fit the carrier bins, or the
    ///   payload is four `0xFF` bytes. That frame is all ones, which is also
    ///   what a flat spectrum decodes to, so the detector never accepts it.
    pub fn try_new(config: ClassicalConfig, key: &[u8], payload: &[u8]) -> Result<Self> {
        validate(&config, key)?;
        if payload.len() != config.payload_len {
            return Err(config_err("payload length does not match configuration"));
        }
        if coding::frames_to_constant(payload) {
            return Err(config_err(
                "payload frames to a constant bit string, which the detector never accepts",
            ));
        }
        Ok(Self {
            config,
            key: key.to_vec(),
            payload: payload.to_vec(),
        })
    }

    /// Embed `self`'s payload into `samples`.
    ///
    /// The returned buffer has the same length as `samples`. Samples past
    /// the last full QIM block are copied unchanged.
    ///
    /// # Errors
    ///
    /// - [`AfpError::UnsupportedSampleRate`] — `rate` is not 16 kHz.
    /// - [`AfpError::NonFiniteSample`] — `samples` contains NaN or an infinity.
    /// - [`AfpError::AudioTooShort`] — QIM needs 2048 samples, or log-host
    ///   needs one STFT frame per framed bit.
    pub fn embed(&self, samples: &[f32], rate: SampleRate) -> Result<Vec<f32>> {
        let n_bits = check_audio(&self.config, samples, rate)?;
        let bits = coding::frame(&self.payload);
        debug_assert_eq!(bits.len(), n_bits);
        let marked = match self.config.algorithm {
            Algorithm::QimBlock => qim::embed(samples, &bits, &self.key),
            Algorithm::LogHost => log_host::embed(samples, &bits, &self.key),
        };
        Ok(marked)
    }
}

/// Blind detector for one key and one payload length.
///
/// `Debug` prints the configuration and redacts the key.
#[derive(Clone)]
pub struct ClassicalDetector {
    config: ClassicalConfig,
    key: Vec<u8>,
}

impl fmt::Debug for ClassicalDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClassicalDetector")
            .field("config", &self.config)
            .field("key", &format_args!("<redacted>"))
            .finish()
    }
}

impl ClassicalDetector {
    /// Build a detector.
    ///
    /// # Errors
    ///
    /// - [`AfpError::Config`] — `key` is empty, the payload length is zero,
    ///   or a [`Algorithm::QimBlock`] payload does not fit the carrier bins.
    pub fn try_new(config: ClassicalConfig, key: &[u8]) -> Result<Self> {
        validate(&config, key)?;
        Ok(Self {
            config,
            key: key.to_vec(),
        })
    }

    /// Detect a payload in `samples`.
    ///
    /// A missing mark is `Ok` with [`ClassicalResult::payload`] equal to
    /// `None`. It is not an error.
    ///
    /// # Errors
    ///
    /// Same as [`ClassicalEmbedder::embed`].
    pub fn detect(&self, samples: &[f32], rate: SampleRate) -> Result<ClassicalResult> {
        let n_bits = check_audio(&self.config, samples, rate)?;
        let payload_len = self.config.payload_len;
        let found = match self.config.algorithm {
            Algorithm::QimBlock => qim::detect(samples, payload_len, n_bits, &self.key),
            Algorithm::LogHost => log_host::detect(samples, payload_len, n_bits, &self.key),
        };
        Ok(ClassicalResult {
            payload: found.payload,
            score: found.score,
            sync_offset: found.sync_offset,
        })
    }
}

/// Detection outcome. [`Self::detected`] is true exactly when `payload` is `Some`.
#[derive(Clone, Debug, PartialEq)]
#[must_use]
pub struct ClassicalResult {
    /// Recovered payload when its CRC-32 matches, otherwise `None`.
    pub payload: Option<Vec<u8>>,
    /// Algorithm statistic at [`Self::sync_offset`]. Larger means a stronger
    /// mark. QIM reports the mean absolute lattice vote, after removing the
    /// level the detector tracked. Log-host reports the mean absolute
    /// correlation of the winning bit phase. Both are `0.0` when there was
    /// nothing to vote on. The two scales are not comparable, and neither is
    /// a detection threshold.
    pub score: f64,
    /// Alignment the detector kept.
    ///
    /// For [`Algorithm::QimBlock`] this is a sample offset in `0..2048`,
    /// found to the sample: where the first whole 2048-sample block starts.
    /// It is `0` for a clip that begins where the embedder's blocks do, and
    /// `(2048 - n % 2048) % 2048` for one with its first `n` samples removed.
    /// For [`Algorithm::LogHost`] this is the winning bit phase, counted in
    /// frames rather than samples.
    pub sync_offset: usize,
}

impl ClassicalResult {
    /// True when the CRC-32 of the recovered frame matched.
    #[must_use]
    pub fn detected(&self) -> bool {
        self.payload.is_some()
    }
}

fn validate(config: &ClassicalConfig, key: &[u8]) -> Result<()> {
    if key.is_empty() {
        return Err(config_err("key must be non-empty"));
    }
    if config.payload_len == 0 {
        return Err(config_err("payload length must be non-zero"));
    }
    let n_bits = framed_bits(config.payload_len)?;
    if config.algorithm == Algorithm::QimBlock && !qim::fits(n_bits) {
        return Err(config_err("payload does not fit the QIM bin budget"));
    }
    let _ = structural_min_samples(config.algorithm, n_bits)?;
    Ok(())
}

fn check_audio(config: &ClassicalConfig, samples: &[f32], rate: SampleRate) -> Result<usize> {
    if rate.hz() != SAMPLE_RATE_HZ {
        return Err(AfpError::UnsupportedSampleRate(rate.hz()));
    }
    if let Some(index) = samples.iter().position(|s| !s.is_finite()) {
        return Err(AfpError::NonFiniteSample { index });
    }
    let n_bits = framed_bits(config.payload_len)?;
    let needed = structural_min_samples(config.algorithm, n_bits)?;
    if samples.len() < needed {
        return Err(AfpError::AudioTooShort {
            needed,
            got: samples.len(),
        });
    }
    Ok(n_bits)
}

fn framed_bits(payload_len: usize) -> Result<usize> {
    coding::framed_bit_len(payload_len).ok_or_else(|| config_err("payload length overflows"))
}

/// Shortest clip the embedder and detector accept: the framed bits once.
fn structural_min_samples(algorithm: Algorithm, n_bits: usize) -> Result<usize> {
    match algorithm {
        Algorithm::QimBlock => Ok(N_FFT),
        Algorithm::LogHost => {
            if n_bits <= 1 {
                Ok(1)
            } else {
                (n_bits - 1)
                    .checked_mul(HOP)
                    .ok_or_else(|| config_err("payload length overflows"))
            }
        }
    }
}

fn config_err(message: &str) -> AfpError {
    AfpError::Config(String::from(message))
}

/// Helpers shared by the unit tests of this module and its algorithms.
#[cfg(test)]
pub(crate) mod testing {
    use alloc::vec::Vec;

    use super::keying::ChaCha;

    /// Seeded white noise, uniform in ±0.2.
    pub(crate) fn noise(n: usize) -> Vec<f32> {
        let mut rng = ChaCha::new(b"noise-host", 1);
        (0..n).map(|_| rng.bipolar() * 0.2).collect()
    }

    /// Write scale 32767, read scale 32768, matching the bakeoff WAV path.
    pub(crate) fn through_pcm16(samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .map(|&s| {
                let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                f32::from(q) / 32768.0
            })
            .collect()
    }

    /// `samples` scaled by `db` decibels, then through PCM16.
    pub(crate) fn gain_db(samples: &[f32], db: f32) -> Vec<f32> {
        let gain = libm::powf(10.0, db / 20.0);
        let scaled: Vec<f32> = samples.iter().map(|&s| s * gain).collect();
        through_pcm16(&scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{noise, through_pcm16};
    use super::{Algorithm, ClassicalConfig, ClassicalDetector, ClassicalEmbedder};
    use crate::{AfpError, SampleRate};

    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn default_algorithm_is_qim_block() {
        assert_eq!(Algorithm::default(), Algorithm::QimBlock);
        assert_eq!(ClassicalConfig::new(8).algorithm(), Algorithm::QimBlock);
    }

    #[test]
    fn both_algorithms_roundtrip_pcm16_on_noise() {
        let samples = noise(16_000 * 8);
        let payload = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
        let key = b"bakeoff-secret-key-0123456789";
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            let config = ClassicalConfig::new(payload.len()).with_algorithm(algorithm);
            let embedder = ClassicalEmbedder::try_new(config, key, &payload).unwrap();
            let marked = embedder.embed(&samples, SampleRate::HZ_16000).unwrap();
            let decoded = through_pcm16(&marked);
            let detector = ClassicalDetector::try_new(config, key).unwrap();
            let result = detector.detect(&decoded, SampleRate::HZ_16000).unwrap();
            assert_eq!(
                result.payload.as_deref(),
                Some(payload.as_slice()),
                "{algorithm:?}"
            );
            assert!(result.detected());
        }
    }

    /// Extreme but finite samples: a bin's `re² + im²` used to overflow
    /// `f32`, so embedding noise at amplitude 1e18 returned mostly
    /// non-finite samples (QIM 63 488 of 64 000, LogHost all of them).
    #[test]
    fn extreme_finite_amplitude_embeds_stay_finite() {
        let host: Vec<f32> = noise(64_000).iter().map(|&s| s * 5e18).collect();
        let payload = [1, 2, 3, 4];
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            let config = ClassicalConfig::new(payload.len()).with_algorithm(algorithm);
            let embedder = ClassicalEmbedder::try_new(config, b"key", &payload).unwrap();
            let marked = embedder.embed(&host, SampleRate::HZ_16000).unwrap();
            assert!(marked.iter().all(|s| s.is_finite()), "{algorithm:?}");
            let detector = ClassicalDetector::try_new(config, b"key").unwrap();
            let result = detector.detect(&marked, SampleRate::HZ_16000).unwrap();
            assert!(result.score.is_finite(), "{algorithm:?}: {result:?}");
            if algorithm == Algorithm::QimBlock {
                assert_eq!(result.payload.as_deref(), Some(payload.as_slice()));
            }
        }
    }

    #[test]
    fn wrong_key_and_unwatermarked_audio_return_no_payload() {
        let samples = noise(16_000 * 8);
        let payload = [0xABu8; 8];
        let key = b"bakeoff-secret-key-0123456789";
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            let config = ClassicalConfig::new(payload.len()).with_algorithm(algorithm);
            let embedder = ClassicalEmbedder::try_new(config, key, &payload).unwrap();
            let marked = through_pcm16(&embedder.embed(&samples, SampleRate::HZ_16000).unwrap());
            let wrong = ClassicalDetector::try_new(config, b"other-key").unwrap();
            let missed = wrong.detect(&marked, SampleRate::HZ_16000).unwrap();
            assert!(missed.payload.is_none(), "{algorithm:?} wrong key");
            assert!(!missed.detected());
            let plain = ClassicalDetector::try_new(config, key).unwrap();
            let unmarked = plain.detect(&samples, SampleRate::HZ_16000).unwrap();
            assert!(unmarked.payload.is_none(), "{algorithm:?} unmarked");
            assert!(!unmarked.detected());
        }
    }

    #[test]
    fn short_non_finite_and_wrong_rate_are_errors() {
        let payload = [0u8; 8];
        let key = b"key";
        let qim = ClassicalConfig::new(payload.len());
        let embedder = ClassicalEmbedder::try_new(qim, key, &payload).unwrap();
        let short = embedder
            .embed(&[0.0; 100], SampleRate::HZ_16000)
            .unwrap_err();
        assert!(matches!(
            short,
            AfpError::AudioTooShort {
                needed: 2048,
                got: 100
            }
        ));
        let detector = ClassicalDetector::try_new(qim, key).unwrap();
        let short_detect = detector
            .detect(&[0.0; 100], SampleRate::HZ_16000)
            .unwrap_err();
        assert!(matches!(
            short_detect,
            AfpError::AudioTooShort {
                needed: 2048,
                got: 100
            }
        ));

        let mut nan = alloc::vec![0.1f32; 3000];
        nan[3] = f32::NAN;
        let non_finite = embedder.embed(&nan, SampleRate::HZ_16000).unwrap_err();
        assert!(matches!(non_finite, AfpError::NonFiniteSample { index: 3 }));

        let rate = embedder
            .embed(&[0.0; 2048], SampleRate::HZ_8000)
            .unwrap_err();
        assert!(matches!(rate, AfpError::UnsupportedSampleRate(8_000)));

        let log_host = ClassicalConfig::new(8).with_algorithm(Algorithm::LogHost);
        let log_embedder = ClassicalEmbedder::try_new(log_host, key, &payload).unwrap();
        let log_short = log_embedder
            .embed(&[0.0; 1000], SampleRate::HZ_16000)
            .unwrap_err();
        assert!(matches!(
            log_short,
            AfpError::AudioTooShort {
                needed: 48_640,
                got: 1000
            }
        ));
    }

    #[test]
    fn same_inputs_produce_identical_samples() {
        let samples = noise(48_640);
        let payload = [9u8, 8, 7, 6, 5, 4, 3, 2];
        let key = b"bakeoff-secret-key-0123456789";
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            let config = ClassicalConfig::new(payload.len()).with_algorithm(algorithm);
            let embedder = ClassicalEmbedder::try_new(config, key, &payload).unwrap();
            let a = embedder.embed(&samples, SampleRate::HZ_16000).unwrap();
            let b = embedder.embed(&samples, SampleRate::HZ_16000).unwrap();
            assert_eq!(a, b, "{algorithm:?}");
        }
    }

    /// FNV-1a over the little-endian bit patterns of `samples`.
    fn fnv1a(samples: &[f32]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for s in samples {
            for byte in s.to_bits().to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
        }
        h
    }

    /// The embedded format is frozen: detector work must not move a single
    /// output bit. The hashes were recorded on the PR #154 head (`25dff11`)
    /// before any detector change. Update them only for a deliberate format
    /// change, together with the algorithm id.
    #[test]
    fn embed_output_is_bit_identical_to_pr_154_head() {
        const KEY: &[u8] = b"bakeoff-secret-key-0123456789";
        let cases: [(Algorithm, &[u8], usize); 5] = [
            (
                Algorithm::QimBlock,
                &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
                16_000 * 8,
            ),
            (Algorithm::QimBlock, &[0xA5], 5_000),
            (Algorithm::QimBlock, &[0x5A; 11], 20_001),
            (
                Algorithm::LogHost,
                &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
                16_000 * 8,
            ),
            (Algorithm::LogHost, &[0xC3; 12], 70_001),
        ];
        const EXPECTED: [u64; 5] = [
            0x9908_3145_030d_bed5,
            0x3bd2_b4d9_7cc4_f6a4,
            0x36bb_61f2_ff4f_5f5e,
            0xdc83_0b4b_b56e_021d,
            0xf820_0486_1f02_6bbf,
        ];
        let got: Vec<u64> = cases
            .iter()
            .map(|&(algorithm, payload, n)| {
                let config = ClassicalConfig::new(payload.len()).with_algorithm(algorithm);
                let embedder = ClassicalEmbedder::try_new(config, KEY, payload).unwrap();
                fnv1a(&embedder.embed(&noise(n), SampleRate::HZ_16000).unwrap())
            })
            .collect();
        assert_eq!(got, EXPECTED, "embed output changed; got {got:#x?}");
    }

    #[test]
    fn rejects_empty_key_and_qim_payload_that_does_not_fit() {
        let err = ClassicalEmbedder::try_new(ClassicalConfig::new(8), b"", &[0; 8]).unwrap_err();
        assert!(matches!(err, AfpError::Config(_)));
        let err = ClassicalDetector::try_new(ClassicalConfig::new(0), b"key").unwrap_err();
        assert!(matches!(err, AfpError::Config(_)));
        let err =
            ClassicalEmbedder::try_new(ClassicalConfig::new(12), b"key", &[0; 12]).unwrap_err();
        assert!(matches!(err, AfpError::Config(_)));
        assert!(
            ClassicalEmbedder::try_new(
                ClassicalConfig::new(12).with_algorithm(Algorithm::LogHost),
                b"key",
                &[0; 12],
            )
            .is_ok()
        );
    }

    /// `n` samples of silence with one click of height `amp` every `period`
    /// samples, the first at `first`. A click has a flat spectrum, so it puts
    /// every carrier bin at one level.
    fn clicks(n: usize, first: usize, period: usize, amp: f32) -> Vec<f32> {
        let mut samples = vec![0.0; n];
        for at in (first..n).step_by(period) {
            samples[at] = amp;
        }
        samples
    }

    /// Inputs with nothing in the carrier bins for a detector to read: no
    /// signal at all, or a signal that says the same thing in every bin.
    fn degenerate(n: usize) -> Vec<(&'static str, Vec<f32>)> {
        vec![
            ("all-zero", vec![0.0; n]),
            ("negative zero", vec![-0.0; n]),
            ("dc", vec![0.25; n]),
            (
                "near-silent",
                (0..n)
                    .map(|i| if i % 2 == 0 { 1e-9 } else { -1e-9 })
                    .collect(),
            ),
            ("one click", clicks(n, n / 2, n, 0.7)),
            ("a click per block", clicks(n, 100, 2048, 0.5)),
        ]
    }

    /// Regression for the review's F1: silence decoded to the all-ones frame,
    /// and `crc32([0xFF; 4]) == 0xFFFF_FFFF` made that a valid 4-byte payload
    /// for every key. Every payload length QIM can carry, on every kind of
    /// evidence-free input, one block and several.
    #[test]
    fn evidence_free_input_never_decodes_for_any_qim_payload_length() {
        for payload_len in 1..=11 {
            let config = ClassicalConfig::new(payload_len);
            let detector = ClassicalDetector::try_new(config, b"asset-key-42").unwrap();
            for n in [2048, 5000] {
                for (name, samples) in degenerate(n) {
                    let result = detector.detect(&samples, SampleRate::HZ_16000).unwrap();
                    assert!(
                        result.payload.is_none() && !result.detected(),
                        "QIM payload_len={payload_len} {name} n={n}: {:?}",
                        result.payload
                    );
                }
            }
        }
        // The payload lengths past 11 are rejected, not silently truncated.
        let config = ClassicalConfig::new(12);
        assert!(ClassicalDetector::try_new(config, b"asset-key-42").is_err());
    }

    /// Same, for LogHost. Its cost is one STFT per framed bit, so only the
    /// lengths around the CRC collision (4) are run end to end here; the
    /// whole 1..=64 range is covered where the decision is made, in
    /// `log_host::tests`.
    #[test]
    fn evidence_free_input_never_decodes_for_log_host() {
        for payload_len in [1, 4] {
            let config = ClassicalConfig::new(payload_len).with_algorithm(Algorithm::LogHost);
            let detector = ClassicalDetector::try_new(config, b"asset-key-42").unwrap();
            let n = (payload_len * 8 + 32) * 512;
            for (name, samples) in degenerate(n) {
                let result = detector.detect(&samples, SampleRate::HZ_16000).unwrap();
                assert!(
                    result.payload.is_none() && !result.detected(),
                    "LogHost payload_len={payload_len} {name}: {:?}",
                    result.payload
                );
            }
        }
    }

    /// Found by fuzzing the detector. A click in silence is not silence: its
    /// spectrum is flat, so every carrier sits at one level, which the
    /// level-tracking QIM detector read as all-equal bits at any amplitude.
    /// With `payload_len = 4` the all-ones frame is a valid frame, so the click
    /// "contained" `[0xFF; 4]` under every key. The PR head before the review
    /// fixes returned the same payload for the same clips.
    #[test]
    fn a_click_in_silence_decodes_to_no_payload_at_any_level() {
        let detector =
            ClassicalDetector::try_new(ClassicalConfig::new(4), b"asset-key-42").unwrap();
        for amp in [0.05, 0.12, 0.355, 0.5, 0.708, 0.9, 1.0, 1.41, 8.0] {
            for samples in [
                clicks(2048 * 40, 2048 * 20 + 17, 2048 * 40, amp),
                clicks(16_000 * 6, 100, 16_000, amp),
            ] {
                let result = detector.detect(&samples, SampleRate::HZ_16000).unwrap();
                assert!(
                    result.payload.is_none() && !result.detected(),
                    "click of {amp}: {:?}",
                    result.payload
                );
            }
        }
        let config = ClassicalConfig::new(4).with_algorithm(Algorithm::LogHost);
        let detector = ClassicalDetector::try_new(config, b"asset-key-42").unwrap();
        for amp in [0.05, 0.708, 1.41] {
            let samples = clicks(16_000 * 3, 100, 16_000, amp);
            let result = detector.detect(&samples, SampleRate::HZ_16000).unwrap();
            assert!(result.payload.is_none(), "LogHost click of {amp}");
        }
    }

    /// The other side of the same rule: the one payload whose frame is a
    /// constant string would embed and then never be found, so the embedder
    /// refuses it. Its neighbours are fine.
    #[test]
    fn embedder_refuses_the_payload_whose_frame_is_constant() {
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            let config = ClassicalConfig::new(4).with_algorithm(algorithm);
            let err = ClassicalEmbedder::try_new(config, b"key", &[0xFF; 4]).unwrap_err();
            assert!(matches!(err, AfpError::Config(_)), "{err:?}");
            for payload in [[0xFF, 0xFF, 0xFF, 0xFE], [0x7F; 4], [0; 4]] {
                assert!(ClassicalEmbedder::try_new(config, b"key", &payload).is_ok());
            }
            assert!(ClassicalDetector::try_new(config, b"key").is_ok());
        }
        // Every other payload length is untouched.
        for len in [1, 2, 3, 5, 8, 11] {
            let config = ClassicalConfig::new(len);
            assert!(ClassicalEmbedder::try_new(config, b"key", &vec![0xFF; len]).is_ok());
        }
    }

    #[test]
    fn debug_output_redacts_the_key_and_the_payload() {
        let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let key = b"hunter2-secret";
        let config = ClassicalConfig::new(payload.len());
        let embedder = ClassicalEmbedder::try_new(config, key, &payload).unwrap();
        let detector = ClassicalDetector::try_new(config, key).unwrap();
        assert_eq!(
            format!("{embedder:?}"),
            "ClassicalEmbedder { config: ClassicalConfig { algorithm: QimBlock, payload_len: 8 }, \
             key: <redacted>, payload: <redacted> }"
        );
        assert_eq!(
            format!("{detector:?}"),
            "ClassicalDetector { config: ClassicalConfig { algorithm: QimBlock, payload_len: 8 }, \
             key: <redacted> }"
        );
        // The only number either may print is the payload length.
        for text in [format!("{embedder:#?}"), format!("{detector:#?}")] {
            assert!(text.contains("<redacted>"), "{text}");
            assert!(!text.contains("hunter2"), "{text}");
            assert!(
                text.chars().filter(char::is_ascii_digit).all(|c| c == '8'),
                "{text}"
            );
        }
    }

    #[test]
    fn min_samples_is_a_usable_length_above_the_structural_minimum() {
        for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
            for payload_len in 1..=11 {
                let config = ClassicalConfig::new(payload_len).with_algorithm(algorithm);
                let n_bits = payload_len * 8 + 32;
                let structural = super::structural_min_samples(algorithm, n_bits).unwrap();
                assert!(
                    config.min_samples() > structural,
                    "{algorithm:?} {payload_len}: {} <= {structural}",
                    config.min_samples()
                );
            }
        }
        // QIM does not need more audio for a longer payload; LogHost does.
        let qim = |len| ClassicalConfig::new(len).min_samples();
        assert_eq!(qim(1), qim(11));
        let log_host = |len| {
            ClassicalConfig::new(len)
                .with_algorithm(Algorithm::LogHost)
                .min_samples()
        };
        assert!(log_host(12) > log_host(8) && log_host(8) > log_host(1));
        // About 4 s for QIM, and about 16 s for an 8-byte LogHost payload,
        // against the 3.04 s the API itself accepts there.
        assert_eq!(qim(8), 67_584);
        assert_eq!(log_host(8), 245_760);
        // `const`, and saturating rather than overflowing.
        const EIGHT: usize = ClassicalConfig::new(8).min_samples();
        assert_eq!(EIGHT, 67_584);
        let huge = ClassicalConfig::new(usize::MAX).with_algorithm(Algorithm::LogHost);
        assert_eq!(huge.min_samples(), usize::MAX);
    }
}
