//! Keyed classical watermark embedder and detector.
//!
//! Two algorithms, selected by [`Algorithm`]:
//!
//! - [`Algorithm::QimBlock`] (the default) quantises log-magnitude on a
//!   non-overlapping 2048-point DFT with a 6 dB step and four bins per bit.
//! - [`Algorithm::LogHost`] adds a ±3 dB chip to an overlapped log-magnitude
//!   STFT and subtracts each bin's mean before correlating.
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
//! # Example
//!
//! ```no_run
//! use audiofp::watermark::classical::{ClassicalConfig, ClassicalDetector, ClassicalEmbedder};
//! use audiofp::SampleRate;
//!
//! let payload = [1u8, 2, 3, 4, 5, 6, 7, 8];
//! let key = b"secret-key";
//! let config = ClassicalConfig::new(payload.len());
//! let embedder = ClassicalEmbedder::try_new(config, key, &payload)?;
//! let samples = vec![0.1f32; 16_000];
//! let marked = embedder.embed(&samples, SampleRate::HZ_16000)?;
//! let detector = ClassicalDetector::try_new(config, key)?;
//! let result = detector.detect(&marked, SampleRate::HZ_16000)?;
//! if result.detected() {
//!     assert_eq!(result.payload.as_deref(), Some(payload.as_slice()));
//! }
//! # Ok::<(), audiofp::AfpError>(())
//! ```

use alloc::string::String;
use alloc::vec::Vec;

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
}

/// Embeds one payload under one key.
#[derive(Clone, Debug)]
pub struct ClassicalEmbedder {
    config: ClassicalConfig,
    key: Vec<u8>,
    payload: Vec<u8>,
}

impl ClassicalEmbedder {
    /// Build an embedder.
    ///
    /// # Errors
    ///
    /// - [`AfpError::Config`] — `key` is empty, `payload.len()` is not
    ///   `config.payload_len`, the payload length is zero, or a
    ///   [`Algorithm::QimBlock`] payload does not fit the carrier bins.
    pub fn try_new(config: ClassicalConfig, key: &[u8], payload: &[u8]) -> Result<Self> {
        validate(&config, key)?;
        if payload.len() != config.payload_len {
            return Err(config_err("payload length does not match configuration"));
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
#[derive(Clone, Debug)]
pub struct ClassicalDetector {
    config: ClassicalConfig,
    key: Vec<u8>,
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
        let (bits, score, sync_offset) = match self.config.algorithm {
            Algorithm::QimBlock => qim::detect(samples, n_bits, &self.key),
            Algorithm::LogHost => log_host::detect(samples, n_bits, &self.key),
        };
        let payload = coding::unframe(&bits, self.config.payload_len);
        Ok(ClassicalResult {
            payload,
            score,
            sync_offset,
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
    /// mark. QIM reports the mean absolute lattice vote. Log-host reports
    /// the mean absolute correlation of the winning bit phase. The two
    /// scales are not comparable, and neither is a detection threshold.
    pub score: f64,
    /// Alignment the detector kept.
    ///
    /// For [`Algorithm::QimBlock`] this is a sample offset, a multiple of 64
    /// searched inside one FFT block. For [`Algorithm::LogHost`] this is the
    /// winning bit phase, counted in frames rather than samples.
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
    let _ = min_samples(config.algorithm, n_bits)?;
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
    let needed = min_samples(config.algorithm, n_bits)?;
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

fn min_samples(algorithm: Algorithm, n_bits: usize) -> Result<usize> {
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

#[cfg(test)]
mod tests {
    use super::{Algorithm, ClassicalConfig, ClassicalDetector, ClassicalEmbedder};
    use crate::watermark::classical::keying::ChaCha;
    use crate::{AfpError, SampleRate};

    use alloc::vec::Vec;

    fn noise(n: usize) -> Vec<f32> {
        let mut rng = ChaCha::new(b"noise-host", 1);
        (0..n).map(|_| rng.bipolar() * 0.2).collect()
    }

    /// Write scale 32767, read scale 32768, matching the bakeoff WAV path.
    fn through_pcm16(samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .map(|&s| {
                let q = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
                f32::from(q) / 32768.0
            })
            .collect()
    }

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
}
