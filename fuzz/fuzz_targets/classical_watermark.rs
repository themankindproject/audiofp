#![no_main]

//! Fuzz the keyed classical watermark embedder and detector.
//!
//! The contract this target pins: no key, payload length, payload, sample
//! rate or clip may panic or abort `try_new`, `embed`, `detect` or
//! `min_samples`. Three promises the documentation makes must also hold on
//! every input:
//!
//! 1. **The result is coherent.** `detected()` is exactly `payload.is_some()`,
//!    a payload has the configured length, and on audio of ordinary
//!    amplitude the score is finite and non-negative.
//! 2. **`AudioTooShort` names a length that is enough.** Retrying with the
//!    `needed` samples it reports must not fail for length again.
//! 3. **Audio that was never marked never decodes.** The clips built here
//!    carry no mark, so for any key and payload length `detect` must return
//!    `payload: None` on them, and on any clip whose samples are all equal.
//!    The chance that noise passes a CRC-32 is about 2^-31, far below what a
//!    fuzz run can reach, so a hit means the detector read structure that was
//!    not there. (`crc32([0xFF; 4]) == 0xFFFF_FFFF` once made silence, and
//!    then a single click, "contain" a 4-byte payload.)
//!
//! Arbitrary `Vec<f32>` inputs would almost never reach the algorithms'
//! minimum lengths under the smoke run's `max_len`, so the clip is built from
//! a few typed pieces (silence, a constant, noise, a square wave) and only
//! `Raw` carries literal floats, for NaN, infinities and subnormals.

use std::num::NonZeroU32;

use arbitrary::{Arbitrary, Unstructured};
use audiofp::watermark::classical::{
    Algorithm, ClassicalConfig, ClassicalDetector, ClassicalEmbedder, ClassicalResult,
};
use audiofp::{AfpError, SampleRate};
use libfuzzer_sys::fuzz_target;

/// Longest clip built: 5 s at 16 kHz. It holds any QIM payload and the LogHost
/// frames for payloads up to 15 bytes, and keeps one input to a few tens of
/// milliseconds.
const MAX_SAMPLES: usize = 80_000;
/// Samples at or below this magnitude are ordinary audio for the finiteness
/// checks; larger finite values overflow the transforms' intermediate sums.
const TAME: f32 = 1.0e6;

#[derive(Arbitrary, Debug)]
enum Piece {
    /// `+0.0`.
    Silence(u16),
    /// `-0.0`.
    NegativeZero(u16),
    /// A constant level.
    Constant { len: u16, level: i16 },
    /// Xorshift noise with peak `10^(peak / 16 - 9)`, from 1e-9 to 1e6.
    Noise { len: u16, seed: u32, peak: u8 },
    /// A square wave of `period + 2` samples.
    Square { len: u16, period: u8, peak: u8 },
    /// Literal samples.
    Raw(Vec<f32>),
}

#[derive(Arbitrary, Debug)]
struct Input {
    log_host: bool,
    /// How `len_raw` becomes a payload length (see [`payload_len`]).
    len_kind: u8,
    len_raw: u8,
    key: Vec<u8>,
    payload: Vec<u8>,
    /// Almost always 16 kHz.
    rate: u8,
    pieces: Vec<Piece>,
    /// Samples cut from the front of the marked clip before detection.
    trim: u16,
    /// Gain applied to the marked clip before detection, in dB.
    gain_db: i8,
}

struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(seed | 1)
    }

    /// Uniform in `[-1, 1)`.
    fn bipolar(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 as i32 as f32) / (i32::MAX as f32 + 1.0)
    }
}

fn peak(code: u8) -> f32 {
    10f32.powf(f32::from(code) / 16.0 - 9.0)
}

fn build(pieces: &[Piece]) -> Vec<f32> {
    let mut clip: Vec<f32> = Vec::new();
    for piece in pieces {
        if clip.len() >= MAX_SAMPLES {
            break;
        }
        match piece {
            Piece::Silence(len) => clip.resize(clip.len() + usize::from(*len), 0.0),
            Piece::NegativeZero(len) => clip.resize(clip.len() + usize::from(*len), -0.0),
            Piece::Constant { len, level } => {
                let level = f32::from(*level) / 256.0;
                clip.resize(clip.len() + usize::from(*len), level);
            }
            Piece::Noise { len, seed, peak: p } => {
                let mut rng = Rng::new(*seed);
                let amp = peak(*p);
                clip.extend((0..*len).map(|_| rng.bipolar() * amp));
            }
            Piece::Square {
                len,
                period,
                peak: p,
            } => {
                let half = usize::from(*period) / 2 + 1;
                let amp = peak(*p);
                clip.extend(
                    (0..usize::from(*len)).map(|i| if (i / half) % 2 == 0 { amp } else { -amp }),
                );
            }
            Piece::Raw(samples) => clip.extend_from_slice(samples),
        }
    }
    clip.truncate(MAX_SAMPLES);
    clip
}

/// A payload length: mostly 0..=16, sometimes anything a byte can say, and
/// sometimes one that overflows when multiplied out to bits.
fn payload_len(kind: u8, raw: u8) -> usize {
    match kind % 8 {
        0 => usize::MAX,
        1 => usize::MAX / 8,
        2 => usize::from(raw),
        _ => usize::from(raw % 17),
    }
}

fn sample_rate(code: u8) -> SampleRate {
    match code % 32 {
        0 => SampleRate::HZ_8000,
        1 => SampleRate::HZ_44100,
        2 => SampleRate(NonZeroU32::MIN),
        _ => SampleRate::HZ_16000,
    }
}

/// Every sample equals the first (`0.0` equals `-0.0`).
fn is_flat(clip: &[f32]) -> bool {
    clip.windows(2).all(|pair| pair[0] == pair[1])
}

fn is_tame(clip: &[f32]) -> bool {
    clip.iter().all(|s| s.is_finite() && s.abs() <= TAME)
}

/// Run `op` on `clip`. When it fails with `AudioTooShort`, check that the
/// report is truthful, then pad the clip with noise to the length it names
/// (when that is within reach) and demand that the second attempt not fail
/// for length.
fn with_enough_audio<T>(
    clip: &mut Vec<f32>,
    op: impl Fn(&[f32]) -> Result<T, AfpError>,
) -> Option<T> {
    match op(clip) {
        Ok(value) => Some(value),
        Err(AfpError::AudioTooShort { needed, got }) => {
            assert_eq!(got, clip.len(), "AudioTooShort reported the wrong length");
            assert!(needed > got, "AudioTooShort asked for {needed} with {got}");
            if needed > MAX_SAMPLES {
                return None;
            }
            let mut rng = Rng::new(needed as u32);
            while clip.len() < needed {
                clip.push(rng.bipolar() * 0.2);
            }
            match op(clip) {
                Ok(value) => Some(value),
                Err(AfpError::AudioTooShort { .. }) => {
                    panic!("{needed} samples were not enough after AudioTooShort named them")
                }
                Err(_) => None,
            }
        }
        Err(_) => None,
    }
}

/// `unmarked`: the clip never went through `embed`.
fn check(
    result: &ClassicalResult,
    payload_len: usize,
    algorithm: Algorithm,
    clip: &[f32],
    unmarked: bool,
) {
    assert_eq!(result.detected(), result.payload.is_some());
    if unmarked {
        assert!(
            !result.detected(),
            "unmarked audio decoded to {:?}",
            result.payload
        );
    }
    if let Some(payload) = &result.payload {
        assert_eq!(payload.len(), payload_len);
    }
    if is_tame(clip) {
        assert!(
            result.score.is_finite() && result.score >= 0.0,
            "score {}",
            result.score
        );
    }
    if algorithm == Algorithm::QimBlock {
        assert!(
            result.sync_offset < 2048,
            "sync_offset {}",
            result.sync_offset
        );
    }
    if is_flat(clip) {
        assert!(!result.detected(), "a constant clip decoded to a payload");
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    let algorithm = if input.log_host {
        Algorithm::LogHost
    } else {
        Algorithm::QimBlock
    };
    let len = payload_len(input.len_kind, input.len_raw);
    let config = ClassicalConfig::new(len).with_algorithm(algorithm);
    // Saturates instead of overflowing, whatever the length.
    let _ = config.min_samples();

    let mut payload = input.payload.clone();
    if len <= usize::from(u8::MAX) {
        payload.resize(len, 0xA5);
    }
    let rate = sample_rate(input.rate);
    let mut clip = build(&input.pieces);

    let detector = ClassicalDetector::try_new(config, &input.key);
    let embedder = ClassicalEmbedder::try_new(config, &input.key, &payload);
    // Both build through one validation, so they agree whenever the payload
    // has the configured length.
    if payload.len() == len {
        assert_eq!(detector.is_ok(), embedder.is_ok());
    }
    if let Ok(detector) = &detector {
        let _ = format!("{detector:?}");
    }

    let marked = embedder.as_ref().ok().and_then(|embedder| {
        let _ = format!("{embedder:?}");
        let marked = with_enough_audio(&mut clip, |c| embedder.embed(c, rate))?;
        assert_eq!(marked.len(), clip.len(), "embed changed the length");
        if is_tame(&clip) {
            assert!(
                marked.iter().all(|s| s.is_finite()),
                "embed produced a non-finite sample"
            );
        }
        Some(marked)
    });

    let Ok(detector) = &detector else {
        return;
    };
    if let Some(found) = with_enough_audio(&mut clip, |c| detector.detect(c, rate)) {
        check(&found, len, algorithm, &clip, true);
    }

    if let Some(marked) = marked {
        // Whatever embed accepts, detect accepts too, unless the output holds
        // a non-finite sample (which `embed` does not yet rule out for
        // astronomically loud input).
        match detector.detect(&marked, rate) {
            Ok(found) => check(&found, len, algorithm, &marked, false),
            Err(error) => assert!(!is_tame(&marked), "detect refused embed's output: {error}"),
        }

        let gain = 10f32.powf(f32::from(input.gain_db) / 20.0);
        let cut = usize::from(input.trim) % 4096;
        let moved: Vec<f32> = marked.iter().skip(cut).map(|s| s * gain).collect();
        if let Ok(found) = detector.detect(&moved, rate) {
            check(&found, len, algorithm, &moved, false);
        }
    }
});
