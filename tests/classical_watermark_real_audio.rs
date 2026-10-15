//! Real-audio tests for the keyed classical watermark detector.
//!
//! The in-crate unit tests host the mark in seeded white noise, the easiest
//! host there is, and that hid what the release review of PR #154 found on
//! real recordings: level changes of 1.5 dB or more, leading trims off the
//! 64-sample search grid, hosts that are mostly digital silence, a LogHost
//! that missed half its clean round trips, and silence that "contained" a
//! payload when `payload_len = 4`.
//!
//! Every case here embeds into a real recording (CC0 speech and piano, a
//! Kevin MacLeod flute, Musopen orchestral excerpts), takes the marked audio
//! through PCM16 as a WAV round trip would, applies the transform, and asks
//! the public detector for the payload back. Decoding goes through
//! `audiofp::io`, so the clips are the 16 kHz mono audio a caller gets from
//! `decode_to_mono_at(path, 16_000)`.
//!
//! Each case was checked first on 12 keys per clip and passed on all of them,
//! and the shipped (PR #154) detector fails the level, trim and silence cases
//! on every one. The LogHost cases are the opposite kind of check: LogHost
//! still loses roughly one clean round trip in five at 16 s, so they pin
//! (clip, key) pairs that the shipped detector missed and the current one
//! recovers, rather than a rate. The rates, and what they do not show, are in
//! the changelog.
#![cfg(all(
    feature = "classical-watermark",
    feature = "std-ogg",
    feature = "std-flac"
))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use audiofp::SampleRate;
use audiofp::io::decode_to_mono_at;
use audiofp::watermark::classical::{
    Algorithm, ClassicalConfig, ClassicalDetector, ClassicalEmbedder, ClassicalResult,
};

const RATE: SampleRate = SampleRate::HZ_16000;
const BLOCK: usize = 2048;

/// One key and payload to embed.
#[derive(Clone, Copy)]
struct Trial {
    key: &'static [u8],
    payload: [u8; 8],
}

const TRIALS: [Trial; 3] = [
    Trial {
        key: b"bakeoff-secret-key-0123456789",
        payload: [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
    },
    Trial {
        key: b"another-key-A",
        payload: [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
    },
    Trial {
        key: b"k3",
        payload: [0xA5; 8],
    },
];

const SPEECH: &str = "tests/assets/speech.ogg";
const PIANO: &str = "tests/assets/piano.ogg";
const GALWAY: &str = "tests/assets/galway.flac";
/// The 30 s catalog track.
const GRIEG: &str = "tests/assets/catalog/grieg_morning.ogg";
const CORIOLAN: &str = "tests/assets/catalog/beethoven_coriolan.ogg";

type Clips = Mutex<HashMap<&'static str, Arc<Vec<f32>>>>;

/// The decoded 16 kHz mono clip, decoded once per test binary.
fn clip(path: &'static str) -> Arc<Vec<f32>> {
    static CLIPS: OnceLock<Clips> = OnceLock::new();
    let mut clips = CLIPS.get_or_init(Default::default).lock().unwrap();
    clips
        .entry(path)
        .or_insert_with(|| {
            Arc::new(decode_to_mono_at(path, 16_000).unwrap_or_else(|e| panic!("{path}: {e}")))
        })
        .clone()
}

/// The first `seconds` of the clip, or all of it.
fn first(path: &'static str, seconds: usize) -> Vec<f32> {
    let samples = clip(path);
    samples[..(seconds * 16_000).min(samples.len())].to_vec()
}

/// Write scale 32767, read scale 32768, as a 16-bit WAV round trip does.
fn pcm16(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| f32::from((s.clamp(-1.0, 1.0) * 32767.0).round() as i16) / 32768.0)
        .collect()
}

fn gain(samples: &[f32], db: f32) -> Vec<f32> {
    let g = 10f32.powf(db / 20.0);
    pcm16(&samples.iter().map(|&s| s * g).collect::<Vec<_>>())
}

fn config(algorithm: Algorithm) -> ClassicalConfig {
    ClassicalConfig::new(8).with_algorithm(algorithm)
}

/// `host` with `trial` embedded, as it comes back from a 16-bit WAV.
fn mark(algorithm: Algorithm, trial: Trial, host: &[f32]) -> Vec<f32> {
    let embedder =
        ClassicalEmbedder::try_new(config(algorithm), trial.key, &trial.payload).expect("embedder");
    pcm16(&embedder.embed(host, RATE).expect("embed"))
}

fn detect(algorithm: Algorithm, key: &[u8], samples: &[f32]) -> ClassicalResult {
    ClassicalDetector::try_new(config(algorithm), key)
        .expect("detector")
        .detect(samples, RATE)
        .expect("detect")
}

fn assert_recovers(algorithm: Algorithm, trial: Trial, samples: &[f32], what: &str) {
    let result = detect(algorithm, trial.key, samples);
    assert_eq!(
        result.payload.as_deref(),
        Some(trial.payload.as_slice()),
        "{algorithm:?}: {what}"
    );
    assert!(result.detected(), "{algorithm:?}: {what}");
}

#[test]
fn qim_clean_pcm16_round_trip_on_real_audio() {
    for (name, path) in [
        ("speech", SPEECH),
        ("piano", PIANO),
        ("galway", GALWAY),
        ("grieg 30 s", GRIEG),
    ] {
        let marked = mark(Algorithm::QimBlock, TRIALS[0], &clip(path));
        assert_recovers(Algorithm::QimBlock, TRIALS[0], &marked, name);
    }
}

/// A gain 1.5 dB or more away from a multiple of 6.02 dB swapped or
/// scattered every QIM bit: the shipped detector recovered 0 of 30 at
/// +/-3 dB and at -7.5 dB, and 0 of 12 here.
#[test]
fn qim_survives_gain_changes_on_real_audio() {
    for (name, path) in [("speech", SPEECH), ("piano", PIANO)] {
        let marked = mark(Algorithm::QimBlock, TRIALS[1], &first(path, 8));
        for db in [-3.0f32, 3.0, -7.5] {
            assert_recovers(
                Algorithm::QimBlock,
                TRIALS[1],
                &gain(&marked, db),
                &format!("{name} {db:+} dB"),
            );
        }
    }
}

/// Trims off the 64-sample search grid lost the mark: the shipped detector
/// recovered 11 to 25 of 30 at 1, 17, 32 and 100 samples. The review's
/// "trims 5/5" used multiples of 64.
#[test]
fn qim_survives_off_grid_lead_trims_on_real_audio() {
    for (name, path) in [("galway", GALWAY), ("grieg", GRIEG)] {
        let marked = mark(Algorithm::QimBlock, TRIALS[2], &first(path, 8));
        for trim in [1usize, 17, 32, 100] {
            let result = detect(Algorithm::QimBlock, TRIALS[2].key, &marked[trim..]);
            assert_eq!(
                result.payload.as_deref(),
                Some(TRIALS[2].payload.as_slice()),
                "{name} trim {trim}"
            );
            // The block grid starts `trim` samples before the clip does.
            assert_eq!(
                result.sync_offset,
                (BLOCK - trim) % BLOCK,
                "{name} trim {trim}"
            );
        }
    }
}

/// A host that is 90% digital silence has almost nothing to vote with; the
/// silent blocks used to vote for all-ones and drown it.
#[test]
fn qim_survives_ninety_percent_silence_on_real_audio() {
    for (name, path) in [("galway", GALWAY), ("grieg 30 s", GRIEG)] {
        let marked = mark(Algorithm::QimBlock, TRIALS[0], &clip(path));
        let n = marked.len();
        let mut quiet = vec![0.0f32; n];
        let (from, to) = (n * 45 / 100, n * 55 / 100);
        quiet[from..to].copy_from_slice(&marked[from..to]);
        assert_recovers(Algorithm::QimBlock, TRIALS[0], &quiet, name);
    }
}

/// A level change and a trim at once: the level estimate has to survive the
/// alignment search being off the grid, and the other way round.
#[test]
fn qim_survives_a_level_change_and_a_trim_together() {
    let marked = mark(Algorithm::QimBlock, TRIALS[1], &first(GALWAY, 8));
    let changed = gain(&marked[100..], -7.5);
    assert_recovers(
        Algorithm::QimBlock,
        TRIALS[1],
        &changed,
        "galway -7.5 dB, trim 100",
    );
}

#[test]
fn qim_wrong_key_and_unmarked_real_audio_return_no_payload() {
    let host = first(GALWAY, 8);
    let marked = mark(Algorithm::QimBlock, TRIALS[0], &host);
    assert!(
        detect(Algorithm::QimBlock, TRIALS[1].key, &marked)
            .payload
            .is_none(),
        "wrong key"
    );
    assert!(
        detect(Algorithm::QimBlock, TRIALS[0].key, &host)
            .payload
            .is_none(),
        "unmarked"
    );
}

/// The shipped LogHost recovered 16 of 30 clean round trips on ten real
/// tracks; the clip-wide mean it subtracted left the host's drift in the
/// correlation. These (clip, key) pairs were missed by it (Coriolan with
/// both keys, the flute with the first) and are recovered now. Only clips at
/// least [`ClassicalConfig::min_samples`] long are asserted: the review
/// measured 0 of 30 at the 3.04 s the API accepts.
#[test]
fn log_host_clean_pcm16_round_trip_on_real_audio() {
    let min = config(Algorithm::LogHost).min_samples();
    for (name, path, trials) in [
        ("coriolan 30 s", CORIOLAN, &TRIALS[..2]),
        ("galway", GALWAY, &TRIALS[..1]),
        ("grieg 30 s", GRIEG, &TRIALS[..1]),
    ] {
        let host = clip(path);
        assert!(host.len() >= min, "{name} is shorter than min_samples");
        for &trial in trials {
            let marked = mark(Algorithm::LogHost, trial, &host);
            assert_recovers(Algorithm::LogHost, trial, &marked, name);
        }
    }
}

/// The USAGE.md configuration (`payload_len = 4`, key `asset-key-42`): both
/// algorithms returned `Some([0xFF; 4])`, with `detected() == true`, for
/// digital silence under any key. Here the silence is the length of a real
/// recording, the way a muted track or an empty channel arrives, and a real
/// recording 120 dB down is a host with nothing the carriers can read.
#[test]
fn silence_never_contains_a_payload_for_the_usage_configuration() {
    let key = b"asset-key-42";
    let galway = clip(GALWAY);
    let n = galway.len();
    let inaudible: Vec<f32> = first(SPEECH, 7).iter().map(|s| s * 1e-6).collect();
    for algorithm in [Algorithm::QimBlock, Algorithm::LogHost] {
        let config = ClassicalConfig::new(4).with_algorithm(algorithm);
        let detector = ClassicalDetector::try_new(config, key).unwrap();
        let mut inputs = vec![
            ("all-zero", vec![0.0f32; n]),
            ("negative zero", vec![-0.0f32; n]),
        ];
        if algorithm == Algorithm::QimBlock {
            inputs.push(("a recording 120 dB down", inaudible.clone()));
        }
        for (what, silence) in inputs {
            let result = detector.detect(&silence, RATE).unwrap();
            assert!(
                !result.detected(),
                "{algorithm:?}: {what}: {:?}",
                result.payload
            );
            assert!(result.payload.is_none());
        }
        // Unmarked real audio is not a payload either.
        let unmarked = detector.detect(&galway, RATE).unwrap();
        assert!(
            !unmarked.detected(),
            "{algorithm:?}: unmarked: {:?}",
            unmarked.payload
        );
    }
}
