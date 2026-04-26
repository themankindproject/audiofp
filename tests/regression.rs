//! Bit-exact regression tests against committed v1 hash output.
//!
//! Each test:
//!   1. Generates a deterministic synthetic input (xorshift32 + two tones).
//!   2. Runs the corresponding fingerprinter.
//!   3. Compares its byte representation against `tests/goldens/<algo>.bin`.
//!
//! To regenerate the goldens after an *intentional* output change:
//!
//! ```bash
//! UPDATE_GOLDENS=1 cargo test --test regression
//! ```
//!
//! The PR description must justify the regeneration — these files protect
//! against unintentional numeric drift.

use std::path::Path;

use audiofp::classical::{Haitsma, Panako, PanakoHash, Wang, WangHash};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

const TONE_LO: f32 = 880.0;
const TONE_HI: f32 = 1320.0;
const SECS: f32 = 6.0;
const GOLDEN_DIR: &str = "tests/goldens";

/// Deterministic xorshift32-driven two-tone-with-noise synthesiser.
fn synth(seed: u32, sr: u32) -> Vec<f32> {
    let n = (sr as f32 * SECS) as usize;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        let s = 0.5 * (2.0 * std::f32::consts::PI * TONE_LO * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * TONE_HI * t).sin()
            + noise;
        out.push(s);
    }
    out
}

/// Read a golden file (panicking with a helpful message if it's missing
/// and we're not in update mode).
fn load_or_update(path: &Path, current: &[u8]) -> Vec<u8> {
    if std::env::var_os("UPDATE_GOLDENS").is_some() {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        std::fs::write(path, current).unwrap();
        eprintln!(
            "[goldens] wrote {} ({} bytes)",
            path.display(),
            current.len()
        );
    }
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "golden file {} missing or unreadable: {e}\n\
             Run with UPDATE_GOLDENS=1 cargo test --test regression to (re)generate.",
            path.display(),
        )
    })
}

/// Friendly error message helper that doesn't dump tens of KB on failure.
fn assert_bytes_equal(actual: &[u8], expected: &[u8], label: &str) {
    if actual == expected {
        return;
    }
    let first_diff = actual
        .iter()
        .zip(expected.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    panic!(
        "{label} hash output drift detected\n  \
         actual:   {} bytes\n  \
         expected: {} bytes\n  \
         first byte mismatch at offset {first_diff}\n\
         If this change is intentional, regenerate with:\n  \
         UPDATE_GOLDENS=1 cargo test --test regression",
        actual.len(),
        expected.len(),
    );
}

#[test]
fn wang_v1_golden() {
    let samples = synth(0xCAFE, 8_000);
    let mut wang = Wang::default();
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    let fp = wang.extract(buf).unwrap();

    let bytes = bytemuck::cast_slice::<WangHash, u8>(&fp.hashes);
    let path = Path::new(GOLDEN_DIR).join("wang_v1.bin");
    let expected = load_or_update(&path, bytes);
    assert_bytes_equal(bytes, &expected, "Wang");
}

#[test]
fn panako_v2_golden() {
    let samples = synth(0xCAFE, 8_000);
    let mut panako = Panako::default();
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    let fp = panako.extract(buf).unwrap();

    let bytes = bytemuck::cast_slice::<PanakoHash, u8>(&fp.hashes);
    let path = Path::new(GOLDEN_DIR).join("panako_v2.bin");
    let expected = load_or_update(&path, bytes);
    assert_bytes_equal(bytes, &expected, "Panako");
}

#[test]
fn haitsma_v1_golden() {
    let samples = synth(0xCAFE, 5_000);
    let mut h = Haitsma::default();
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::new(5_000).unwrap(),
    };
    let fp = h.extract(buf).unwrap();

    let bytes = bytemuck::cast_slice::<u32, u8>(&fp.frames);
    let path = Path::new(GOLDEN_DIR).join("haitsma_v1.bin");
    let expected = load_or_update(&path, bytes);
    assert_bytes_equal(bytes, &expected, "Haitsma");
}
