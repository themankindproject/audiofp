//! Fuzz the binary fingerprint serialization layer (`src/serial.rs`).
//!
//! Two properties are exercised:
//!
//! 1. **Untrusted-input safety** — arbitrary bytes fed to every public
//!    deserializer (`WangFingerprint::from_bytes`, `PanakoFingerprint::from_bytes`,
//!    `HaitsmaFingerprint::from_bytes`, `FingerprintEnvelope::peek`) must
//!    never panic, regardless of magic/version/algorithm/length/fps values.
//!    This is the crate's classic untrusted-binary attack surface (persisted
//!    blobs, cross-service exchange).
//! 2. **Roundtrip integrity** — an arbitrary-but-valid fingerprint
//!    (finite, positive fps) serialized via `to_bytes` must deserialize back
//!    to an identical value.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use audiofp::FingerprintEnvelope;
use audiofp::classical::{
    HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangFingerprint, WangHash,
};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct WangHashInput {
    hash: u32,
    t_anchor: u32,
}

#[derive(Arbitrary, Debug)]
struct PanakoHashInput {
    hash: u32,
    t_anchor: u32,
    t_b: u32,
    t_c: u32,
}

#[derive(Arbitrary, Debug)]
struct Input {
    /// Which deserializer(s) to exercise with the raw bytes.
    variant: u8,
    /// Raw bytes for the untrusted-parse path.
    raw: Vec<u8>,
    /// Structured fingerprint for the roundtrip path.
    wang_hashes: Vec<WangHashInput>,
    panako_hashes: Vec<PanakoHashInput>,
    haitsma_frames: Vec<u32>,
    /// Frame rate; clamped to finite-and-positive before use.
    fps: f32,
}

/// Clamp an arbitrary f32 into the valid frame-rate domain (finite, > 0).
fn valid_fps(f: f32) -> f32 {
    if !f.is_finite() || f <= 0.0 { 62.5 } else { f }
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = Unstructured::new(data).arbitrary::<Input>() else {
        return;
    };

    // --- Property 1: untrusted bytes must never panic any deserializer. ---
    // Each call must return Ok or Err, never unwind.
    let _ = WangFingerprint::from_bytes(&input.raw);
    let _ = PanakoFingerprint::from_bytes(&input.raw);
    let _ = HaitsmaFingerprint::from_bytes(&input.raw);
    let _ = FingerprintEnvelope::peek(&input.raw);

    // --- Property 2: valid fingerprints roundtrip exactly. ---
    let fps = valid_fps(input.fps);

    // Wang.
    let wang = WangFingerprint {
        hashes: input
            .wang_hashes
            .iter()
            .map(|h| WangHash {
                hash: h.hash,
                t_anchor: h.t_anchor,
            })
            .collect(),
        frames_per_sec: fps,
    };
    let wang_bytes = wang.to_bytes();
    let wang_rt =
        WangFingerprint::from_bytes(&wang_bytes).expect("valid Wang blob must deserialize");
    assert_eq!(wang_rt.hashes, wang.hashes);
    assert_eq!(wang_rt.frames_per_sec, wang.frames_per_sec);
    // Envelope metadata must agree with the serialized header.
    let env = FingerprintEnvelope::peek(&wang_bytes).expect("valid Wang blob must peek");
    assert_eq!(env.hash_count, wang.hashes.len());
    assert_eq!(env.frames_per_sec, fps);

    // Panako.
    let panako = PanakoFingerprint {
        hashes: input
            .panako_hashes
            .iter()
            .map(|h| PanakoHash {
                hash: h.hash,
                t_anchor: h.t_anchor,
                t_b: h.t_b,
                t_c: h.t_c,
            })
            .collect(),
        frames_per_sec: fps,
    };
    let panako_bytes = panako.to_bytes();
    let panako_rt =
        PanakoFingerprint::from_bytes(&panako_bytes).expect("valid Panako blob must deserialize");
    assert_eq!(panako_rt.hashes, panako.hashes);
    assert_eq!(panako_rt.frames_per_sec, panako.frames_per_sec);

    // Haitsma.
    let haitsma = HaitsmaFingerprint {
        frames: input.haitsma_frames.clone(),
        frames_per_sec: fps,
    };
    let haitsma_bytes = haitsma.to_bytes();
    let haitsma_rt = HaitsmaFingerprint::from_bytes(&haitsma_bytes)
        .expect("valid Haitsma blob must deserialize");
    assert_eq!(haitsma_rt.frames, haitsma.frames);
    assert_eq!(haitsma_rt.frames_per_sec, haitsma.frames_per_sec);

    // Cross-algorithm rejection: a Wang blob must not parse as Panako/Haitsma
    // (algorithm-id mismatch), and vice versa. Only check when non-empty so
    // the header is the distinguishing factor regardless of payload.
    let _ = input.variant; // variant reserved for future targeted paths
    if !wang.hashes.is_empty() {
        assert!(PanakoFingerprint::from_bytes(&wang_bytes).is_err());
        assert!(HaitsmaFingerprint::from_bytes(&wang_bytes).is_err());
    }
});
