#![no_main]

use std::sync::atomic::{AtomicU64, Ordering};

use audiofp::classical::WangFingerprint;
use libfuzzer_sys::fuzz_target;

static SEQ: AtomicU64 = AtomicU64::new(0);

fn scratch_path(tag: &str, data: &[u8]) -> std::path::PathBuf {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let hash = data
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
    std::env::temp_dir().join(format!(
        "audiofp_fuzz_{tag}_{}_{}_{hash:016x}.bin",
        std::process::id(),
        n
    ))
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 256_000 {
        return;
    }

    // --- Untrusted cache blobs must never panic. ---
    let _ = WangFingerprint::from_bytes(data);
    let _ = audiofp::FingerprintEnvelope::peek(data);

    // --- Decoder paths: write raw bytes to a unique temp file and exercise
    // lenient + strict + report combinations (no shared fixed name). ---
    let path = scratch_path("decode", data);
    if std::fs::write(&path, data).is_err() {
        return;
    }

    let limits = audiofp::io::DecodeLimits::both(1_000_000, 500_000);
    let _ = audiofp::io::decode_to_mono_limited(&path, limits);
    let _ = audiofp::io::decode_to_mono_limited(&path, limits.strict());
    let _ = audiofp::io::decode_to_mono_at_limited(&path, 8_000, limits);
    let _ = audiofp::io::decode_to_mono_at_limited(&path, 8_000, limits.strict());
    let _ = audiofp::io::decode_to_mono_report(&path, limits);
    let _ = audiofp::io::decode_to_mono_report(&path, limits.strict());

    let _ = std::fs::remove_file(&path);
});
