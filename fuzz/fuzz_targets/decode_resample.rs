#![no_main]

use std::sync::atomic::{AtomicU64, Ordering};

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

static SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<u8>,
    target_sr: u16,
    strict: bool,
}

fuzz_target!(|input: Input| {
    if input.data.len() > 500_000 {
        return;
    }
    let sr = (input.target_sr as u32).max(1000).min(48_000);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let hash = input
        .data
        .iter()
        .fold(0u64, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as u64));
    let path = std::env::temp_dir().join(format!(
        "audiofp_fuzz_at_{}_{}_{hash:016x}.bin",
        std::process::id(),
        n
    ));
    if std::fs::write(&path, &input.data).is_ok() {
        let limits = audiofp::io::DecodeLimits::both(500_000, 250_000);
        let limits = if input.strict {
            limits.strict()
        } else {
            limits
        };
        let _ = audiofp::io::decode_to_mono_at_limited(&path, sr, limits);
        let _ = audiofp::io::decode_to_mono_report(&path, limits);
        let _ = std::fs::remove_file(&path);
    }
});
