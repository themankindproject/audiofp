#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() > 1_000_000 {
        return;
    }
    let path = std::env::temp_dir().join(format!("audiofp_fuzz_{}.bin", std::process::id()));
    if std::fs::write(&path, data).is_ok() {
        let _ = audiofp::io::decode_to_mono_limited(
            &path,
            audiofp::io::DecodeLimits::both(1_000_000, 500_000),
        );
        let _ = std::fs::remove_file(&path);
    }
});
