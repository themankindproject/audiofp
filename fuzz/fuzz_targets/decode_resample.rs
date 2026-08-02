#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<u8>,
    target_sr: u16,
}

fuzz_target!(|input: Input| {
    if input.data.len() > 500_000 {
        return;
    }
    let sr = (input.target_sr as u32).max(1000).min(48000);
    let path = std::env::temp_dir().join(format!("audiofp_fuzz_at_{}.bin", std::process::id()));
    if std::fs::write(&path, &input.data).is_ok() {
        let _ = audiofp::io::decode_to_mono_at_limited(
            &path,
            sr,
            audiofp::io::DecodeLimits::both(500_000, 250_000),
        );
        let _ = std::fs::remove_file(&path);
    }
});
