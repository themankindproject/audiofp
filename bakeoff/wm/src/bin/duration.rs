//! Duration / payload-capacity sweep: how many frames per payload bit are
//! needed before clean recovery is reliable? This sets the minimum clip
//! length the API should require.

use audiofp_wm_bakeoff::algo::{Candidate, MagDsss};
use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;

fn dec(src: &Path, sr: u32) -> Vec<f32> {
    let t = std::env::temp_dir().join("dur.wav");
    std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-ac", "1", "-ar", &sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(&t)
        .status()
        .unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}

fn ber(a: &[u8], b: &[u8]) -> f64 {
    (0..a.len()).filter(|&i| a[i] != b[i]).count() as f64 / a.len() as f64
}

fn main() {
    let sr = 16_000u32;
    let key = b"bakeoff-secret-key-0123456789";
    let hop = 512usize;
    let fps = sr as f64 / hop as f64; // 31.25 fps
    let mut p = dsp::ChaCha::new(b"payload", 0);
    let payload64: Vec<u8> = (0..64).map(|_| (p.u32() & 1) as u8).collect();
    let payload32: Vec<u8> = payload64[..32].to_vec();

    // Long source to slice from.
    let mut src = dec(Path::new("assets/grieg.ogg"), sr);
    let m = dec(Path::new("assets/mozart.ogg"), sr);
    src.extend_from_slice(&m);

    let mut c = MagDsss::new();
    println!("frames/sec = {fps:.2}");
    println!(
        "{:>6} {:>7} {:>9} {:>12} {:>12}",
        "secs", "frames", "fr/bit64", "BER(64bit)", "BER(32bit)"
    );
    for &secs in &[4usize, 8, 12, 16, 20, 24, 30, 45, 60] {
        let n = secs * sr as usize;
        if n > src.len() {
            break;
        }
        let seg = &src[..n];
        let frames = n / hop;
        let (wm, _) = c.embed(seg, &payload64, key);
        let b64 = ber(&c.detect(&wm, 64, key).0, &payload64);
        let (wm2, _) = c.embed(seg, &payload32, key);
        let b32 = ber(&c.detect(&wm2, 32, key).0, &payload32);
        println!(
            "{secs:>6} {frames:>7} {:>9.1} {:>12.3} {:>12.3}",
            frames as f64 / 64.0,
            b64,
            b32
        );
    }
}
