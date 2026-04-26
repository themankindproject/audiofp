//! Enroll an audio file by computing its Wang fingerprint.
//!
//! ```bash
//! cargo run --example enroll_file -- path/to/song.flac
//! ```
//!
//! Decodes any Symphonia-supported format, resamples to Wang's 8 kHz, and
//! prints the hash count plus a sample of hashes for inspection.

use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| "usage: enroll_file <audio-file>".to_string())?;

    println!("Decoding {path}...");
    let samples = decode_to_mono_at(&path, 8_000)?;
    let secs = samples.len() as f32 / 8_000.0;
    println!(
        "  {} samples ({:.2} s of mono 8 kHz audio)",
        samples.len(),
        secs
    );

    println!("Extracting Wang fingerprint...");
    let mut wang = Wang::default();
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    let fp = wang.extract(buf)?;

    println!(
        "  {} hashes at {:.1} fps ({:.1} hashes/s)",
        fp.hashes.len(),
        fp.frames_per_sec,
        fp.hashes.len() as f32 / secs,
    );
    println!("  algorithm: {}", wang.name());

    let preview = fp.hashes.iter().take(8);
    println!("\nFirst 8 hashes (t_anchor, hash):");
    for h in preview {
        println!("  {:>6}  {:08x}", h.t_anchor, h.hash);
    }

    Ok(())
}
