//! Compare two audio files using the Wang fingerprint matcher.
//!
//! This is a sound replacement for the old hash-set-overlap approach.
//! It uses the offset-histogram voter from `WangMatcher` — matching
//! landmark hashes must agree on a constant time offset, which
//! eliminates the false positives that plague naive hash-set overlap.
//!
//! ```bash
//! cargo run --example match_two_files -- song.flac song_re_encoded.mp3
//! ```

use audiofp::classical::{Wang, WangFingerprint};
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn fingerprint(wang: &mut Wang, path: &str) -> Result<WangFingerprint, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    
    Ok(wang.extract(&samples, SampleRate::HZ_8000)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let a = args
        .next()
        .ok_or("usage: match_two_files <file-a> <file-b>")?;
    let b = args
        .next()
        .ok_or("usage: match_two_files <file-a> <file-b>")?;

    let mut wang = Wang::default();

    println!("Fingerprinting {a}...");
    let fprint_a = fingerprint(&mut wang, &a)?;
    println!("  {} landmarks", fprint_a.hashes.len());

    println!("Fingerprinting {b}...");
    let fprint_b = fingerprint(&mut wang, &b)?;
    println!("  {} landmarks", fprint_b.hashes.len());

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let result = matcher.match_one(&fprint_a, &fprint_b);

    println!();
    println!("  score:      {:.4}", result.score);
    println!("  votes:      {}", result.votes);
    println!("  prominence: {:.2}", result.prominence);
    println!(
        "  offset:     {} ms ({} frames)",
        result.offset.ms, result.offset.frames
    );

    if result.is_match {
        println!(
            "\n  → Same recording (confidence {:.1}%)",
            result.score * 100.0
        );
    } else if result.score > 0.05 {
        println!("\n  → Possibly related (partial overlap, cover, or edit).");
    } else {
        println!("\n  → Unrelated.");
    }

    Ok(())
}
