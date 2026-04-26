//! Compare two audio files by Wang hash overlap.
//!
//! ```bash
//! cargo run --example match_two_files -- song.flac song_re_encoded.mp3
//! ```
//!
//! Useful as a starter for "is B a re-encoding of A?" detection. For real
//! matching at scale you'd use `t_anchor` to verify same-offset collisions
//! and apply a histogram-of-time-deltas voter.

use std::collections::HashSet;

use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn fingerprint(wang: &mut Wang, path: &str) -> Result<HashSet<u32>, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    Ok(wang
        .extract(buf)?
        .hashes
        .into_iter()
        .map(|h| h.hash)
        .collect())
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
    let fa = fingerprint(&mut wang, &a)?;
    println!("  {} unique hashes", fa.len());

    println!("Fingerprinting {b}...");
    let fb = fingerprint(&mut wang, &b)?;
    println!("  {} unique hashes", fb.len());

    let shared = fa.intersection(&fb).count();
    let union = fa.union(&fb).count();
    let pct_max = 100.0 * shared as f64 / fa.len().max(fb.len()) as f64;
    let jaccard = if union == 0 {
        0.0
    } else {
        shared as f64 / union as f64
    };

    println!();
    println!("  shared hashes:    {shared}");
    println!("  overlap (max):    {pct_max:.1} %");
    println!("  jaccard:          {:.3}", jaccard);

    if pct_max >= 50.0 {
        println!("\n  → Likely the same recording.");
    } else if pct_max >= 10.0 {
        println!("\n  → Possibly related (cover, edit, or partial overlap).");
    } else {
        println!("\n  → Unrelated.");
    }

    Ok(())
}
