//! Run all three classical fingerprinters on the same file, side by side.
//!
//! ```bash
//! cargo run --example compare_algorithms -- path/to/song.flac
//! ```
//!
//! Prints output size, frame rate, and elapsed time per algorithm so you
//! can pick the right one for your storage / latency budget.

// Examples are allowed to use Instant::now directly; the library-side
// clippy disallowed-methods lint is about testable time injection.
#![allow(clippy::disallowed_methods)]

use std::time::Instant;

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: compare_algorithms <audio-file>")?;

    println!("Decoding {path}...");
    let samples_8k = decode_to_mono_at(&path, 8_000)?;
    let samples_5k = decode_to_mono_at(&path, 5_000)?;
    let secs = samples_8k.len() as f32 / 8_000.0;
    println!("  {:.2} s of audio\n", secs);

    println!(
        "{:<12} {:>10} {:>10} {:>10} {:>14}",
        "algorithm", "hashes", "fps", "ms", "bytes/s"
    );
    println!("{:-<60}", "");

    // Wang
    {
        let mut wang = Wang::default();
        let buf = AudioBuffer {
            samples: &samples_8k,
            rate: SampleRate::HZ_8000,
        };
        let t = Instant::now();
        let fp = wang.extract(buf)?;
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;
        let bytes = fp.hashes.len() * core::mem::size_of::<audiofp::classical::WangHash>();
        println!(
            "{:<12} {:>10} {:>10.1} {:>10.1} {:>14.1}",
            wang.name(),
            fp.hashes.len(),
            fp.frames_per_sec,
            elapsed,
            bytes as f32 / secs,
        );
    }

    // Panako
    {
        let mut panako = Panako::default();
        let buf = AudioBuffer {
            samples: &samples_8k,
            rate: SampleRate::HZ_8000,
        };
        let t = Instant::now();
        let fp = panako.extract(buf)?;
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;
        let bytes = fp.hashes.len() * core::mem::size_of::<audiofp::classical::PanakoHash>();
        println!(
            "{:<12} {:>10} {:>10.1} {:>10.1} {:>14.1}",
            panako.name(),
            fp.hashes.len(),
            fp.frames_per_sec,
            elapsed,
            bytes as f32 / secs,
        );
    }

    // Haitsma
    {
        let mut h = Haitsma::default();
        let buf = AudioBuffer {
            samples: &samples_5k,
            rate: SampleRate::new(5_000).unwrap(),
        };
        let t = Instant::now();
        let fp = h.extract(buf)?;
        let elapsed = t.elapsed().as_secs_f32() * 1000.0;
        let bytes = fp.frames.len() * core::mem::size_of::<u32>();
        println!(
            "{:<12} {:>10} {:>10.1} {:>10.1} {:>14.1}",
            h.name(),
            fp.frames.len(),
            fp.frames_per_sec,
            elapsed,
            bytes as f32 / secs,
        );
    }

    Ok(())
}
