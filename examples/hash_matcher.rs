//! Time-aligned hash voter — the actual matching algorithm a Shazam-style
//! system runs on top of `audiofp`'s landmark output.
//!
//! ```bash
//! cargo run --example hash_matcher --release -- ref1.flac ref2.flac -- query.mp3
//! ```
//!
//! Everything before `--` is enrolled into an in-memory database; everything
//! after is queried against it. For each query the program prints the top
//! reference matches with a vote count and the best-matching time offset.

use audiofp::classical::{Wang, WangFingerprint, WangHash};
use audiofp::io::decode_to_mono_at;
use audiofp::matcher::Matcher;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn fingerprint(wang: &mut Wang, path: &str) -> Result<WangFingerprint, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    let buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_8000,
    };
    Ok(wang.extract(buf)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sep = args
        .iter()
        .position(|a| a == "--")
        .ok_or("usage: hash_matcher <ref...> -- <query...>")?;
    let refs = &args[..sep];
    let queries = &args[sep + 1..];
    if refs.is_empty() || queries.is_empty() {
        return Err("need at least one reference and a query".into());
    }

    let mut wang = Wang::default();
    let mut m: Matcher<WangHash> = Matcher::new();

    println!("Enrolling {} reference track(s)...", refs.len());
    for path in refs {
        let fp = fingerprint(&mut wang, path)?;
        let id = m.enroll(path.clone(), &fp.hashes);
        println!(
            "  [{:>2}] {}  ({} hashes, index {} hashes)",
            id,
            path,
            fp.hashes.len(),
            m.hash_count()
        );
    }

    for query_path in queries {
        println!("\n--- Querying: {} ---", query_path);
        let qfp = fingerprint(&mut wang, query_path)?;
        let frames_per_sec = qfp.frames_per_sec;
        println!(
            "  {} query hashes at {:.1} fps",
            qfp.hashes.len(),
            frames_per_sec
        );

        let results = m.query(&qfp.hashes);
        if results.is_empty() {
            println!("  no matches found in any reference");
            continue;
        }

        println!();
        println!("  Score   Hits  Offset    Track");
        println!("  -----  -----  --------  -----");
        for r in results.iter().take(5) {
            let offset_secs = r.offset_frames as f32 / frames_per_sec;
            println!(
                "  {:>5}  {:>5}  {:>+7.2}s  {}",
                r.score, r.total_collisions, offset_secs, r.track_name,
            );
        }

        // Heuristic verdict: a peak that's at least 5× higher than the
        // mean per-bucket noise floor is almost certainly a real match.
        if let Some(best) = results.first() {
            let noise = if best.total_collisions > best.score {
                ((best.total_collisions - best.score) / 100).max(1)
            } else {
                1
            };
            if best.score > noise * 5 {
                println!("\n  → confident match: {}", best.track_name);
            } else {
                println!(
                    "\n  → no confident match (best score {} too close to noise)",
                    best.score
                );
            }
        }
    }

    Ok(())
}
