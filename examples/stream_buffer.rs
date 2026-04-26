//! Drive `StreamingWang` with a synthetic audio buffer fed in chunks.
//!
//! ```bash
//! cargo run --example stream_buffer
//! ```
//!
//! No external file needed — generates a 5-second two-tone-plus-noise
//! signal and feeds it to the streaming fingerprinter in 200 ms chunks
//! to simulate realtime delivery.

use audiofp::StreamingFingerprinter;
use audiofp::classical::StreamingWang;
use core::f32::consts::PI;

fn synth(seed: u32, sr: u32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = ((x as i32 as f32) / (i32::MAX as f32)) * 0.05;
        let t = i as f32 / sr as f32;
        let s = 0.5 * (2.0 * PI * 880.0 * t).sin() + 0.3 * (2.0 * PI * 1320.0 * t).sin() + noise;
        out.push(s);
    }
    out
}

fn main() {
    const SR: u32 = 8_000;
    const CHUNK_MS: f32 = 200.0;
    let chunk_samples = (SR as f32 * CHUNK_MS / 1000.0) as usize;

    let samples = synth(0xC0FFEE, SR, 5.0);
    println!(
        "Synthesised {} samples ({:.2} s) at {} Hz; chunk size: {} samples ({:.0} ms)",
        samples.len(),
        samples.len() as f32 / SR as f32,
        SR,
        chunk_samples,
        CHUNK_MS,
    );

    let mut s = StreamingWang::default();
    println!("Streaming latency: {} ms\n", s.latency_ms());

    let mut total = 0usize;
    for (i, chunk) in samples.chunks(chunk_samples).enumerate() {
        let emitted = s.push(chunk);
        if !emitted.is_empty() {
            println!(
                "chunk #{:>3}  push {:>4} samples  → {:>3} hashes",
                i,
                chunk.len(),
                emitted.len(),
            );
            total += emitted.len();
        }
    }

    let tail = s.flush();
    if !tail.is_empty() {
        println!("flush                          → {:>3} hashes", tail.len());
        total += tail.len();
    }

    println!("\n  total hashes emitted: {total}");
}
