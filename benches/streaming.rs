//! Streaming-throughput microbenches for the classical fingerprinters.
//!
//! Each fingerprinter is benched in two push patterns that expose
//! different cost centres:
//!
//! - **Small chunks** (256 samples / push): mimics a realtime mic
//!   capture loop. Per-frame overhead (allocations, drain shifts) is a
//!   larger fraction of total work.
//! - **Large chunks** (1 s / push): mimics offline batch ingestion. A
//!   single push processes many frames; per-frame `drain` becomes
//!   O(frames²) over a long buffer.
//!
//! Run with:
//! ```bash
//! cargo bench --bench streaming
//! ```

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};

use audiofp::StreamingFingerprinter;
use audiofp::classical::{StreamingHaitsma, StreamingPanako, StreamingWang};

const SECS: usize = 5;
const SMALL_CHUNK: usize = 256;

fn synth(seed: u32, sr: u32, secs: usize) -> Vec<f32> {
    let n = sr as usize * secs;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        out.push(
            0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
                + noise,
        );
    }
    out
}

fn run_in_chunks<S: StreamingFingerprinter>(s: &mut S, audio: &[f32], chunk: usize) {
    let mut start = 0;
    while start < audio.len() {
        let end = (start + chunk).min(audio.len());
        black_box(s.push(&audio[start..end]));
        start = end;
    }
    black_box(s.flush());
}

fn bench_streaming_wang(c: &mut Criterion) {
    let audio = synth(1, 8_000, SECS);
    let large = audio.len();

    let mut g = c.benchmark_group("streaming/wang");
    g.throughput(Throughput::Elements(audio.len() as u64));
    g.bench_function("small_chunk_256", |b| {
        b.iter_batched(
            StreamingWang::default,
            |mut s| run_in_chunks(&mut s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("large_chunk_1s", |b| {
        b.iter_batched(
            StreamingWang::default,
            |mut s| run_in_chunks(&mut s, &audio, large / SECS),
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_streaming_panako(c: &mut Criterion) {
    let audio = synth(2, 8_000, SECS);
    let large = audio.len();

    let mut g = c.benchmark_group("streaming/panako");
    g.throughput(Throughput::Elements(audio.len() as u64));
    g.bench_function("small_chunk_256", |b| {
        b.iter_batched(
            StreamingPanako::default,
            |mut s| run_in_chunks(&mut s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("large_chunk_1s", |b| {
        b.iter_batched(
            StreamingPanako::default,
            |mut s| run_in_chunks(&mut s, &audio, large / SECS),
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_streaming_haitsma(c: &mut Criterion) {
    let audio = synth(3, 5_000, SECS);
    let large = audio.len();

    let mut g = c.benchmark_group("streaming/haitsma");
    g.throughput(Throughput::Elements(audio.len() as u64));
    g.bench_function("small_chunk_256", |b| {
        b.iter_batched(
            StreamingHaitsma::default,
            |mut s| run_in_chunks(&mut s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("large_chunk_1s", |b| {
        b.iter_batched(
            StreamingHaitsma::default,
            |mut s| run_in_chunks(&mut s, &audio, large / SECS),
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

criterion_group!(
    streaming,
    bench_streaming_wang,
    bench_streaming_panako,
    bench_streaming_haitsma
);
criterion_main!(streaming);
