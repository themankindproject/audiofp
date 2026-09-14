//! Streaming-throughput microbenches for the classical fingerprinters.
//!
//! Each fingerprinter is benched in two ingestion patterns:
//!
//! - **Small chunks** (256 samples / push): mimics a realtime mic capture loop.
//! - **Large chunks** (1 s / push): mimics offline batch ingestion.
//!
//! Naming conventions:
//!
//! - `push_flush_session`: one full pass (push all chunks + flush) per timed
//!   iteration. Streamer construction runs in criterion's untimed setup;
//!   destructor runs between batches (outside the timed routine).
//! - `push_with_callback_warmed`: after an untimed warmup pass, times a single
//!   steady-state `push_with` call — no flush, no drop in the timed routine.
//!
//! Run with:
//! ```bash
//! cargo bench --bench streaming
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

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

fn push_flush_session<S: StreamingFingerprinter>(s: &mut S, audio: &[f32], chunk: usize) {
    let mut start = 0;
    while start < audio.len() {
        let end = (start + chunk).min(audio.len());
        black_box(s.push(&audio[start..end]).unwrap());
        start = end;
    }
    black_box(s.flush().unwrap());
}

fn warm_callbacks<S: StreamingFingerprinter>(s: &mut S, audio: &[f32]) {
    for _ in 0..4 {
        for chunk in audio.chunks(SMALL_CHUNK) {
            s.push_with(chunk, |_, _| {}).unwrap();
        }
    }
}

fn bench_streaming_wang(c: &mut Criterion) {
    let audio = synth(1, 8_000, SECS);
    let large = audio.len();
    let chunk_1s = large / SECS;

    let mut g = c.benchmark_group("streaming/wang");
    g.throughput(Throughput::Elements(audio.len() as u64));

    g.bench_function("push_flush_session/small_chunk_256", |b| {
        b.iter_batched_ref(
            StreamingWang::default,
            |s| push_flush_session(s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("push_flush_session/large_chunk_1s", |b| {
        b.iter_batched_ref(
            StreamingWang::default,
            |s| push_flush_session(s, &audio, chunk_1s),
            criterion::BatchSize::SmallInput,
        );
    });
    g.throughput(Throughput::Elements(SMALL_CHUNK as u64));
    g.bench_function("push_with_callback_warmed/small_chunk_256", |b| {
        b.iter_batched_ref(
            || {
                let mut s = StreamingWang::default();
                warm_callbacks(&mut s, &audio);
                s
            },
            |s| {
                black_box(s.push_with(&audio[..SMALL_CHUNK], |_, _| {}).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_streaming_panako(c: &mut Criterion) {
    let audio = synth(2, 8_000, SECS);
    let large = audio.len();
    let chunk_1s = large / SECS;

    let mut g = c.benchmark_group("streaming/panako");
    g.throughput(Throughput::Elements(audio.len() as u64));

    g.bench_function("push_flush_session/small_chunk_256", |b| {
        b.iter_batched_ref(
            StreamingPanako::default,
            |s| push_flush_session(s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("push_flush_session/large_chunk_1s", |b| {
        b.iter_batched_ref(
            StreamingPanako::default,
            |s| push_flush_session(s, &audio, chunk_1s),
            criterion::BatchSize::SmallInput,
        );
    });
    g.throughput(Throughput::Elements(SMALL_CHUNK as u64));
    g.bench_function("push_with_callback_warmed/small_chunk_256", |b| {
        b.iter_batched_ref(
            || {
                let mut s = StreamingPanako::default();
                warm_callbacks(&mut s, &audio);
                s
            },
            |s| {
                black_box(s.push_with(&audio[..SMALL_CHUNK], |_, _| {}).unwrap());
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_streaming_haitsma(c: &mut Criterion) {
    let audio = synth(3, 5_000, SECS);
    let large = audio.len();
    let chunk_1s = large / SECS;

    let mut g = c.benchmark_group("streaming/haitsma");
    g.throughput(Throughput::Elements(audio.len() as u64));

    g.bench_function("push_flush_session/small_chunk_256", |b| {
        b.iter_batched_ref(
            StreamingHaitsma::default,
            |s| push_flush_session(s, &audio, SMALL_CHUNK),
            criterion::BatchSize::SmallInput,
        );
    });
    g.bench_function("push_flush_session/large_chunk_1s", |b| {
        b.iter_batched_ref(
            StreamingHaitsma::default,
            |s| push_flush_session(s, &audio, chunk_1s),
            criterion::BatchSize::SmallInput,
        );
    });
    g.throughput(Throughput::Elements(SMALL_CHUNK as u64));
    g.bench_function("push_with_callback_warmed/small_chunk_256", |b| {
        b.iter_batched_ref(
            || {
                let mut s = StreamingHaitsma::default();
                warm_callbacks(&mut s, &audio);
                s
            },
            |s| {
                black_box(s.push_with(&audio[..SMALL_CHUNK], |_, _| {}).unwrap());
            },
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
