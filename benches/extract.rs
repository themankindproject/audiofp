//! Criterion benchmarks for the three classical fingerprinters.
//!
//! Run with:
//! ```bash
//! cargo bench --bench extract
//! cargo bench --bench extract -- wang/extract_30s   # filter
//! cargo bench --bench extract -- --save-baseline main
//! ```

use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

/// Deterministic xorshift32 + two-tone synthetic input. Same generator as the
/// regression-golden test so bench numbers are reproducible.
fn synth(seed: u32, sr: u32, secs: f32) -> Vec<f32> {
    let n = (sr as f32 * secs) as usize;
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / sr as f32;
        let s = 0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
            + noise;
        out.push(s);
    }
    out
}

fn bench_wang(c: &mut Criterion) {
    let mut group = c.benchmark_group("wang");
    for &secs in &[2.0_f32, 5.0, 30.0] {
        let samples = synth(0xCAFE, 8_000, secs);
        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("extract", format!("{}s", secs as u32)),
            &samples,
            |b, samples| {
                let mut wang = Wang::default();
                b.iter(|| {
                    let buf = AudioBuffer {
                        samples: black_box(samples),
                        rate: SampleRate::HZ_8000,
                    };
                    wang.extract(buf).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_panako(c: &mut Criterion) {
    let mut group = c.benchmark_group("panako");
    for &secs in &[2.0_f32, 5.0, 30.0] {
        let samples = synth(0xCAFE, 8_000, secs);
        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("extract", format!("{}s", secs as u32)),
            &samples,
            |b, samples| {
                let mut panako = Panako::default();
                b.iter(|| {
                    let buf = AudioBuffer {
                        samples: black_box(samples),
                        rate: SampleRate::HZ_8000,
                    };
                    panako.extract(buf).unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_haitsma(c: &mut Criterion) {
    let mut group = c.benchmark_group("haitsma");
    let sr = SampleRate::new(5_000).unwrap();
    for &secs in &[2.0_f32, 5.0, 30.0] {
        let samples = synth(0xCAFE, 5_000, secs);
        group.throughput(Throughput::Elements(samples.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("extract", format!("{}s", secs as u32)),
            &samples,
            |b, samples| {
                let mut h = Haitsma::default();
                b.iter(|| {
                    let buf = AudioBuffer {
                        samples: black_box(samples),
                        rate: sr,
                    };
                    h.extract(buf).unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_wang, bench_panako, bench_haitsma);
criterion_main!(benches);
