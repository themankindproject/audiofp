//! Matching microbenches — Wang / Haitsma 1:1 and WangIndex 1:N.
//!
//! ```bash
//! cargo bench --bench matching
//! ```

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use audiofp::Fingerprinter;
use audiofp::SampleRate;
use audiofp::classical::{Haitsma, Panako, Wang, WangFingerprint};
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangIndex,
    WangMatchConfig, WangMatcher, WangRefIndex,
};

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

fn wang_fp(seed: u32, secs: usize) -> WangFingerprint {
    let samples = synth(seed, 8_000, secs);

    Wang::default()
        .extract(&samples, SampleRate::HZ_8000)
        .expect("wang extract")
}

fn bench_wang_one(c: &mut Criterion) {
    let query = wang_fp(1, 5);
    let reference = query.clone();
    let matcher = WangMatcher::new(WangMatchConfig::default());
    let index = WangRefIndex::build(&reference, &WangMatchConfig::default()).unwrap();

    let mut g = c.benchmark_group("matching/wang_1to1");
    g.throughput(Throughput::Elements(query.hashes.len() as u64));
    g.bench_function("self_match_5s", |b| {
        b.iter(|| black_box(matcher.match_one(black_box(&query), black_box(&reference))));
    });
    // audit C1: same operation with a prebuilt reference index — the
    // per-call O(R log R) `SortedPostings::build` is paid once up front.
    g.bench_function("self_match_5s_prebuilt", |b| {
        b.iter(|| black_box(matcher.match_one_prebuilt(black_box(&query), black_box(&index))));
    });
    g.finish();

    // Repeated single-reference matching against fixed audio: the exact
    // 1:1 use case C1 optimises.
    let queries: Vec<WangFingerprint> = (1..20u32).map(|i| wang_fp(100 + i, 5)).collect();
    let mut g = c.benchmark_group("matching/wang_1to1_reuse");
    g.bench_function("20_queries_vs_one_ref", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(matcher.match_one(black_box(q), black_box(&reference)));
            }
        });
    });
    g.bench_function("20_queries_vs_one_ref_prebuilt", |b| {
        b.iter(|| {
            for q in &queries {
                black_box(matcher.match_one_prebuilt(black_box(q), black_box(&index)));
            }
        });
    });
    g.finish();
}

fn bench_haitsma_one(c: &mut Criterion) {
    let samples = synth(2, 5_000, 5);

    let fp = Haitsma::default()
        .extract(&samples, SampleRate::HZ_5000)
        .expect("haitsma extract");
    // Exact path: keep refs short enough that LUT is skipped, or force use_lut=false.
    let matcher = HaitsmaMatcher::new(HaitsmaMatchConfig {
        use_lut: false,
        min_overlap_frames: 64,
        ..Default::default()
    });

    let mut g = c.benchmark_group("matching/haitsma_1to1");
    g.throughput(Throughput::Elements(fp.frames.len() as u64));
    g.bench_function("self_match_5s_exact", |b| {
        b.iter(|| black_box(matcher.match_one(black_box(&fp), black_box(&fp))));
    });
    g.finish();

    // audit C3: the pathological case the coarse-to-fine path targets —
    // a huge forced-exact reference with NO alignment (every delta's
    // hamming runs near its full length because no good BER exists to
    // early-abort on). The exhaustive scan is O(q·r·32) here; the
    // sampled sweep + refinement is O(probes·overlap/8 + window·overlap).
    let huge_q: Vec<u32> = (0..12_000u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    let huge_r: Vec<u32> = (0..12_000u32).map(|i| i.wrapping_mul(7_919_021)).collect();
    let exhaustive = HaitsmaMatcher::new(HaitsmaMatchConfig {
        use_lut: false,
        min_overlap_frames: 256,
        ..Default::default()
    });
    let coarse = HaitsmaMatcher::new(HaitsmaMatchConfig {
        use_lut: false,
        coarse_to_fine: true,
        min_overlap_frames: 256,
        ..Default::default()
    });
    let mut g = c.benchmark_group("matching/haitsma_1to1_no_match_24k_frames");
    g.bench_function("exhaustive", |b| {
        b.iter(|| {
            black_box(
                exhaustive.match_one(black_box(&fp_plain(&huge_q)), black_box(&fp_plain(&huge_r))),
            )
        });
    });
    g.bench_function("coarse_to_fine", |b| {
        b.iter(|| {
            black_box(
                coarse.match_one(black_box(&fp_plain(&huge_q)), black_box(&fp_plain(&huge_r))),
            )
        });
    });
    g.finish();
}

fn fp_plain(frames: &[u32]) -> audiofp::classical::HaitsmaFingerprint {
    audiofp::classical::HaitsmaFingerprint {
        frames: frames.to_vec(),
        frames_per_sec: 78.125,
    }
}

fn bench_panako_one(c: &mut Criterion) {
    let samples = synth(3, 8_000, 5);

    let fp = Panako::default()
        .extract(&samples, SampleRate::HZ_8000)
        .expect("panako extract");
    let matcher = PanakoMatcher::new(PanakoMatchConfig::default());

    let mut g = c.benchmark_group("matching/panako_1to1");
    g.throughput(Throughput::Elements(fp.hashes.len() as u64));
    g.bench_function("self_match_5s", |b| {
        b.iter(|| black_box(matcher.match_one(black_box(&fp), black_box(&fp))));
    });
    g.finish();
}

fn bench_wang_index(c: &mut Criterion) {
    let query = wang_fp(7, 3);
    let refs: Vec<WangFingerprint> = (0..100u32).map(|i| wang_fp(10 + i, 3)).collect();
    // Plant the true match in the middle.
    let mut catalog = refs;
    catalog[50] = query.clone();
    let index = WangIndex::build(&catalog, 100);
    let cfg = WangMatchConfig::default();

    let mut g = c.benchmark_group("matching/wang_index");
    g.throughput(Throughput::Elements(catalog.len() as u64));
    g.bench_function("n100_query", |b| {
        b.iter(|| black_box(index.query(black_box(&query), black_box(&cfg))));
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_wang_one,
    bench_haitsma_one,
    bench_panako_one,
    bench_wang_index
);
criterion_main!(benches);
