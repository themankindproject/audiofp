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
#[cfg(feature = "rayon")]
use audiofp::matching::par_match_ranked;
use audiofp::matching::{
    HaitsmaMatchConfig, HaitsmaMatcher, Matcher, PanakoMatchConfig, PanakoMatcher, WangIndex,
    WangMatchConfig, WangMatcher, WangRefIndex, match_ranked,
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

    // audit C5: sequential vs rayon-parallel 1:N ranking on the same
    // catalog (bench compiles without the feature; the par case is
    // cfg'd out). Requires `cargo bench --features rayon`.
    let matcher = WangMatcher::new(WangMatchConfig::default());
    let mut g = c.benchmark_group("matching/wang_ranked_n100");
    g.bench_function("sequential", |b| {
        b.iter(|| {
            black_box(match_ranked(&matcher, &query, &catalog));
        });
    });
    #[cfg(feature = "rayon")]
    g.bench_function("parallel", |b| {
        b.iter(|| {
            black_box(par_match_ranked(&matcher, &query, &catalog));
        });
    });
    g.finish();
}

fn bench_wang_index_insert(c: &mut Criterion) {
    // Incremental enroll: insert one 3 s fingerprint into a warm n=100
    // index. Compare against full rebuild of n=101 to show the scaling
    // (insert is O(hashes), rebuild is O(catalog)).
    let refs: Vec<WangFingerprint> = (0..100u32).map(|i| wang_fp(10 + i, 3)).collect();
    let new_fp = wang_fp(999, 3);

    let mut g = c.benchmark_group("matching/wang_index_insert");
    g.bench_function("insert_one_into_n100", |b| {
        b.iter_batched(
            || WangIndex::build(&refs, 100),
            |mut index| black_box(index.insert(black_box(&new_fp), 100)),
            criterion::BatchSize::SmallInput,
        );
    });
    let refs101: Vec<WangFingerprint> = (0..101u32).map(|i| wang_fp(10 + i, 3)).collect();
    g.bench_function("rebuild_n101", |b| {
        b.iter(|| black_box(WangIndex::build(black_box(&refs101), 100)));
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_wang_one,
    bench_haitsma_one,
    bench_panako_one,
    bench_wang_index,
    bench_wang_index_insert
);
criterion_main!(benches);
