//! Watermark plan-cache microbench (audit §4.2 #11 / F23).
//!
//! ```bash
//! cargo bench --bench watermark_plans --features watermark
//! ```
//!
//! Compares two access patterns with **equal total samples processed per
//! timed iteration** (6144 = 4096 + 2048):
//!
//! * `same_length` — one `detect` on a 6144-sample buffer (one plan).
//! * `alternating_two_lengths` — `detect(4096)` then `detect(2048)` (two
//!   plans, LRU exercise) in the same iteration.
//!
//! Throughput claims must not compare unequal work. Correctness of cache
//! behaviour (stable results across patterns) is checked in `#[cfg(test)]`
//! below; plan build/eviction counts are not observable from the public API
//! without modifying the detector.
//!
//! The identity ONNX stub exercises plan-cache plumbing only. Absolute timings
//! and any relative speedup between arms are model-dependent and must not be
//! treated as production watermark performance.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use tract_onnx::pb;

use audiofp::SampleRate;
use audiofp::watermark::{WatermarkConfig, WatermarkDetector};

/// Samples processed per timed iteration in both bench arms.
const SAMPLES_PER_ITER: usize = 6144;
const LEN_A: usize = 4096;
const LEN_B: usize = 2048;

/// Two f32 input dims plus a symbolic time dim.
fn f32_tensor(shapes: Vec<Option<i64>>) -> pb::type_proto::Tensor {
    pb::type_proto::Tensor {
        elem_type: 1,
        shape: Some(pb::TensorShapeProto {
            dim: shapes
                .into_iter()
                .map(|d| pb::tensor_shape_proto::Dimension {
                    value: Some(match d {
                        Some(v) => pb::tensor_shape_proto::dimension::Value::DimValue(v),
                        None => pb::tensor_shape_proto::dimension::Value::DimParam("T".into()),
                    }),
                    denotation: String::new(),
                })
                .collect(),
        }),
    }
}

fn value_info(name: &str) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(f32_tensor(vec![
                Some(1),
                Some(1),
                None,
            ]))),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

fn identity_node(name: &str, out: &str) -> pb::NodeProto {
    pb::NodeProto {
        input: vec!["x".to_string()],
        output: vec![out.to_string()],
        name: name.to_string(),
        op_type: "Identity".to_string(),
        ..Default::default()
    }
}

/// Write a two-output identity model (`det`, `msg` both alias the input).
fn write_model() -> std::path::PathBuf {
    let graph = pb::GraphProto {
        node: vec![identity_node("d", "det"), identity_node("m", "msg")],
        name: "identity_wm".to_string(),
        input: vec![value_info("x")],
        output: vec![value_info("det"), value_info("msg")],
        ..Default::default()
    };
    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    let path = std::env::temp_dir().join(format!("audiofp_wmbench_{}.onnx", std::process::id()));
    let bytes = prost::Message::encode_to_vec(&model);
    std::fs::write(&path, bytes).expect("write bench onnx");
    path
}

fn detector(path: &std::path::Path) -> WatermarkDetector {
    WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()))
        .expect("load bench onnx")
}

fn verify_equal_work_and_cache_correctness(path: &std::path::Path) {
    assert_eq!(
        LEN_A + LEN_B,
        SAMPLES_PER_ITER,
        "bench arms must process equal samples per iteration"
    );
    let rate = SampleRate::new(16_000).expect("rate");
    let uniform: Vec<f32> = vec![0.75; SAMPLES_PER_ITER];
    let a: Vec<f32> = vec![0.75; LEN_A];
    let b: Vec<f32> = vec![0.75; LEN_B];
    let mut det = detector(path);
    let r_uniform = det.detect(&uniform, rate).expect("uniform detect");
    let r_a = det.detect(&a, rate).expect("a detect");
    let r_b = det.detect(&b, rate).expect("b detect");
    assert!(
        (r_uniform.confidence - 0.75).abs() < 1e-4,
        "identity model confidence drift on uniform buffer"
    );
    assert!(
        (r_a.confidence - 0.75).abs() < 1e-4 && (r_b.confidence - 0.75).abs() < 1e-4,
        "identity model confidence drift on split buffers"
    );
    assert_eq!(
        r_uniform.message, r_a.message,
        "message must not depend on cache path"
    );
}

fn bench_watermark_plans(c: &mut Criterion) {
    let path = write_model();
    verify_equal_work_and_cache_correctness(&path);
    let rate = SampleRate::new(16_000).expect("rate");
    assert_eq!(LEN_A + LEN_B, SAMPLES_PER_ITER);
    let uniform: Vec<f32> = vec![0.5; SAMPLES_PER_ITER / 2];
    let a: Vec<f32> = vec![0.5; LEN_A];
    let b: Vec<f32> = vec![0.5; LEN_B];

    let mut g = c.benchmark_group("watermark/plan_cache");
    g.throughput(criterion::Throughput::Elements(SAMPLES_PER_ITER as u64));

    // Two calls of one length: 3072 + 3072 samples per timed iteration.
    g.bench_function("same_length", |bencher| {
        let mut det = detector(&path);
        det.detect(&uniform, rate).expect("warmup detect");
        bencher.iter(|| {
            black_box(det.detect(black_box(&uniform), rate).expect("detect"));
            black_box(det.detect(black_box(&uniform), rate).expect("detect"));
        });
    });

    // Two distinct lengths, equal total samples per iteration (4096 + 2048).
    g.bench_function("alternating_two_lengths", |bencher| {
        let mut det = detector(&path);
        det.detect(&a, rate).expect("warmup a");
        det.detect(&b, rate).expect("warmup b");
        bencher.iter(|| {
            black_box(det.detect(black_box(&a), rate).expect("detect a"));
            black_box(det.detect(black_box(&b), rate).expect("detect b"));
        });
    });

    g.finish();
    let _ = std::fs::remove_file(&path);
}

criterion_group!(benches, bench_watermark_plans);
criterion_main!(benches);
