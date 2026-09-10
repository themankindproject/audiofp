//! Watermark plan-cache microbench (audit §4.2 #11).
//!
//! ```bash
//! cargo bench --bench watermark_plans --features watermark
//! ```
//!
//! Measures `WatermarkDetector::detect` for two access patterns over the
//! same total call count:
//!
//! * `same_length` — every call uses one length, so the plan is built once.
//! * `alternating` — calls alternate between two lengths. With the old
//!   single-slot cache this rebuilt the tract plan (clone + optimise) on
//!   *every* call; with the LRU it should behave like `same_length`.
//!
//! The two groups must land within noise of each other after the fix. The
//! audit measured 33 ms (same) vs 195 ms (alternating) before it.
//!
//! The model is an identity graph built here with prost, so the bench needs
//! no committed ONNX weights and no download.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use tract_onnx::pb;

use audiofp::SampleRate;
use audiofp::watermark::{WatermarkConfig, WatermarkDetector};

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

fn bench_watermark_plans(c: &mut Criterion) {
    let path = write_model();
    let rate = SampleRate::new(16_000).expect("rate");
    let a: Vec<f32> = vec![0.5; 4096];
    let b: Vec<f32> = vec![0.5; 2048];

    let mut g = c.benchmark_group("watermark/plan_cache");

    // One length throughout: the plan is built once and reused.
    g.bench_function("same_length", |bencher| {
        let mut det = detector(&path);
        // Warm the plan so the measured loop is pure reuse.
        det.detect(&a, rate).expect("warmup detect");
        bencher.iter(|| {
            black_box(det.detect(black_box(&a), rate).expect("detect"));
        });
    });

    // Two lengths alternating. This is the case the LRU fixes: the old
    // single-slot cache evicted on every switch, so every call paid a
    // full clone + optimise of the graph.
    g.bench_function("alternating_two_lengths", |bencher| {
        let mut det = detector(&path);
        det.detect(&a, rate).expect("warmup a");
        det.detect(&b, rate).expect("warmup b");
        let mut flip = false;
        bencher.iter(|| {
            flip = !flip;
            let s = if flip { &a } else { &b };
            black_box(det.detect(black_box(s), rate).expect("detect"));
        });
    });

    g.finish();
    let _ = std::fs::remove_file(&path);
}

criterion_group!(benches, bench_watermark_plans);
criterion_main!(benches);
