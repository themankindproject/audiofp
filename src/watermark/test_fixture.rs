//! ONNX fixture builder for tests that need a *real* model file on disk.
//!
//! The crate ships no ONNX weights, and `AUDIT.md` §4.3 #13 records the
//! consequence: the watermark `detect()` body had no positive-path test,
//! because every existing test only asserted construction / validation
//! errors. tract's `tract_onnx::pb` module re-exports the prost-generated
//! ONNX protobuf types, so a minimal but genuinely valid model can be
//! serialised here and written to a temp path — no committed binary and no
//! external download.
//!
//! The model is an identity: one input `x` of shape `[1, 1, T]` (with `T`
//! symbolic, like a real AudioSeal export), and two outputs `det` and `msg`
//! that both alias it. `WatermarkDetector::detect` requires **≥ 2 outputs**
//! (detection scores, then message logits), so two `Identity` nodes are
//! needed. Every step of the positive path — load, `with_input_fact`,
//! `into_typed`, `into_optimized`, `into_runnable`, run, decode bits —
//! needs only a satisfiable graph, not real AudioSeal weights.
//!
//! Compiled only for tests of the `watermark` feature.

use std::path::PathBuf;

use tract_onnx::pb;

/// A unique temp path for a generated ONNX model.
fn unique_path(stem: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "audiofp_wmfix_{stem}_{}_{n}.onnx",
        std::process::id()
    ))
}

/// f32 tensor type proto with the given dimensions. A `None` entry becomes a
/// symbolic dimension, so the shape can be re-facted per call.
fn f32_tensor(shapes: Vec<Option<i64>>) -> pb::type_proto::Tensor {
    pb::type_proto::Tensor {
        elem_type: 1, // TensorProto::FLOAT
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

fn value_info(name: &str, shape: Vec<Option<i64>>) -> pb::ValueInfoProto {
    pb::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(pb::TypeProto {
            value: Some(pb::type_proto::Value::TensorType(f32_tensor(shape))),
            denotation: String::new(),
        }),
        doc_string: String::new(),
    }
}

fn identity_node(op_name: &str, out: &str) -> pb::NodeProto {
    pb::NodeProto {
        input: vec!["x".to_string()],
        output: vec![out.to_string()],
        name: op_name.to_string(),
        op_type: "Identity".to_string(),
        ..Default::default()
    }
}

/// Write a two-output identity ONNX model to a temp file and return the path.
///
/// Output 0 (`det`) feeds `WatermarkResult::localization` / `confidence`,
/// output 1 (`msg`) feeds the message-bit decode. Both alias the input, so
/// with input `[1, 1, n]` the confidence is `mean(samples)` and the message
/// bits are read from the first `message_bits` samples.
pub fn write_identity_onnx(stem: &str) -> PathBuf {
    let graph = pb::GraphProto {
        node: vec![
            identity_node("det_node", "det"),
            identity_node("msg_node", "msg"),
        ],
        name: "identity_wm".to_string(),
        input: vec![value_info("x", vec![Some(1), Some(1), None])],
        output: vec![
            value_info("det", vec![Some(1), Some(1), None]),
            value_info("msg", vec![Some(1), Some(1), None]),
        ],
        ..Default::default()
    };

    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "audiofp-test-fixture".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    let path = unique_path(stem);
    // `prost::Message` is implemented for the generated types.
    let bytes = prost::Message::encode_to_vec(&model);
    std::fs::write(&path, bytes).expect("write onnx fixture");
    path
}

/// Write an ONNX model whose detection output is a length-`n` constant tensor
/// (`det_value` at every position) and whose message output aliases the input.
pub fn write_constant_detection_onnx(stem: &str, n: usize, det_value: f32) -> PathBuf {
    let det_bytes = det_value.to_le_bytes().repeat(n);
    let graph = pb::GraphProto {
        initializer: vec![pb::TensorProto {
            name: "det_const".to_string(),
            data_type: 1,
            dims: vec![1, 1, n as i64],
            raw_data: det_bytes,
            ..Default::default()
        }],
        node: vec![
            pb::NodeProto {
                input: vec!["det_const".to_string()],
                output: vec!["det".to_string()],
                name: "det_identity".to_string(),
                op_type: "Identity".to_string(),
                ..Default::default()
            },
            identity_node("msg_node", "msg"),
        ],
        name: "constant_det_wm".to_string(),
        input: vec![value_info("x", vec![Some(1), Some(1), None])],
        output: vec![
            value_info("det", vec![Some(1), Some(1), Some(n as i64)]),
            value_info("msg", vec![Some(1), Some(1), None]),
        ],
        ..Default::default()
    };

    let model = pb::ModelProto {
        ir_version: 8,
        opset_import: vec![pb::OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        producer_name: "audiofp-test-fixture".to_string(),
        graph: Some(graph),
        ..Default::default()
    };

    let path = unique_path(stem);
    let bytes = prost::Message::encode_to_vec(&model);
    std::fs::write(&path, bytes).expect("write onnx fixture");
    path
}

/// Remove a fixture written by [`write_identity_onnx`] or
/// [`write_constant_detection_onnx`].
pub fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}
