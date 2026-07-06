//! Public-API integration tests for the watermark detector.
//!
//! These exercise the public contract: error handling, configuration
//! validation, and result types. End-to-end inference is not tested here
//! because it requires a shipped ONNX model; in-tree unit tests cover
//! Tract wiring via corrupt-file validation.

#![cfg(feature = "watermark")]

use audiofp::watermark::{WatermarkConfig, WatermarkDetector};
use audiofp::AfpError;

#[test]
fn config_new_uses_audioseal_defaults() {
    let cfg = WatermarkConfig::new("model.onnx");
    assert_eq!(cfg.message_bits, 16);
    assert_eq!(cfg.threshold, 0.5);
    assert_eq!(cfg.sample_rate, 16_000);
    assert_eq!(cfg.model_path, "model.onnx");
}

#[test]
fn empty_model_path_returns_model_not_found() {
    let res = WatermarkDetector::new(WatermarkConfig::new(""));
    match res {
        Err(AfpError::ModelNotFound(path)) => assert!(path.is_empty()),
        Ok(_) => panic!("expected ModelNotFound, got Ok"),
        Err(e) => panic!("expected ModelNotFound, got Err({e:?})"),
    }
}

#[test]
fn missing_model_file_returns_model_not_found() {
    let cfg = WatermarkConfig::new("/nonexistent/path/to/audioseal.onnx");
    let res = WatermarkDetector::new(cfg);
    match res {
        Err(AfpError::ModelNotFound(p)) => {
            assert_eq!(p, "/nonexistent/path/to/audioseal.onnx");
        }
        Ok(_) => panic!("expected ModelNotFound, got Ok"),
        Err(e) => panic!("expected ModelNotFound, got Err({e:?})"),
    }
}

#[test]
fn message_bits_above_32_is_config_error() {
    let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
    cfg.message_bits = 33;
    match WatermarkDetector::new(cfg) {
        Err(AfpError::Config(msg)) => assert!(msg.contains("message_bits")),
        Ok(_) => panic!("expected Config error, got Ok"),
        Err(e) => panic!("expected Config error, got Err({e:?})"),
    }
}

#[test]
fn threshold_outside_unit_interval_is_config_error() {
    for bad in [-0.1_f32, 1.01, 2.0, -100.0] {
        let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
        cfg.threshold = bad;
        match WatermarkDetector::new(cfg) {
            Err(AfpError::Config(msg)) => assert!(msg.contains("threshold")),
            Ok(_) => panic!("expected Config for threshold={bad}, got Ok"),
            Err(e) => panic!("expected Config for threshold={bad}, got Err({e:?})"),
        }
    }
}

#[test]
fn zero_sample_rate_is_config_error() {
    let mut cfg = WatermarkConfig::new("/tmp/dummy.onnx");
    cfg.sample_rate = 0;
    match WatermarkDetector::new(cfg) {
        Err(AfpError::Config(msg)) => assert!(msg.contains("sample_rate")),
        Ok(_) => panic!("expected Config error, got Ok"),
        Err(e) => panic!("expected Config error, got Err({e:?})"),
    }
}

#[test]
fn corrupt_file_returns_model_load_error() {
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "audiofp-watermark-integ-{}.bin",
        std::process::id()
    ));
    {
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03,
                      0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03]).unwrap();
    }
    let res = WatermarkDetector::new(WatermarkConfig::new(path.to_string_lossy().into_owned()));
    std::fs::remove_file(&path).ok();
    match res {
        Err(AfpError::ModelLoad(_)) => {}
        Ok(_) => panic!("expected ModelLoad, got Ok"),
        Err(e) => panic!("expected ModelLoad, got Err({e:?})"),
    }
}
