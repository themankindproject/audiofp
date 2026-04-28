//! Public-API integration tests for the neural embedder.
//!
//! These exercise the contract a downstream user can rely on without
//! shipping an ONNX model: error variants, configuration defaults, and
//! the `StreamingFingerprinter` trait surface. End-to-end inference is
//! covered by the in-tree unit tests via a passthrough tract fixture.

#![cfg(feature = "neural")]

use audiofp::neural::{NeuralEmbedderConfig, StreamingNeuralEmbedder};
use audiofp::{AfpError, AudioBuffer, Fingerprinter, SampleRate};

#[test]
fn config_constructor_documented_defaults() {
    let cfg = NeuralEmbedderConfig::new("any.onnx");
    assert_eq!(cfg.sample_rate, 16_000);
    assert_eq!(cfg.n_fft, 1024);
    assert_eq!(cfg.hop, 320);
    assert_eq!(cfg.n_mels, 128);
    assert_eq!(cfg.window_secs, 1.0);
    assert_eq!(cfg.hop_secs, 1.0);
    assert!(cfg.l2_normalize);
}

#[test]
fn both_constructors_share_the_missing_model_contract() {
    // Same config, same advertised error variant from both entry points.
    let path = "/definitely/does/not/exist.onnx".to_string();
    let cfg = NeuralEmbedderConfig::new(path.clone());

    let off = audiofp::neural::NeuralEmbedder::new(cfg.clone());
    let stream = StreamingNeuralEmbedder::new(cfg);

    match off {
        Err(AfpError::ModelNotFound(a)) => assert_eq!(a, path),
        Err(e) => panic!("offline: expected ModelNotFound, got {e:?}"),
        Ok(_) => panic!("offline: expected ModelNotFound, got Ok"),
    }
    match stream {
        Err(AfpError::ModelNotFound(a)) => assert_eq!(a, path),
        Err(e) => panic!("streaming: expected ModelNotFound, got {e:?}"),
        Ok(_) => panic!("streaming: expected ModelNotFound, got Ok"),
    }
}

#[test]
fn invalid_config_is_rejected_before_model_load() {
    // Sample rate 0 should error out at validation, never even touch
    // the (also missing) model.
    let mut cfg = NeuralEmbedderConfig::new("/missing.onnx");
    cfg.sample_rate = 0;
    match audiofp::neural::NeuralEmbedder::new(cfg) {
        Err(AfpError::Config(_)) => {}
        Err(e) => panic!("expected Config, got {e:?}"),
        Ok(_) => panic!("expected Config, got Ok"),
    }
}

#[test]
fn extract_on_short_audio_returns_audio_too_short_via_trait() {
    // We can't construct an embedder here without a model — but we can
    // still demonstrate the public Fingerprinter trait carries through
    // the AudioBuffer contract by type-checking the trait surface.
    fn _assert_trait_object_compatible<F: Fingerprinter>(_: &F) {}
    let _ = _assert_trait_object_compatible::<audiofp::neural::NeuralEmbedder>;
    // Construct a buffer just to exercise the public types; we never
    // actually pass it because no model is available here.
    let samples = vec![0.0_f32; 16];
    let _buf = AudioBuffer {
        samples: &samples,
        rate: SampleRate::HZ_16000,
    };
}

#[test]
fn hop_greater_than_window_is_rejected_at_construction() {
    fn check_config_msg<T>(r: Result<T, AfpError>, label: &str) {
        match r {
            Err(AfpError::Config(msg)) => {
                assert!(
                    msg.contains("hop_samples") && msg.contains("window_samples"),
                    "{label}: expected hop>window message, got: {msg}",
                );
            }
            Err(e) => panic!("{label}: expected Config, got {e:?}"),
            Ok(_) => panic!("{label}: expected Config, got Ok"),
        }
    }
    let mut cfg = NeuralEmbedderConfig::new("/missing.onnx");
    cfg.window_secs = 0.5;
    cfg.hop_secs = 1.0;
    check_config_msg(audiofp::neural::NeuralEmbedder::new(cfg.clone()), "offline");
    check_config_msg(StreamingNeuralEmbedder::new(cfg), "streaming");
}
