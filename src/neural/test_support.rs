//! Test fixtures: build a tract `Runnable` in-process so we can exercise
//! the full embedder + streaming code paths without shipping an ONNX
//! file.
//!
//! Strategy: a passthrough model whose single output is its single input.
//! Output shape equals input shape `[1, n_mels, n_frames]`, so the flat
//! embedding length is `n_mels * n_frames` — the embedder's L2 norm and
//! windowing logic are exercised, and offline/streaming bit-exactness
//! becomes verifiable end-to-end.

use alloc::vec;
use alloc::vec::Vec;

use tract_onnx::prelude::*;

use crate::Result;
use crate::dsp::mel::MelScale;
use crate::dsp::windows::WindowKind;

use super::embedder::{EmbedderCore, NeuralEmbedder, NeuralEmbedderConfig, Runnable};
use super::streaming::StreamingNeuralEmbedder;

/// Build a tract `Runnable` whose single output equals its single input.
fn build_passthrough_runnable(n_mels: usize, n_frames: usize) -> Runnable {
    let mut model = TypedModel::default();
    let input = model
        .add_source("x", f32::fact([1, n_mels, n_frames]))
        .expect("add_source");
    model
        .select_output_outlets(&[input])
        .expect("set_output_outlets");
    model
        .into_optimized()
        .expect("optimize")
        .into_runnable()
        .expect("runnable")
}

/// Build a batched tract `Runnable` whose single output equals its input.
fn build_passthrough_runnable_batched(batch: usize, n_mels: usize, n_frames: usize) -> Runnable {
    let mut model = TypedModel::default();
    let input = model
        .add_source("x", f32::fact([batch, n_mels, n_frames]))
        .expect("add_source");
    model
        .select_output_outlets(&[input])
        .expect("set_output_outlets");
    model
        .into_optimized()
        .expect("optimize")
        .into_runnable()
        .expect("runnable")
}

/// Build a `NeuralEmbedder` backed by a passthrough model. Skips ONNX
/// load + probe entirely — useful for exercising the front-end +
/// windowing in tests.
pub(crate) fn passthrough_embedder(cfg: NeuralEmbedderConfig) -> Result<NeuralEmbedder> {
    let core = passthrough_core(cfg)?;
    Ok(NeuralEmbedder { core })
}

/// Build a `StreamingNeuralEmbedder` backed by a passthrough model.
pub(crate) fn passthrough_streaming(cfg: NeuralEmbedderConfig) -> Result<StreamingNeuralEmbedder> {
    let core = passthrough_core(cfg)?;
    Ok(StreamingNeuralEmbedder::__from_core_for_test(core))
}

fn passthrough_core(cfg: NeuralEmbedderConfig) -> Result<EmbedderCore> {
    let geometry = EmbedderCore::geometry(&cfg)?;
    let n_frames = geometry.n_frames;
    let runnable = build_passthrough_runnable(cfg.n_mels, n_frames);
    let batch_runnable = if cfg.batch_size > 1 {
        Some(build_passthrough_runnable_batched(
            cfg.batch_size,
            cfg.n_mels,
            n_frames,
        ))
    } else {
        None
    };
    let embedding_dim = cfg.n_mels * n_frames;
    Ok(EmbedderCore::from_parts(
        cfg,
        geometry,
        runnable,
        batch_runnable,
        embedding_dim,
    ))
}

/// Synthetic deterministic PCM: sum of two sines + low-amplitude noise.
pub(crate) fn synth_audio(seed: u32, n: usize, sr: u32) -> Vec<f32> {
    let f1 = 440.0_f32;
    let f2 = 1100.0_f32;
    let mut out = vec![0.0_f32; n];
    let mut state = seed.wrapping_add(1);
    for (i, s) in out.iter_mut().enumerate() {
        // xorshift32 for cheap deterministic noise.
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = (state as f32 / u32::MAX as f32 - 0.5) * 0.05;
        let t = i as f32 / sr as f32;
        *s = (2.0 * core::f32::consts::PI * f1 * t).sin() * 0.6
            + (2.0 * core::f32::consts::PI * f2 * t).sin() * 0.3
            + noise;
    }
    out
}

/// A small, fast config for tests that still hits real STFT + mel paths.
pub(crate) fn small_cfg() -> NeuralEmbedderConfig {
    NeuralEmbedderConfig {
        model_path: "test-fixture".into(), // never actually loaded
        sample_rate: 16_000,
        n_fft: 256,
        hop: 128,
        n_mels: 8,
        fmin: 0.0,
        fmax: 8_000.0,
        mel_scale: MelScale::Slaney,
        window_kind: WindowKind::Hann,
        window_secs: 0.25, // 4_000 samples → (4000-256)/128 + 1 = 30 frames
        hop_secs: 0.25,
        l2_normalize: true,
        max_input_samples: None,
        max_push_samples: None,
        batch_size: 1,
    }
}
