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
use crate::dsp::mel::{MelFilterBank, MelScale};
use crate::dsp::stft::{ShortTimeFFT, StftConfig};
use crate::dsp::windows::WindowKind;

use super::embedder::{EmbedderCore, NeuralEmbedder, NeuralEmbedderConfig, Runnable};
use super::frontend::LogMelFrontend;
use super::streaming::StreamingNeuralEmbedder;

/// Build a tract `Runnable` whose single output equals its single input.
fn build_passthrough_runnable(n_mels: usize, n_frames: usize) -> Runnable {
    let mut model = TypedModel::default();
    let input = model
        .add_source("x", f32::fact([1, n_mels, n_frames]))
        .expect("add_source");
    model
        .set_output_outlets(&[input])
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
    use crate::AfpError;
    use alloc::format;
    use alloc::string::ToString;

    if cfg.sample_rate == 0 {
        return Err(AfpError::Config("sample_rate must be > 0".to_string()));
    }
    if cfg.n_fft < 2 || !cfg.n_fft.is_power_of_two() {
        return Err(AfpError::Config(format!(
            "n_fft must be a power of two >= 2 (got {})",
            cfg.n_fft,
        )));
    }
    if cfg.hop == 0 || cfg.hop > cfg.n_fft {
        return Err(AfpError::Config(format!(
            "hop must satisfy 0 < hop <= n_fft (hop={}, n_fft={})",
            cfg.hop, cfg.n_fft,
        )));
    }
    if cfg.n_mels == 0 {
        return Err(AfpError::Config("n_mels must be > 0".to_string()));
    }
    let nyquist = cfg.sample_rate as f32 / 2.0;
    if !(cfg.fmin >= 0.0 && cfg.fmax > cfg.fmin && cfg.fmax <= nyquist) {
        return Err(AfpError::Config("fmin/fmax out of range".to_string()));
    }
    let window_samples = (cfg.window_secs * cfg.sample_rate as f32).round() as usize;
    let hop_samples = (cfg.hop_secs * cfg.sample_rate as f32).round() as usize;
    if window_samples < cfg.n_fft || hop_samples == 0 || hop_samples > window_samples {
        return Err(AfpError::Config("window/hop sizing invalid".to_string()));
    }
    let n_frames = (window_samples - cfg.n_fft) / cfg.hop + 1;

    let runnable = build_passthrough_runnable(cfg.n_mels, n_frames);
    let stft = ShortTimeFFT::new(StftConfig {
        n_fft: cfg.n_fft,
        hop: cfg.hop,
        window: cfg.window_kind,
        center: false,
    });
    let mel = MelFilterBank::new(
        cfg.n_mels,
        cfg.n_fft,
        cfg.sample_rate,
        cfg.fmin,
        cfg.fmax,
        cfg.mel_scale,
    );
    let frontend = LogMelFrontend::new(stft, mel, window_samples);
    let embedding_dim = cfg.n_mels * n_frames;

    Ok(EmbedderCore {
        cfg,
        frontend,
        runnable,
        window_samples,
        hop_samples,
        n_frames,
        embedding_dim,
    })
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
    }
}
