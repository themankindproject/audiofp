//! Microbenches for the neural embedder hot path.
//!
//! Focuses on work that scales with audio duration and that perf
//! changes to the embedder will touch:
//!
//! - `log_mel_pipeline`: STFT + log-mel on one analysis window. This is
//!   the front-end loop in [`audiofp::neural::NeuralEmbedder`] (one
//!   iteration per second of audio at default settings).
//! - `strided_tensor_write`: the strided fill that copies log-mel rows
//!   into the model input tensor's `[1, n_mels, n_frames]` layout.
//! - `l2_normalize`: L2 normalisation of one embedding vector.
//!
//! Run with:
//! ```bash
//! cargo bench --features neural --bench neural_frontend
//! ```

#![cfg(feature = "neural")]

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use audiofp::dsp::mel::{MelFilterBank, MelScale};
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
use audiofp::dsp::windows::WindowKind;

const SR: u32 = 16_000;
const N_FFT: usize = 1024;
const HOP: usize = 320;
const N_MELS: usize = 128;
const WINDOW_SAMPLES: usize = SR as usize; // 1 s
const N_FRAMES: usize = (WINDOW_SAMPLES - N_FFT) / HOP + 1;
const EMBEDDING_DIM: usize = 1024; // realistic for VGGish/YAMNet/OpenL3-class models

fn synth(seed: u32, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut x = seed.max(1);
    for i in 0..n {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let noise = (x as i32 as f32) / (i32::MAX as f32) * 0.05;
        let t = i as f32 / SR as f32;
        let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin()
            + noise;
        out.push(s);
    }
    out
}

fn bench_log_mel_pipeline(c: &mut Criterion) {
    let mut stft = ShortTimeFFT::new(StftConfig {
        n_fft: N_FFT,
        hop: HOP,
        window: WindowKind::Hann,
        center: false,
    });
    let mel = MelFilterBank::new(N_MELS, N_FFT, SR, 0.0, SR as f32 / 2.0, MelScale::Slaney);
    let window = synth(1, WINDOW_SAMPLES);
    let mut power = vec![0.0_f32; N_FFT / 2 + 1];
    let mut mel_row = vec![0.0_f32; N_MELS];

    c.bench_function("neural_frontend/log_mel_pipeline_1s_window", |b| {
        b.iter(|| {
            for f in 0..N_FRAMES {
                let frame = &window[f * HOP..f * HOP + N_FFT];
                stft.process_frame_power(frame, &mut power);
                mel.log_mel_from_power(&power, &mut mel_row);
                black_box(&mel_row);
            }
        });
    });
}

fn bench_strided_tensor_write(c: &mut Criterion) {
    // Mimics `EmbedderCore::embed_window_into`'s inner strided fill —
    // the work that #12 (`get_unchecked`) would touch.
    let mel_per_frame = synth(2, N_MELS);
    let mut dst = vec![0.0_f32; N_MELS * N_FRAMES];

    c.bench_function("neural_frontend/strided_tensor_write", |b| {
        b.iter(|| {
            for f in 0..N_FRAMES {
                for m in 0..N_MELS {
                    dst[m * N_FRAMES + f] = mel_per_frame[m];
                }
            }
            black_box(&dst);
        });
    });
}

fn bench_l2_normalize(c: &mut Criterion) {
    let mut v = synth(3, EMBEDDING_DIM);

    c.bench_function("neural_frontend/l2_normalize_1024d", |b| {
        b.iter(|| {
            let sumsq: f32 = v.iter().map(|x| x * x).sum();
            let norm = sumsq.sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
            black_box(&v);
        });
    });
}

criterion_group!(
    neural_frontend,
    bench_log_mel_pipeline,
    bench_strided_tensor_write,
    bench_l2_normalize,
);
criterion_main!(neural_frontend);
