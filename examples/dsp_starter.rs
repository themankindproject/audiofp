//! Build a custom analysis chain from public DSP primitives.
//!
//! ```bash
//! cargo run --example dsp_starter
//! ```
//!
//! No audio file and no optional features required: synthesises a 440 Hz
//! tone, runs STFT → log-mel → peak pick, and prints a short summary.

use core::f32::consts::PI;

use audiofp::dsp::mel::{MelFilterBank, MelScale};
use audiofp::dsp::peaks::{PeakPicker, PeakPickerConfig};
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
use audiofp::dsp::windows::WindowKind;

fn main() {
    let sample_rate = 16_000u32;
    let secs = 1.0_f32;
    let n = (sample_rate as f32 * secs) as usize;
    let freq = 440.0_f32;

    let samples: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq * (i as f32) / sample_rate as f32).sin() * 0.5)
        .collect();

    let n_fft = 512;
    let hop = 160; // 10 ms at 16 kHz
    let mut stft = ShortTimeFFT::new(StftConfig {
        n_fft,
        hop,
        window: WindowKind::Hann,
        center: true,
    });

    let (power, n_frames, n_bins) = stft.power_flat(&samples);
    println!("STFT: {n_frames} frames × {n_bins} bins (n_fft={n_fft}, hop={hop})");

    let n_mels = 64;
    let mel = MelFilterBank::new(
        n_mels,
        n_fft,
        sample_rate,
        0.0,
        sample_rate as f32 / 2.0,
        MelScale::Slaney,
    );
    let mut log_mel = vec![0.0_f32; n_frames * n_mels];
    for f in 0..n_frames {
        let row = &power[f * n_bins..(f + 1) * n_bins];
        mel.log_mel_from_power(row, &mut log_mel[f * n_mels..(f + 1) * n_mels]);
    }
    println!("Mel: {n_frames} frames × {n_mels} bands");

    // Peak-pick on the mel spectrogram (linear-ish log-mel values; floor is loose).
    let frames_per_sec = sample_rate as f32 / hop as f32;
    let mut picker = PeakPicker::new(PeakPickerConfig {
        neighborhood_t: 2,
        neighborhood_f: 2,
        min_magnitude: -20.0,
        target_per_sec: 40,
    });
    let peaks = picker.pick(&log_mel, n_frames, n_mels, frames_per_sec);

    println!(
        "Peaks: {} (~{:.1} / s at {:.1} fps)",
        peaks.len(),
        peaks.len() as f32 / secs,
        frames_per_sec
    );
    for p in peaks.iter().take(8) {
        println!("  t_frame={} f_bin={} mag={:.2}", p.t_frame, p.f_bin, p.mag);
    }
}
