//! Re-verify the inverse-STFT / overlap-add round-trip fidelity cited in the
//! issue: a correct synthesis path is a prerequisite for the whole design.

use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;

fn main() {
    let sr = 16_000u32;
    let tmp = std::env::temp_dir().join("rt_verify.wav");
    std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(Path::new("assets/galway.flac"))
        .args(["-ac", "1", "-ar", &sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(&tmp)
        .status()
        .unwrap();
    let x = io::read_wav_mono_f32(&tmp).unwrap();

    for &(n_fft, hop) in &[(2048usize, 512usize), (1024, 256), (2048, 1024)] {
        let mut stft = dsp::Stft::new(n_fft, hop);
        let (spec, nf) = stft.analyze(&x);
        let mut istft = dsp::Istft::new(n_fft, hop);
        let y = istft.synthesize(&spec, nf, x.len());
        // Exclude the first/last frame (edge padding).
        let a = &x[n_fft..x.len() - n_fft];
        let b = &y[n_fft..y.len() - n_fft];
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let mut maxd = 0.0f32;
        for i in 0..a.len() {
            let d = (a[i] - b[i]) as f64;
            num += d * d;
            den += (a[i] as f64) * (a[i] as f64);
            if (a[i] - b[i]).abs() > maxd {
                maxd = (a[i] - b[i]).abs();
            }
        }
        println!(
            "n_fft={n_fft} hop={hop}: round-trip SNR = {:.1} dB (max abs err {:.2e})",
            10.0 * (den / num).log10(),
            maxd
        );
    }
}
