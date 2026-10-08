//! Detector-statistic study: does a self-normalising z-score (computed over
//! the phase ensemble) restore the separation that the max-over-phases sync
//! search destroys? This is the calibration recommendation for the issue.

use audiofp_wm_bakeoff::algo::{Candidate, MagDsss};
use audiofp_wm_bakeoff::{dsp, io};
use std::path::Path;

fn dec(src: &Path, sr: u32) -> Vec<f32> {
    let t = std::env::temp_dir().join("zscore.wav");
    std::process::Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-ac", "1", "-ar", &sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(&t)
        .status()
        .unwrap();
    io::read_wav_mono_f32(&t).unwrap()
}

fn ber(a: &[u8], b: &[u8]) -> f64 {
    (0..a.len()).filter(|&i| a[i] != b[i]).count() as f64 / a.len() as f64
}

/// Per-bit correlations for one trial phase.
fn correlate(
    spec: &[rustfft::num_complex::Complex<f32>],
    n_frames: usize,
    nb: usize,
    order: &[usize],
    bin_mean: &[f64],
    key: &[u8],
    ph0: usize,
    n_bits: usize,
) -> Vec<f64> {
    let mut corr = vec![0.0f64; n_bits];
    for f in 0..n_frames {
        let b = (f + ph0) % n_bits;
        let mut r = dsp::ChaCha::new(&dsp::derive_key(key, b"mag-spread"), b as u64);
        for &bin in order.iter() {
            let chip = r.bipolar() as f64;
            let resid = spec[f * nb + bin].norm() as f64 - bin_mean[bin];
            corr[b] += (resid / bin_mean[bin].max(1e-9)) * chip;
        }
    }
    corr
}

/// Returns (max-over-phases mean|corr|, z-score of the winning phase).
fn detect_stats(samples: &[f32], key: &[u8], c: &MagDsss) -> (f64, f64, Vec<u8>) {
    let n_bits = 64usize;
    let mut stft = dsp::Stft::new(c.n_fft, c.hop);
    let (spec, n_frames) = stft.analyze(samples);
    let nb = c.n_fft / 2 + 1;
    let bin_hi = ((nb as f32) * c.bin_hi_frac) as usize;
    let mut rng = dsp::ChaCha::new(&dsp::derive_key(key, b"mag-bins"), 0);
    let mut order: Vec<usize> = (c.bin_lo..bin_hi).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    order.truncate(c.bins_per_frame);

    let mut bin_mean = vec![0.0f64; nb];
    for f in 0..n_frames {
        for &bin in order.iter() {
            bin_mean[bin] += spec[f * nb + bin].norm() as f64;
        }
    }
    for m in bin_mean.iter_mut() {
        *m /= n_frames.max(1) as f64;
    }

    // Ensemble of per-phase statistics.
    let mut ens: Vec<f64> = Vec::with_capacity(n_bits);
    let mut best = Vec::new();
    let mut best_stat = f64::MIN;
    for ph0 in 0..n_bits {
        let corr = correlate(&spec, n_frames, nb, &order, &bin_mean, key, ph0, n_bits);
        let stat = corr.iter().map(|c| c.abs()).sum::<f64>() / n_bits as f64;
        ens.push(stat);
        if stat > best_stat {
            best_stat = stat;
            best = corr;
        }
    }
    let mean = ens.iter().sum::<f64>() / ens.len() as f64;
    let sd = (ens.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / ens.len() as f64).sqrt();
    let z = if sd > 0.0 { (best_stat - mean) / sd } else { 0.0 };
    let bits: Vec<u8> = best.iter().map(|&c| if c >= 0.0 { 1 } else { 0 }).collect();
    (best_stat, z, bits)
}

fn main() {
    let sr = 16_000;
    let key = b"bakeoff-secret-key-0123456789";
    let mut p = dsp::ChaCha::new(b"payload", 0);
    let payload: Vec<u8> = (0..64).map(|_| (p.u32() & 1) as u8).collect();
    let tracks = [
        "assets/galway.flac",
        "assets/freak.flac",
        "assets/acidjazz.flac",
        "assets/digya.flac",
        "assets/grieg.ogg",
        "assets/mozart.ogg",
    ];
    let mut c = MagDsss::new();

    let mut sig = Vec::new();
    let mut nul = Vec::new();
    let mut sig_z = Vec::new();
    let mut nul_z = Vec::new();
    let mut bers = Vec::new();

    println!("track                 stat(sig) z(sig)  stat(null) z(null)  BER");
    for t in tracks {
        let x = dec(Path::new(t), sr);
        let (wm, _) = c.embed(&x, &payload, key);
        let (s1, z1, bits) = detect_stats(&wm, key, &c);
        let (s0, z0, _) = detect_stats(&x, key, &c);
        let b = ber(&bits, &payload);
        bers.push(b);
        sig.push(s1);
        nul.push(s0);
        sig_z.push(z1);
        nul_z.push(z0);
        println!("{t:20} {s1:8.2} {z1:6.2}  {s0:10.2} {z0:7.2}  {b:.3}");
    }

    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &Vec<f64>| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    println!("\n-- raw max-statistic --");
    println!(
        "signal mean={:.2} sd={:.2} | null mean={:.2} sd={:.2} | separation={:.2} sigma",
        mean(&sig),
        sd(&sig),
        mean(&nul),
        sd(&nul),
        (mean(&sig) - mean(&nul)) / sd(&nul)
    );
    println!("-- z-score statistic --");
    println!(
        "signal mean={:.2} sd={:.2} | null mean={:.2} sd={:.2} | separation={:.2} sigma",
        mean(&sig_z),
        sd(&sig_z),
        mean(&nul_z),
        sd(&nul_z),
        (mean(&sig_z) - mean(&nul_z)) / sd(&nul_z)
    );
    println!(
        "\nclean BER mean={:.4} max={:.4}",
        mean(&bers),
        bers.iter().cloned().fold(0.0, f64::max)
    );
}
