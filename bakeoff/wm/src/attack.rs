//! Attack matrix + metrics for the watermark bakeoff.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One attack definition: a name and the ffmpeg filter/args to apply.
/// `codec_args` are the *output* encoding args; the result is always decoded
/// back to 16-bit mono PCM WAV by the runner.
pub struct Attack {
    pub name: &'static str,
    pub ffmpeg_args: Vec<String>,
}

/// Sample rate every decode is forced to. Kept in one place so the codec
/// round-trip cannot silently mix rates (see the Opus note in `run_codec`).
pub const RATE: u32 = 16_000;

/// Paths / workspace for a single run.
pub struct Work {
    pub dir: PathBuf,
    pub sample_rate: u32,
}

impl Work {
    pub fn new(sample_rate: u32) -> Self {
        let dir = std::env::temp_dir().join(format!("wm-bakeoff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self { dir, sample_rate }
    }
}

/// Run ffmpeg: decode `src` to 16-bit mono PCM WAV, apply `args` (output
/// codec/quality), then decode that *back* to 16-bit mono PCM WAV.
///
/// `args` should describe the lossy encode; the function handles the
/// round-trip decode so callers only specify the codec.
pub fn run_codec(src: &Path, out: &Path, args: &[String]) {
    // Encode step: src (PCM WAV in) -> temp lossy file with a codec-appropriate
    // extension so ffmpeg picks the right muxer.
    let ext = codec_ext(args).unwrap_or("bin");
    let tmp = out.with_extension(ext);
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(args)
        .arg(&tmp)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "ffmpeg encode failed for {args:?}");
    // Decode step: temp lossy -> PCM WAV at the harness rate. The rate MUST
    // be pinned: Opus always decodes at 48 kHz, and reading a 48 kHz buffer
    // as if it were the harness rate shifts every bin.
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&tmp)
        .args(["-ac", "1", "-ar", &RATE.to_string(), "-c:a", "pcm_s16le"])
        .arg(out)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "ffmpeg decode failed for {args:?}");
    let _ = std::fs::remove_file(&tmp);
}

/// Pick a container extension from the ffmpeg output args.
fn codec_ext(args: &[String]) -> Option<&'static str> {
    let joined = args.join(" ");
    if joined.contains("libmp3lame") { Some("mp3") }
    else if joined.contains("libvorbis") { Some("ogg") }
    else if joined.contains("libopus") { Some("opus") }
    else if joined.contains("aac") || joined.contains("m4a") { Some("m4a") }
    else { None }
}

/// Run ffmpeg with an arbitrary audio *filter* chain (no re-encode), then
/// decode to PCM WAV. `filter` is e.g. `volume=0.5` or `highpass=f=200`.
pub fn run_filter(src: &Path, out: &Path, filter: &str, sample_rate: u32) {
    let tmp = out.with_extension("flt.wav");
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args(["-af", filter, "-ac", "1", "-ar", &sample_rate.to_string(), "-c:a", "pcm_s16le"])
        .arg(&tmp)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "ffmpeg filter failed: {filter}");
    std::fs::copy(&tmp, out).unwrap();
    let _ = std::fs::remove_file(&tmp);
}

/// Resample using audiofp-quality sinc (via ffmpeg's soxr) to `to_sr` then
/// back to the original rate — models the common "rate conversion" attack.
pub fn run_resample(src: &Path, out: &Path, from_sr: u32, to_sr: u32) {
    let tmp = out.with_extension("rs.wav");
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(src)
        .args([
            "-af",
            &format!("aresample={to_sr}"),
            "-ac",
            "1",
            "-ar",
            &to_sr.to_string(),
            "-c:a",
            "pcm_s16le",
        ])
        .arg(&tmp)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "ffmpeg resample failed");
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i"])
        .arg(&tmp)
        .args(["-ac", "1", "-ar", &from_sr.to_string(), "-c:a", "pcm_s16le"])
        .arg(out)
        .status()
        .expect("ffmpeg spawn failed");
    assert!(status.success(), "ffmpeg resample-back failed");
    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Signal-to-noise ratio in dB between `reference` and `test`
/// (i.e. interpreting `reference - test` as noise).
pub fn snr_db(reference: &[f32], test: &[f32]) -> f64 {
    let n = reference.len().min(test.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for i in 0..n {
        let s = reference[i] as f64;
        let e = (reference[i] - test[i]) as f64;
        sig += s * s;
        noise += e * e;
    }
    if noise <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (sig / noise).log10()
}

/// Log-spectral distance (dB) between two signals, per Hanning-windowed
/// frame of `n_fft`, averaged over frames.
pub fn log_spectral_distance(reference: &[f32], test: &[f32], n_fft: usize, hop: usize) -> f64 {
    let n = reference.len().min(test.len());
    if n < n_fft {
        return f64::NAN;
    }
    let win: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos())
        .collect();
    let mut planner = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let nbins = n_fft / 2 + 1;
    let mut buf_a = vec![rustfft::num_complex::Complex::new(0.0f64, 0.0); n_fft];
    let mut buf_b = vec![rustfft::num_complex::Complex::new(0.0f64, 0.0); n_fft];
    let mut acc = 0.0f64;
    let mut frames = 0usize;
    let floor = 1e-10f64;
    let mut pos = 0usize;
    while pos + n_fft <= n {
        for i in 0..n_fft {
            buf_a[i] = rustfft::num_complex::Complex::new(reference[pos + i] as f64 * win[i], 0.0);
            buf_b[i] = rustfft::num_complex::Complex::new(test[pos + i] as f64 * win[i], 0.0);
        }
        fft.process(&mut buf_a);
        fft.process(&mut buf_b);
        let mut frame_sq = 0.0f64;
        for k in 0..nbins {
            let pa = (buf_a[k].norm_sqr()).max(floor);
            let pb = (buf_b[k].norm_sqr()).max(floor);
            let d = 10.0 * (pa / pb).log10();
            frame_sq += d * d;
        }
        acc += (frame_sq / nbins as f64).sqrt();
        frames += 1;
        pos += hop;
    }
    if frames == 0 {
        f64::NAN
    } else {
        acc / frames as f64
    }
}

/// Bit-error rate between two bit slices (min-length).
pub fn ber(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let errs = (0..n).filter(|&i| a[i] != b[i]).count();
    errs as f64 / n as f64
}
