//! One-shot audio file decoding via Symphonia.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{AudioBuffer, AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::dsp::resample::SincResampler;
use crate::{AfpError, Result};

/// Decode an audio file into a mono `f32` buffer at the file's native
/// sample rate.
///
/// Multi-channel files are downmixed to mono by averaging channels per
/// frame. The returned tuple is `(samples, sample_rate_hz)`.
///
/// Supported formats: MP3, FLAC, WAV, OGG-Vorbis, AAC-in-MP4, raw PCM
/// (whatever Symphonia's default registries provide with the features
/// enabled in `Cargo.toml`).
pub fn decode_to_mono<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|e| AfpError::Io(format!("open {}: {e}", path.display())))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    decode_inner(mss, &hint)
}

/// Decode an audio file and resample it to `target_sr` Hz mono `f32`.
///
/// Pass-through (no resample) when the file already matches `target_sr`.
/// Otherwise resamples via [`SincResampler`] at default quality.
pub fn decode_to_mono_at<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<Vec<f32>> {
    let (samples, sr) = decode_to_mono(path)?;
    if sr == target_sr {
        Ok(samples)
    } else {
        let r = SincResampler::new(sr, target_sr);
        Ok(r.process(&samples))
    }
}

fn decode_inner(mss: MediaSourceStream, hint: &Hint) -> Result<(Vec<f32>, u32)> {
    let probed = symphonia::default::get_probe()
        .format(hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| AfpError::Io(format!("probe: {e}")))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AfpError::Io("no audio track".into()))?
        .clone();
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| AfpError::Io("missing sample rate".into()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| AfpError::Io(format!("make decoder: {e}")))?;

    let mut samples: Vec<f32> = Vec::new();
    let mut convert_buf: Option<AudioBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => return Err(AfpError::Io(format!("next_packet: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }

        let decoded: AudioBufferRef = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable per-packet failures: skip and keep going.
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => {
                continue;
            }
            Err(e) => return Err(AfpError::Io(format!("decode: {e}"))),
        };

        // Lazily allocate the f32 conversion buffer once the first packet
        // tells us the channel layout / capacity.
        if convert_buf.is_none() {
            let spec = *decoded.spec();
            let cap = decoded.capacity() as u64;
            convert_buf = Some(AudioBuffer::<f32>::new(cap, spec));
        }
        let buf = convert_buf.as_mut().unwrap();

        decoded.convert(buf);

        let n_frames = buf.frames();
        let n_chans = buf.spec().channels.count();

        if n_chans == 1 {
            samples.extend_from_slice(&buf.chan(0)[..n_frames]);
        } else {
            samples.reserve(n_frames);
            for i in 0..n_frames {
                let mut sum = 0.0_f32;
                for c in 0..n_chans {
                    sum += buf.chan(c)[i];
                }
                samples.push(sum / n_chans as f32);
            }
        }
    }

    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;
    use std::io::Write;

    fn write_test_wav(channels: u16, sr: u32, len: usize) -> std::path::PathBuf {
        // Counter ensures each test gets a unique path so parallel runs
        // don't clobber each other.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "afp-decoder-test-{}-{}-{}-{}-{}.wav",
            std::process::id(),
            channels,
            sr,
            len,
            n,
        ));
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        let amp = (i16::MAX as f32) * 0.5;
        for i in 0..len {
            // 440 Hz tone on every channel (mono on every channel for
            // multichannel files = identical channels, downmix is identity).
            let s = libm::sinf(2.0 * PI * 440.0 * i as f32 / sr as f32) * amp;
            for _c in 0..channels {
                writer.write_sample(s as i16).unwrap();
            }
        }
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn open_missing_file_returns_io_error() {
        let res = decode_to_mono("/nonexistent/path/that/does/not/exist.wav");
        match res {
            Err(AfpError::Io(_)) => {}
            other => panic!("expected Io error, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_mono_wav() {
        let path = write_test_wav(1, 8_000, 8_000);
        let result = decode_to_mono(&path);
        std::fs::remove_file(&path).ok();
        let (samples, sr) = result.unwrap();
        assert_eq!(sr, 8_000);
        assert_eq!(samples.len(), 8_000);

        // Spot-check a sample mid-buffer.
        let expected = libm::sinf(2.0 * PI * 440.0 * 100.0 / 8_000.0) * 0.5;
        // 16-bit truncation introduces ~3e-5 error; allow a generous bound.
        assert!(
            (samples[100] - expected).abs() < 0.01,
            "sample[100] = {}, expected ≈ {expected}",
            samples[100]
        );
    }

    #[test]
    fn stereo_wav_downmixes_to_mono() {
        // Both channels are identical so downmix should be the same signal.
        let path = write_test_wav(2, 16_000, 16_000);
        let result = decode_to_mono(&path);
        std::fs::remove_file(&path).ok();
        let (samples, sr) = result.unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 16_000);

        let expected = libm::sinf(2.0 * PI * 440.0 * 200.0 / 16_000.0) * 0.5;
        assert!((samples[200] - expected).abs() < 0.01);
    }

    #[test]
    fn decode_to_mono_at_resamples() {
        let path = write_test_wav(1, 16_000, 16_000); // 1 sec @ 16 kHz
        let result = decode_to_mono_at(&path, 8_000);
        std::fs::remove_file(&path).ok();
        let samples = result.unwrap();
        // 16k → 8k means roughly half as many samples.
        assert!(
            (samples.len() as i64 - 8_000).abs() < 16,
            "resampled len = {}",
            samples.len()
        );
    }

    #[test]
    fn decode_to_mono_at_passthrough_when_rates_match() {
        let path = write_test_wav(1, 8_000, 4_000);
        let result = decode_to_mono_at(&path, 8_000);
        std::fs::remove_file(&path).ok();
        let samples = result.unwrap();
        assert_eq!(samples.len(), 4_000);
    }

    #[test]
    fn unknown_extension_still_decodes() {
        // Symphonia probes magic bytes too, so an extensionless file still
        // works as long as it's a recognised format.
        let path = write_test_wav(1, 8_000, 4_000);
        let renamed = path.with_extension("");
        std::fs::rename(&path, &renamed).unwrap();

        let result = decode_to_mono(&renamed);
        std::fs::remove_file(&renamed).ok();

        // Use of ? syntax via match: succeed or report.
        let (samples, sr) = match result {
            Ok(v) => v,
            Err(e) => panic!("decode without extension failed: {e}"),
        };
        assert_eq!(sr, 8_000);
        assert_eq!(samples.len(), 4_000);
    }

    /// Ensure the public APIs don't hold onto the file handle past
    /// successful decode (otherwise removing the file would fail on
    /// Windows; on Unix it would leak a descriptor).
    #[test]
    fn temp_file_can_be_deleted_after_decode() {
        let path = write_test_wav(1, 8_000, 1_000);
        decode_to_mono(&path).unwrap();
        // Should not error out.
        std::fs::remove_file(&path).unwrap();
    }

    /// Dummy `Write` ensures the unused-import pruner doesn't strip
    /// `std::io::Write` if a future test needs in-memory writers.
    #[allow(dead_code)]
    fn _writer_witness<W: Write>(_w: W) {}

    fn write_test_wav_float(channels: u16, sr: u32, len: usize) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "afp-decoder-float-{}-{}-{}-{}.wav",
            std::process::id(),
            channels,
            sr,
            n,
        ));
        let spec = hound::WavSpec {
            channels,
            sample_rate: sr,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..len {
            let s = libm::sinf(2.0 * PI * 440.0 * i as f32 / sr as f32) * 0.5;
            for _c in 0..channels {
                writer.write_sample(s).unwrap();
            }
        }
        writer.finalize().unwrap();
        path
    }

    #[test]
    fn float_wav_decodes_with_higher_precision() {
        let path = write_test_wav_float(1, 16_000, 4_000);
        let result = decode_to_mono(&path);
        std::fs::remove_file(&path).ok();
        let (samples, sr) = result.unwrap();
        assert_eq!(sr, 16_000);
        assert_eq!(samples.len(), 4_000);
        // 32-bit float should give near-exact reconstruction.
        let expected = libm::sinf(2.0 * PI * 440.0 * 100.0 / 16_000.0) * 0.5;
        assert!(
            (samples[100] - expected).abs() < 1e-6,
            "sample[100] = {}, expected {expected}",
            samples[100]
        );
    }

    #[test]
    fn high_sample_rate_preserved() {
        let path = write_test_wav(1, 48_000, 4_800);
        let result = decode_to_mono(&path);
        std::fs::remove_file(&path).ok();
        let (samples, sr) = result.unwrap();
        assert_eq!(sr, 48_000);
        assert_eq!(samples.len(), 4_800);
    }

    #[test]
    fn decode_to_mono_at_handles_upsample() {
        let path = write_test_wav(1, 8_000, 4_000);
        let result = decode_to_mono_at(&path, 16_000);
        std::fs::remove_file(&path).ok();
        let samples = result.unwrap();
        // 8k → 16k should give roughly 2× samples.
        assert!(
            (samples.len() as i64 - 8_000).abs() < 16,
            "upsampled len = {}",
            samples.len()
        );
    }
}
