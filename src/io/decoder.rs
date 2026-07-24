//! One-shot audio file decoding via Symphonia.

use std::fs::File;
use std::path::Path;

use symphonia::core::audio::{Audio, AudioBuffer, GenericAudioBufferRef};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, FormatReader, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::dsp::resample::SincResampler;
use crate::error::IoError;
use crate::{AfpError, Result};

/// Resource limits for untrusted-upload decoding.
///
/// Use **both** caps in production: `max_bytes` rejects oversized files
/// before opening the stream, while `max_samples` bounds decoded mono
/// PCM growth (critical for compressed formats where on-disk size does
/// not bound decoded size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeLimits {
    /// Reject when on-disk file size exceeds this many bytes.
    /// `0` disables the byte check.
    pub max_bytes: u64,
    /// Reject when decoded mono PCM would exceed this many samples.
    /// `None` disables the sample check.
    pub max_samples: Option<usize>,
}

impl DecodeLimits {
    /// Byte-only cap (`max_samples = None`). Prefer [`Self::both`] for
    /// compressed uploads.
    #[must_use]
    pub const fn bytes(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            max_samples: None,
        }
    }

    /// Sample-only cap (`max_bytes = 0`).
    #[must_use]
    pub const fn samples(max_samples: usize) -> Self {
        Self {
            max_bytes: 0,
            max_samples: Some(max_samples),
        }
    }

    /// Both on-disk and decoded-PCM caps.
    #[must_use]
    pub const fn both(max_bytes: u64, max_samples: usize) -> Self {
        Self {
            max_bytes,
            max_samples: Some(max_samples),
        }
    }
}

/// Decode an audio file into a mono `f32` buffer at the file's native
/// sample rate.
///
/// Multi-channel files are downmixed to mono by averaging channels per
/// frame. The returned tuple is `(samples, sample_rate_hz)`.
///
/// # Supported formats
///
/// MP3, FLAC, WAV, OGG-Vorbis, AAC-in-MP4, raw PCM — whatever Symphonia's
/// default registries provide with the features enabled in
/// `audiofp`'s `Cargo.toml`. The decoder probes magic bytes too, so
/// extension-less files still work as long as they're a recognised format.
///
/// # Errors
///
/// - [`AfpError::Io`] if the file is missing, the format isn't recognised,
///   or a stream-fatal decode error happens. Recoverable per-packet failures
///   inside Symphonia are silently skipped so a single corrupt block
///   doesn't kill the whole-file decode.
///
/// # Example
///
/// # Security
///
/// This function applies **no resource limits**. A compressed
/// decompression bomb (tiny on-disk, expands to gigabytes of PCM) will
/// succeed and may OOM the process. For untrusted uploads use
/// [`decode_to_mono_limited`] with [`DecodeLimits::both`] so both
/// on-disk size and decoded PCM are bounded.
///
/// ```no_run
/// use audiofp::io::decode_to_mono;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let (samples, sr) = decode_to_mono("song.flac")?;
/// println!("{} samples at {sr} Hz", samples.len());
/// # Ok(()) }
/// ```
pub fn decode_to_mono<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    decode_to_mono_limited(path, DecodeLimits::default())
}

/// Decode with explicit on-disk and/or decoded-PCM caps.
///
/// # Errors
///
/// - [`AfpError::InputTooLarge`] if the file exceeds `max_bytes` or
///   decoded mono samples would exceed `max_samples`.
/// - [`AfpError::Io`] for missing/unrecognised/corrupt streams (same as
///   [`decode_to_mono`]).
pub fn decode_to_mono_limited<P: AsRef<Path>>(
    path: P,
    limits: DecodeLimits,
) -> Result<(Vec<f32>, u32)> {
    let path = path.as_ref();
    // Pre-check: don't even open files that are clearly too large.
    // Note: this is best-effort against TOCTOU (file can grow after the
    // stat); `max_samples` is the hard bound on decoded PCM.
    if limits.max_bytes > 0 {
        let meta = std::fs::metadata(path).map_err(|e| AfpError::io_with_path(path, e))?;
        let len = meta.len();
        if len > limits.max_bytes {
            return Err(AfpError::InputTooLarge {
                limit: usize::try_from(limits.max_bytes).unwrap_or(usize::MAX),
                provided: usize::try_from(len).unwrap_or(usize::MAX),
            });
        }
    }
    let file = File::open(path).map_err(|e| AfpError::io_with_path(path, e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    decode_inner(mss, &hint, limits.max_samples)
}

/// Decode an audio file and resample it to `target_sr` Hz mono `f32`.
///
/// Pass-through (no resample) when the file already matches `target_sr`.
/// Otherwise resamples via [`SincResampler`] at default quality
/// (32-tap Kaiser, β = 8.6). Equivalent to calling [`decode_to_mono`]
/// then [`SincResampler::process`] yourself, but in one step.
///
/// # Errors
///
/// Surfaces every error [`decode_to_mono`] can return; resampling itself
/// cannot fail with the built-in [`SincResampler`].
///
/// # Example
///
/// ```no_run
/// use audiofp::io::decode_to_mono_at;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Get audio ready for Wang in one line:
/// let samples = decode_to_mono_at("song.mp3", 8_000)?;
/// # Ok(()) }
/// ```
pub fn decode_to_mono_at<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<Vec<f32>> {
    decode_to_mono_at_limited(path, target_sr, DecodeLimits::default())
}

/// Same as [`decode_to_mono_at`] with full [`DecodeLimits`].
pub fn decode_to_mono_at_limited<P: AsRef<Path>>(
    path: P,
    target_sr: u32,
    limits: DecodeLimits,
) -> Result<Vec<f32>> {
    if target_sr == 0 {
        return Err(AfpError::Config("target sample rate must be > 0".into()));
    }
    let (samples, sr) = decode_to_mono_limited(path, limits)?;
    if sr == target_sr {
        Ok(samples)
    } else {
        let r = SincResampler::new(sr, target_sr);
        Ok(r.process(&samples))
    }
}

fn decode_inner(
    mss: MediaSourceStream,
    hint: &Hint,
    max_samples: Option<usize>,
) -> Result<(Vec<f32>, u32)> {
    let mut format: Box<dyn FormatReader> = symphonia::default::get_probe()
        .probe(
            hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| {
            AfpError::Io(IoError::without_path(std::io::Error::other(format!(
                "probe: {e}"
            ))))
        })?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| {
            AfpError::Io(IoError::without_path(std::io::Error::other(
                "no audio track",
            )))
        })?
        .clone();
    let track_id = track.id;

    let audio_params = match track.codec_params.as_ref() {
        Some(symphonia::core::codecs::CodecParameters::Audio(params)) => params,
        _ => {
            return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                "no audio codec params",
            ))));
        }
    };

    let sample_rate = audio_params.sample_rate.ok_or_else(|| {
        AfpError::Io(IoError::without_path(std::io::Error::other(
            "missing sample rate",
        )))
    })?;

    let codecs = symphonia::default::get_codecs();
    let decoder_factory = codecs
        .get_audio_decoder(audio_params.codec)
        .ok_or_else(|| {
            AfpError::Io(IoError::without_path(std::io::Error::other(
                "unsupported codec",
            )))
        })?;
    let mut decoder = (decoder_factory.factory)(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| {
            AfpError::Io(IoError::without_path(std::io::Error::other(format!(
                "make decoder: {e}"
            ))))
        })?;

    let mut samples: Vec<f32> = Vec::new();
    let mut convert_buf: Option<AudioBuffer<f32>> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => {
                return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                    format!("next_packet: {e}"),
                ))));
            }
        };
        if packet.track_id != track_id {
            continue;
        }

        let decoded: GenericAudioBufferRef = match decoder.decode(&packet) {
            Ok(d) => d,
            // Recoverable per-packet failures: skip and keep going.
            Err(SymphoniaError::IoError(_)) | Err(SymphoniaError::DecodeError(_)) => {
                continue;
            }
            Err(e) => {
                return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                    format!("decode: {e}"),
                ))));
            }
        };

        // Lazily allocate the f32 conversion buffer once the first packet
        // tells us the channel layout / capacity. Reallocate if a later
        // packet decodes to more frames than the current buffer can hold
        // (the first packet's capacity is not guaranteed to bound the rest).
        let needed_cap = decoded.frames().max(decoded.capacity());
        let needs_buf = match &convert_buf {
            None => true,
            Some(buf) => needed_cap > buf.capacity(),
        };
        if needs_buf {
            let spec = decoded.spec().clone();
            convert_buf = Some(AudioBuffer::<f32>::new(spec, needed_cap));
        }
        let buf = convert_buf.as_mut().unwrap();

        // In symphonia 0.6, copy_to requires the destination to have the
        // same frame count as the source. Set it before copying.
        buf.resize_uninit(decoded.frames());
        decoded.copy_to::<f32, _>(buf);

        let n_frames = buf.frames();
        let n_chans = buf.spec().channels().count();

        // Defensive: skip packets that report 0 channels (malformed /
        // corrupt). Avoids division by zero and `.plane(0).unwrap()` panic.
        if n_chans == 0 {
            continue;
        }

        // Bound decoded PCM growth before allocating more samples.
        if let Some(limit) = max_samples {
            let next = samples.len().saturating_add(n_frames);
            if next > limit {
                return Err(AfpError::InputTooLarge {
                    limit,
                    provided: next,
                });
            }
        }

        if n_chans == 1 {
            samples.extend_from_slice(&buf.plane(0).unwrap()[..n_frames]);
        } else {
            samples.reserve(n_frames);
            for i in 0..n_frames {
                let mut sum = 0.0_f32;
                for c in 0..n_chans {
                    sum += buf.plane(c).unwrap()[i];
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
            "audiofp-decoder-test-{}-{}-{}-{}-{}.wav",
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
            "audiofp-decoder-float-{}-{}-{}-{}.wav",
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

    #[test]
    fn capped_rejects_oversized_file_with_input_too_large() {
        let path = write_test_wav(1, 8_000, 8_000);
        let meta_len = std::fs::metadata(&path).unwrap().len();
        assert!(meta_len > 100, "expected a non-trivial wav, got {meta_len}");
        let err = decode_to_mono_limited(&path, DecodeLimits::bytes(100)).unwrap_err();
        std::fs::remove_file(&path).ok();
        match err {
            AfpError::InputTooLarge { limit, provided } => {
                assert_eq!(limit, 100);
                assert_eq!(provided, usize::try_from(meta_len).unwrap());
            }
            other => panic!("expected InputTooLarge, got {other:?}"),
        }
    }

    #[test]
    fn capped_accepts_file_under_byte_limit() {
        let path = write_test_wav(1, 8_000, 1_000);
        let meta_len = std::fs::metadata(&path).unwrap().len();
        let (samples, sr) = decode_to_mono_limited(&path, DecodeLimits::bytes(meta_len)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(sr, 8_000);
        assert_eq!(samples.len(), 1_000);
    }

    #[test]
    fn limited_rejects_when_decoded_samples_exceed_cap() {
        let path = write_test_wav(1, 8_000, 4_000);
        let err = decode_to_mono_limited(&path, DecodeLimits::samples(100)).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            matches!(err, AfpError::InputTooLarge { limit: 100, .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn limited_both_caps_small_file_ok() {
        let path = write_test_wav(1, 8_000, 500);
        let meta_len = std::fs::metadata(&path).unwrap().len();
        let (samples, sr) =
            decode_to_mono_limited(&path, DecodeLimits::both(meta_len, 500)).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(sr, 8_000);
        assert_eq!(samples.len(), 500);
    }
}
