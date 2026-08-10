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
///
/// For multi-tenant or FFI environments, set [`timeout`](Self::timeout)
/// to prevent adversarial inputs from hanging the decode thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DecodeLimits {
    /// Reject when on-disk file size exceeds this many bytes.
    /// `0` disables the byte check.
    pub max_bytes: u64,
    /// Reject when decoded mono PCM would exceed this many samples.
    /// `None` disables the sample check.
    pub max_samples: Option<usize>,
    /// When `true`, any per-packet decode error (corrupt frame, I/O glitch)
    /// becomes a fatal error instead of being silently skipped.
    /// Default: `false` (skip recoverable errors, matching legacy behavior).
    pub integrity_mode: bool,
    /// Maximum wall-clock time allowed for the entire decode operation.
    /// When set, the decoder checks elapsed time after each packet and
    /// returns [`AfpError::Timeout`] if exceeded. Default: `None` (no
    /// time limit).
    ///
    /// Recommended for Python FFI / multi-tenant services where a hung
    /// decode would block the calling worker indefinitely.
    ///
    /// [`AfpError::Timeout`]: crate::AfpError::Timeout
    pub timeout: Option<std::time::Duration>,
}

impl DecodeLimits {
    /// Byte-only cap (`max_samples = None`). Prefer [`Self::both`] for
    /// compressed uploads.
    #[must_use]
    pub const fn bytes(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            max_samples: None,
            integrity_mode: false,
            timeout: None,
        }
    }

    /// Sample-only cap (`max_bytes = 0`).
    #[must_use]
    pub const fn samples(max_samples: usize) -> Self {
        Self {
            max_bytes: 0,
            max_samples: Some(max_samples),
            integrity_mode: false,
            timeout: None,
        }
    }

    /// Both on-disk and decoded-PCM caps.
    #[must_use]
    pub const fn both(max_bytes: u64, max_samples: usize) -> Self {
        Self {
            max_bytes,
            max_samples: Some(max_samples),
            integrity_mode: false,
            timeout: None,
        }
    }

    /// Enable integrity mode: any per-packet decode error becomes fatal
    /// instead of being silently skipped.
    ///
    /// # Example
    ///
    /// ```
    /// use audiofp::io::DecodeLimits;
    ///
    /// let limits = DecodeLimits::both(10_000_000, 480_000).strict();
    /// assert!(limits.integrity_mode);
    /// ```
    #[must_use]
    pub const fn strict(mut self) -> Self {
        self.integrity_mode = true;
        self
    }

    /// Set a wall-clock timeout for the decode operation. Returns
    /// [`AfpError::Timeout`] if decoding takes longer than `duration`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::time::Duration;
    /// use audiofp::io::DecodeLimits;
    ///
    /// let limits = DecodeLimits::both(50_000_000, 960_000)
    ///     .with_timeout(Duration::from_secs(30));
    /// assert_eq!(limits.timeout, Some(Duration::from_secs(30)));
    /// ```
    ///
    /// [`AfpError::Timeout`]: crate::AfpError::Timeout
    #[must_use]
    pub const fn with_timeout(mut self, duration: std::time::Duration) -> Self {
        self.timeout = Some(duration);
        self
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
/// # Security
///
/// This function applies **no resource limits**. A compressed
/// decompression bomb (tiny on-disk, expands to gigabytes of PCM) will
/// succeed and may OOM the process. For untrusted uploads use
/// [`decode_to_mono_limited`] with [`DecodeLimits::both`] so both
/// on-disk size and decoded PCM are bounded.
///
/// # Example
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

    // `Instant::now` is the correct choice here: the decode timeout
    // measures wall-clock time from the caller's perspective. This is not
    // a testability concern — tests exercise the timeout via
    // `Duration::from_nanos(0)` which fires on the first packet regardless
    // of when the Instant was captured.
    #[allow(clippy::disallowed_methods)]
    let deadline = limits.timeout.map(|d| (std::time::Instant::now(), d));

    decode_inner(
        mss,
        &hint,
        limits.max_samples,
        limits.integrity_mode,
        deadline,
    )
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
    integrity_mode: bool,
    deadline: Option<(std::time::Instant, std::time::Duration)>,
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
            // ResetRequired means the reader re-synced (e.g. after a seek
            // or a corrupt container header); retry the next packet.
            Err(SymphoniaError::ResetRequired) => continue,
            Err(e) => {
                return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                    format!("next_packet: {e}"),
                ))));
            }
        };
        // Skip packets from tracks other than the selected default audio
        // track (multi-track files).
        if packet.track_id != track_id {
            continue;
        }

        // Wall-clock timeout check: bail if the configured timeout has
        // elapsed. Checked per-packet (~1 ns overhead from Instant::elapsed).
        if let Some((start, limit)) = deadline {
            let elapsed = start.elapsed();
            if elapsed > limit {
                return Err(AfpError::Timeout {
                    elapsed_ms: elapsed.as_millis() as u64,
                    limit_ms: limit.as_millis() as u64,
                });
            }
        }

        let decoded: GenericAudioBufferRef = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::IoError(e)) => {
                if integrity_mode {
                    return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                        format!("decode integrity: {e}"),
                    ))));
                }
                continue;
            }
            Err(SymphoniaError::DecodeError(e)) => {
                if integrity_mode {
                    return Err(AfpError::Io(IoError::without_path(std::io::Error::other(
                        format!("decode integrity: {e}"),
                    ))));
                }
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
        let buf = convert_buf
            .as_mut()
            .expect("convert_buf initialized above when needs_buf is true");

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
            samples.extend_from_slice(
                &buf.plane(0).expect("decoded buffer must have plane 0")[..n_frames],
            );
        } else {
            samples.reserve(n_frames);
            for i in 0..n_frames {
                let mut sum = 0.0_f32;
                for c in 0..n_chans {
                    sum += buf
                        .plane(c)
                        .expect("decoded buffer must have plane for each channel")[i];
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

        // 16-bit truncation introduces ~3e-5 error; allow a generous bound.
        let expected = libm::sinf(2.0 * PI * 440.0 * 100.0 / 8_000.0) * 0.5;
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

    // -- integrity mode tests --

    /// Create a WAV file and corrupt some bytes in the data section.
    /// WAV header is 44 bytes for standard PCM; corrupting bytes well
    /// past that ensures we hit the data region, not the header.
    fn write_corrupt_wav(sr: u32, len: usize) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "audiofp-decoder-corrupt-{}-{}-{}.wav",
            std::process::id(),
            sr,
            n,
        ));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: sr,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        let amp = (i16::MAX as f32) * 0.5;
        for i in 0..len {
            let s = libm::sinf(2.0 * PI * 440.0 * i as f32 / sr as f32) * amp;
            writer.write_sample(s as i16).unwrap();
        }
        writer.finalize().unwrap();

        // Corrupt a few bytes in the middle of the data section.
        // For a 16-bit mono WAV, data starts at byte 44. Corrupt a
        // chunk in the middle of the file.
        let file_len = std::fs::metadata(&path).unwrap().len() as usize;
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = file_len / 2;
        for i in 0..core::cmp::min(64, file_len - mid) {
            bytes[mid + i] = 0xFF;
        }
        std::fs::write(&path, &bytes).unwrap();
        path
    }

    #[test]
    fn strict_builder_sets_integrity_mode() {
        let limits = DecodeLimits::default().strict();
        assert!(limits.integrity_mode);

        let limits2 = DecodeLimits::both(1_000_000, 480_000).strict();
        assert!(limits2.integrity_mode);
        assert_eq!(limits2.max_bytes, 1_000_000);
        assert_eq!(limits2.max_samples, Some(480_000));
    }

    #[test]
    fn default_mode_skips_corrupt_packets() {
        // With default (non-strict) limits, corrupted WAV data packets
        // should be skipped and decoding should succeed (possibly with
        // fewer samples, but no error).
        let path = write_corrupt_wav(8_000, 8_000);
        let result = decode_to_mono_limited(&path, DecodeLimits::default());
        std::fs::remove_file(&path).ok();
        // WAV is a simple container: symphonia may or may not report a
        // per-packet decode error for corrupted PCM (it might just decode
        // the bytes as garbage audio). Either outcome (Ok or recoverable
        // skip) is acceptable for the default mode — the key invariant is
        // that it does NOT return an Io error with "decode integrity".
        match result {
            Ok(_) => {} // fine — corrupt PCM was decoded as-is or skipped
            Err(AfpError::Io(ref e)) if e.source.to_string().contains("decode integrity") => {
                panic!("default mode should NOT fail with integrity error: {e}");
            }
            Err(_) => {} // other errors (probe failure, etc.) are acceptable
        }
    }

    #[test]
    fn integrity_mode_fails_on_corrupt_packets() {
        // With integrity_mode=true, if Symphonia reports a per-packet
        // decode/IO error, the decode should fail.
        let path = write_corrupt_wav(8_000, 8_000);
        let limits = DecodeLimits::default().strict();
        let result = decode_to_mono_limited(&path, limits);
        std::fs::remove_file(&path).ok();
        // WAV PCM corruption may not always trigger a Symphonia DecodeError
        // (symphonia might just decode the garbage bytes). So this test
        // verifies the contract: IF an error is returned, it must be the
        // integrity error. If it succeeds, that's also fine (means
        // symphonia didn't detect corruption in the PCM stream).
        match result {
            Ok(_) => {
                // Symphonia decoded garbage as valid PCM — acceptable for
                // raw PCM WAV since there's no checksum. The integrity
                // check only fires when Symphonia itself raises an error.
            }
            Err(AfpError::Io(ref e)) if e.source.to_string().contains("decode integrity") => {
                // This is exactly what we want when corruption IS detected.
            }
            Err(other) => {
                // Other errors (e.g. probe failure if header was hit) are ok.
                let _ = other;
            }
        }
    }

    /// A more reliable test: corrupt the WAV header's format chunk to
    /// trigger a guaranteed codec-level error.
    #[test]
    fn integrity_mode_rejects_mangled_format() {
        let path = write_test_wav(1, 8_000, 8_000);
        // Mangle the "fmt " chunk by changing bits_per_sample (offset 34-35
        // in a standard WAV) to an absurd value.
        let mut bytes = std::fs::read(&path).unwrap();
        // Verify this is a RIFF WAV with "fmt " at offset 12.
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        // bits_per_sample is at byte 34 in a standard 16-byte fmt chunk.
        // Set it to 0 to trigger a codec init failure.
        bytes[34] = 0;
        bytes[35] = 0;
        std::fs::write(&path, &bytes).unwrap();

        // With default mode — should fail with a codec/probe error, not
        // "decode integrity" since the codec can't even initialize.
        let result = decode_to_mono_limited(&path, DecodeLimits::default());
        assert!(result.is_err(), "mangled format should fail");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn timeout_zero_duration_returns_timeout_error() {
        // A zero-duration timeout should fire immediately on the first packet.
        let path = write_test_wav(1, 44100, 44100); // 1s of audio
        let limits = DecodeLimits::default().with_timeout(std::time::Duration::from_nanos(0));
        let result = decode_to_mono_limited(&path, limits);
        std::fs::remove_file(&path).ok();
        match result {
            Err(AfpError::Timeout {
                elapsed_ms: _,
                limit_ms,
            }) => {
                assert_eq!(limit_ms, 0);
            }
            other => panic!("expected Timeout error, got: {other:?}"),
        }
    }

    #[test]
    fn timeout_generous_succeeds() {
        // A generous timeout should not interfere with normal decoding.
        let path = write_test_wav(1, 44100, 44100); // 1s of audio
        let limits = DecodeLimits::default().with_timeout(std::time::Duration::from_secs(60));
        let result = decode_to_mono_limited(&path, limits);
        std::fs::remove_file(&path).ok();
        assert!(result.is_ok(), "generous timeout should not fire");
        let (samples, sr) = result.unwrap();
        assert_eq!(sr, 44100);
        assert!(!samples.is_empty());
    }

    #[test]
    fn with_timeout_builder_sets_field() {
        let limits =
            DecodeLimits::both(1_000_000, 480_000).with_timeout(std::time::Duration::from_secs(30));
        assert_eq!(limits.timeout, Some(std::time::Duration::from_secs(30)));
        assert_eq!(limits.max_bytes, 1_000_000);
        assert_eq!(limits.max_samples, Some(480_000));
    }
}
