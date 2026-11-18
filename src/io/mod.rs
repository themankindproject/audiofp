//! Audio file I/O helpers.
//!
//! Available only when at least one per-codec feature is enabled
//! (`std-wav`, `std-mp3`, `std-flac`, `std-ogg`, `std-aac`, `std-mp4`,
//! `std-aiff`, `std-mkv`, `std-adpcm`, `std-alac`), or `all-codecs` for
//! every codec at once. Each sub-feature pulls the matching Symphonia
//! decoder; the bare `std` feature alone enables no codecs and produces a
//! compile error if you try to use this module. Wraps Symphonia's probe →
//! format-reader → decoder pipeline behind simple,
//! allocation-conservative helpers:
//!
//! - [`decode_to_mono`] — decode a file at its native sample rate.
//! - [`decode_to_mono_at`] — decode and resample to a target rate in one
//!   step, using `audiofp`'s built-in [`SincResampler`].
//! - [`decode_to_mono_limited`] / [`decode_to_mono_at_limited`] — same as
//!   above with on-disk and/or decoded-PCM caps for untrusted uploads.
//!
//! Every helper returns mono `f32` PCM (multi-channel files are downmixed
//! by averaging channels per frame), ready to feed into any
//! [`Fingerprinter`].
//!
//! **Range:** integer PCM is scaled to `[-1.0, 1.0]`, but *float* WAVs are
//! passed through **unnormalised** — a file may contain samples outside
//! `[-1.0, 1.0]`, plus `NaN`/`±Inf` in a corrupt or crafted one. The
//! offline `extract` paths reject non-finite input with
//! [`AfpError::NonFiniteSample`](crate::AfpError::NonFiniteSample);
//! validate or clamp yourself if you need a hard range guarantee.
//!
//! Files with more than 64 channels are rejected (the per-packet downmix
//! buffer is `frames × channels` samples): with
//! [`AfpError::UnsupportedChannels`](crate::AfpError::UnsupportedChannels)
//! when the decoder finds the count, and with an
//! [`AfpError::Io`](crate::AfpError::Io) of kind `InvalidData` when the
//! WAV / AIFF header check catches it first, before the file is probed.
//!
//! **Pipes.** A FIFO or `/dev/stdin` decodes like a file when its format
//! can be read without seeking (WAV, FLAC, MP3, Ogg), as long as no
//! `max_bytes` cap is set — a pipe has no length to check. Opening a FIFO
//! that has no writer blocks inside the operating system's `open`, before
//! any limit or timeout applies.
//!
//! Supported formats are whatever Symphonia provides with the codec
//! features enabled in `audiofp`'s `Cargo.toml`: MP3, FLAC, WAV,
//! OGG-Vorbis, AAC-in-MP4, raw PCM, plus AIFF / Matroska / ADPCM / ALAC
//! behind their respective extra features.
//!
//! [`Fingerprinter`]: crate::Fingerprinter
//! [`SincResampler`]: crate::dsp::resample::SincResampler

mod riff_preflight;

/// Maximum channel count `audiofp` will decode.
///
/// The decoder downmixes to mono, so more than a handful of channels is
/// already pathological for fingerprinting — and the per-packet
/// conversion buffer is `frames × channels` `f32`s, so an unbounded
/// channel count is a memory amplification vector (Matroska
/// `Channels::Discrete(u16)` and AIFF can both carry counts up to
/// 65535). 64 comfortably covers 7.1, ambisonics, and stem files.
const MAX_DECODE_CHANNELS: usize = 64;

pub mod decoder;

pub use decoder::{
    DecodeLimits, DecodeReport, DecodeStats, decode_to_mono, decode_to_mono_at,
    decode_to_mono_at_limited, decode_to_mono_limited, decode_to_mono_report,
};
