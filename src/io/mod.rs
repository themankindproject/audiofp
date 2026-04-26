//! Audio file I/O helpers.
//!
//! Available only when the `std` feature is enabled (the default on
//! desktop targets). Wraps Symphonia's probe → format-reader → decoder
//! pipeline behind two simple, allocation-conservative helpers:
//!
//! - [`decode_to_mono`] — decode a file at its native sample rate.
//! - [`decode_to_mono_at`] — decode and resample to a target rate in one
//!   step, using `audiofp`'s built-in [`SincResampler`].
//!
//! Both helpers return mono `f32` PCM in `[-1.0, 1.0]` (multi-channel
//! files are downmixed by averaging channels per frame), ready to feed
//! into any [`Fingerprinter`].
//!
//! Supported formats are whatever Symphonia provides with the features
//! enabled in `audiofp`'s `Cargo.toml`: MP3, FLAC, WAV, OGG-Vorbis,
//! AAC-in-MP4, and raw PCM at the time of writing.
//!
//! [`Fingerprinter`]: crate::Fingerprinter
//! [`SincResampler`]: crate::dsp::resample::SincResampler

pub mod decoder;

pub use decoder::{decode_to_mono, decode_to_mono_at};
