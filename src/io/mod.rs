//! Audio file I/O helpers.
//!
//! Available only when the `std` feature is enabled (the default on
//! desktop targets). Provides one-shot decoding of MP3 / FLAC / WAV /
//! OGG-Vorbis / AAC-in-MP4 / raw PCM into a mono `f32` buffer ready
//! for any [`Fingerprinter`].
//!
//! [`Fingerprinter`]: crate::Fingerprinter

pub mod decoder;

pub use decoder::{decode_to_mono, decode_to_mono_at};
