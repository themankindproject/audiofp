//! `audiofp` — audio fingerprinting SDK for Rust.
//!
//! `audiofp` extracts compact, codec-tolerant perceptual hashes from audio
//! so you can identify the same recording across re-encoding, modest
//! noise, and (for some algorithms) tempo or pitch changes — the
//! fundamental primitive behind systems like Shazam or AcoustID.
//!
//! The crate compiles **`no_std + alloc`** by default (when the `std`
//! feature is disabled), so the DSP primitives and classical
//! fingerprinters can target embedded systems such as
//! `thumbv7em-none-eabihf`. The file decoder ([`io`]) and watermark
//! detector ([`watermark`]) live behind feature flags and require `std`.
//!
//! # Quick tour
//!
//! - **Errors** — [`AfpError`] (`#[non_exhaustive]`) plus the
//!   [`Result`] alias.
//! - **Value types** — [`SampleRate`] (newtype around `NonZeroU32` with
//!   `HZ_*` constants), [`AudioBuffer`] (borrowed mono PCM view), and
//!   [`TimestampMs`] (ordered millisecond timestamp).
//! - **Traits** — [`Fingerprinter`] for whole-buffer extraction,
//!   [`StreamingFingerprinter`] for incremental extraction. Every
//!   algorithm in the crate implements both.
//! - **Classical fingerprinters** — [`classical::Wang`] (Shazam-style
//!   landmark pairs), [`classical::Panako`] (tempo-invariant triplets),
//!   [`classical::Haitsma`] (Philips robust hash bands), each with a
//!   streaming sibling.
//! - **DSP primitives** — [`dsp`] exposes STFT, mel filterbank, peak
//!   picker, resampler, and tapered windows for users building their
//!   own pipelines on top of `audiofp`.
//!
//! # Example
//!
//! Identify a song by counting Wang hash collisions between two files:
//!
//! ```no_run
//! use audiofp::classical::Wang;
//! use audiofp::io::decode_to_mono_at;
//! use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
//! use std::collections::HashSet;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let samples = decode_to_mono_at("song.flac", 8_000)?;
//! let mut wang = Wang::default();
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
//! let fp = wang.extract(buf)?;
//!
//! let unique: HashSet<u32> = fp.hashes.into_iter().map(|h| h.hash).collect();
//! println!("{} unique landmark hashes", unique.len());
//! # Ok(()) }
//! ```
//!
//! # Cargo features
//!
//! | Feature      | Default | Description                                                       |
//! | ------------ | :-----: | ----------------------------------------------------------------- |
//! | `std`        |   ✅    | Pulls in [`symphonia`](https://docs.rs/symphonia) → [`io`].       |
//! | `watermark`  |         | Pulls in [`tract-onnx`](https://docs.rs/tract-onnx) → [`watermark`]. |
//! | `neural`     |         | Reserved for the upcoming neural fingerprinter.                   |
//! | `mimalloc`   |         | Installs `mimalloc::MiMalloc` as the process-wide allocator.      |
//!
//! See [`USAGE.md`](https://github.com/themankindproject/audiofp/blob/main/USAGE.md)
//! for the complete API guide.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod classical;
pub mod dsp;
#[cfg(feature = "std")]
pub mod io;
#[cfg(feature = "watermark")]
pub mod watermark;

mod error;
mod fp;
mod types;

pub use error::{AfpError, Result};
pub use fp::{Fingerprinter, StreamingFingerprinter};
pub use types::{AudioBuffer, SampleRate, TimestampMs};

/// Crate version string, sourced from `Cargo.toml`.
///
/// Useful when persisting fingerprints alongside the producer version,
/// or when emitting diagnostics that need to identify the SDK build.
///
/// # Example
///
/// ```
/// assert!(!audiofp::VERSION.is_empty());
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
