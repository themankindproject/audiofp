//! `audiofp` — audio fingerprinting SDK for Rust.
//!
//! `audiofp` extracts compact, codec-tolerant perceptual hashes from audio
//! so you can identify the same recording across re-encoding, modest
//! noise, and (for some algorithms) tempo or pitch changes — the
//! fundamental primitive behind systems like Shazam or AcoustID.
//!
//! The crate is **`no_std + alloc`** in API shape when the `std`
//! feature is disabled, but the current FFT dependency chain still
//! keeps the no_std path host-only today. The file decoder ([`io`]) and
//! watermark detector ([`watermark`]) live behind feature flags and
//! require `std`.
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
//! # Panics in streaming APIs
//!
//! All `StreamingFingerprinter::push` implementations are infallible
//! **except** [`neural::StreamingNeuralEmbedder::push`], which panics
//! if the underlying ONNX model reports an inference error. The
//! non-panicking counterpart [`neural::StreamingNeuralEmbedder::try_push`]
//! returns `Result` for any code that needs to surface those failures
//! (audio callbacks, `tokio::spawn` workers, etc.). Classical
//! streaming fingerprinters (Wang / Panako / Haitsma) never panic on
//! valid input.
//!
//! [`neural::StreamingNeuralEmbedder::push`]: crate::neural::StreamingNeuralEmbedder::push
//! [`neural::StreamingNeuralEmbedder::try_push`]: crate::neural::StreamingNeuralEmbedder::try_push
//!
//! # Example
//!
//! Identify a song by counting Wang hash collisions between two files:
//!
//! ```
//! extern crate alloc;
//! use audiofp::classical::Wang;
//! use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
//!
//! let samples: alloc::vec::Vec<f32> = alloc::vec![0.0_f32; 8_000 * 4];
//! let mut wang = Wang::default();
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
//! let fp = wang.extract(buf).unwrap();
//!
//! let unique: alloc::collections::BTreeSet<u32> =
//!     fp.hashes.into_iter().map(|h| h.hash).collect();
//! println!("{} unique landmark hashes", unique.len());
//! ```
//!
//! # Cargo features
//!
//! | Feature      | Default | Description                                                       |
//! | ------------ | :-----: | ----------------------------------------------------------------- |
//! | `std`        |   ✅    | Pulls in [`symphonia`](https://docs.rs/symphonia) → [`io`].       |
//! | `watermark`  |         | Pulls in [`tract-onnx`](https://docs.rs/tract-onnx) → [`watermark`]. |
//! | `neural`     |         | Generic ONNX log-mel embedder ([`neural`]); pulls in [`tract-onnx`]. |
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
#[cfg(feature = "neural")]
pub mod neural;
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
/// assert_eq!(audiofp::VERSION, env!("CARGO_PKG_VERSION"));
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
