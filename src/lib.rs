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
//!   `HZ_*` constants) and [`TimestampMs`] (ordered millisecond
//!   timestamp). Extraction takes `&[f32]` samples plus a
//!   [`SampleRate`] directly (the old `AudioBuffer` wrapper was removed
//!   in 0.4.0; see `MIGRATING_0.4.md`).
//! - **Traits** — [`Fingerprinter`] for whole-buffer extraction,
//!   [`StreamingFingerprinter`] for incremental extraction. Every
//!   algorithm in the crate implements both.
//! - **Classical fingerprinters** — [`classical::Wang`] (Shazam-style
//!   landmark pairs), [`classical::Panako`] (tempo-invariant triplets),
//!   [`classical::Haitsma`] (Philips robust hash bands), each with a
//!   streaming sibling.
//! - **Matching** — [`matching`] identifies recordings from fingerprints
//!   in memory (`WangMatcher`, `HaitsmaMatcher`, `PanakoMatcher` with
//!   tempo-invariant 2-D Hough + RANSAC, optional neural cosine), plus
//!   `WangIndex` / `HaitsmaIndex` / `PanakoIndex` 1:N accelerators.
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
//! Match two Wang fingerprints with the offset-histogram voter:
//!
//! ```
//! extern crate alloc;
//! use audiofp::classical::{WangFingerprint, WangHash};
//! use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
//!
//! let fp = WangFingerprint {
//!     hashes: (0..8u32)
//!         .map(|i| WangHash {
//!             hash: i,
//!             // 10 STFT frames apart = 160 ms at 62.5 fps.
//!             t_anchor: audiofp::TimestampMs(i as u64 * 160),
//!         })
//!         .collect(),
//!     frames_per_sec: 62.5,
//! };
//!
//! let matcher = WangMatcher::new(WangMatchConfig::default());
//! let m = matcher.match_one(&fp, &fp);
//! assert!(m.is_match);
//! assert_eq!(m.offset.frames, 0);
//! ```
//!
//! # Cargo features
//!
//! | Feature      | Default | Description                                                       |
//! | ------------ | :-----: | ----------------------------------------------------------------- |
//! | `std`        |   ✅    | Pulls in [`symphonia`](https://docs.rs/symphonia) → [`io`].       |
//! | `watermark`  |         | Pulls in [`tract-onnx`](https://docs.rs/tract-onnx) → [`watermark`]. |
//! | `neural`     |         | Generic ONNX log-mel embedder ([`neural`]); pulls in [`tract-onnx`](https://docs.rs/tract-onnx). |
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
pub mod matching;
#[cfg(feature = "neural")]
pub mod neural;
/// Convenience re-exports of the most commonly used types. See
/// [`prelude`] for details.
pub mod prelude;
pub mod serial;
#[cfg(feature = "watermark")]
pub mod watermark;

mod error;
mod fp;
mod pcm;
mod types;

#[cfg(feature = "std")]
pub use error::IoError;
pub use error::{AfpError, Result};
pub use fp::{Fingerprinter, StreamingFingerprinter};
pub use serial::FingerprintEnvelope;
pub use types::{SampleRate, TimestampMs};

/// Convenience re-exports of the classical algorithm types at the crate
/// root. The canonical location remains [`classical`]; these are aliases
/// so `use audiofp::Wang;` works directly.
pub use classical::{
    Haitsma, HaitsmaConfig, HaitsmaFingerprint, Panako, PanakoConfig, PanakoFingerprint,
    PanakoHash, StreamingHaitsma, StreamingPanako, StreamingWang, Wang, WangConfig,
    WangFingerprint, WangHash,
};

/// Multi-threaded batch fingerprinting (requires the `rayon` feature).
#[cfg(feature = "rayon")]
pub use fp::fingerprint_batch_parallel;

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
