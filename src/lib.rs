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
//!   in 0.4.0; see the migration guide in `CHANGELOG.md`).
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
//! All `StreamingFingerprinter::push` / `flush` implementations are
//! fallible and return `Result` — including
//! [`neural::StreamingNeuralEmbedder::push`], which propagates ONNX
//! inference errors instead of panicking (use
//! [`neural::StreamingNeuralEmbedder::try_push`] / `try_push_with`
//! for the callback-style equivalents). Classical streaming
//! fingerprinters (Wang / Panako / Haitsma) never error on valid
//! input. Constructors named `new` (e.g. [`ShortTimeFFT::new`]) panic
//! on invalid configs; each has a `try_new` counterpart returning
//! `Result`.
//!
//! [`neural::StreamingNeuralEmbedder::push`]: crate::neural::StreamingNeuralEmbedder::push
//! [`neural::StreamingNeuralEmbedder::try_push`]: crate::neural::StreamingNeuralEmbedder::try_push
//! [`ShortTimeFFT::new`]: crate::dsp::stft::ShortTimeFFT::new
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
//!             // 10 STFT frames apart (frame index in t_anchor).
//!             t_anchor: i * 10,
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
//! The default build is `no_std + alloc` with **no codecs**. File decoding
//! (`audiofp::io`) is opt-in per codec; each `std-*` feature pulls the
//! matching [`symphonia`](https://docs.rs/symphonia) decoder:
//!
//! | Feature      | Default | Description                                                       |
//! | ------------ | :-----: | ----------------------------------------------------------------- |
//! | `std`        |         | Symphonia itself (no codecs; combine with a `std-*` feature).     |
//! | `std-wav`    |         | WAV + raw PCM decoding → [`io`].                                  |
//! | `std-mp3`    |         | MP3 decoding → [`io`].                                            |
//! | `std-flac`   |         | FLAC decoding → [`io`].                                           |
//! | `std-ogg`    |         | Ogg-Vorbis decoding → [`io`].                                     |
//! | `std-aac`    |         | AAC decoding → [`io`].                                            |
//! | `std-mp4`    |         | AAC-in-MP4 / ISO-BMFF decoding → [`io`].                          |
//! | `std-aiff` / `std-mkv` / `std-adpcm` / `std-alac` | | Extended codecs → [`io`]. |
//! | `all-codecs` |         | Every format/codec above at once → [`io`] (the pre-0.4.0 `std`). |
//! | `rayon`      |         | Parallel batch fingerprinting via [`fingerprint_batch_parallel`] (implies `std`). |
//! | `watermark`  |         | Pulls in [`tract-onnx`](https://docs.rs/tract-onnx) → [`watermark`] (implies `std`). |
//! | `neural`     |         | Generic ONNX log-mel embedder ([`neural`]); pulls in [`tract-onnx`](https://docs.rs/tract-onnx) (implies `std`). |
//! | `mimalloc`   |         | Installs `mimalloc::MiMalloc` as the process-wide allocator (implies `std`). |
//!
//! [`fingerprint_batch_parallel`]: crate::fingerprint_batch_parallel
//!
//! See [`USAGE.md`](https://github.com/themankindproject/audiofp/blob/main/USAGE.md)
//! for the complete API guide.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
// doc_auto_cfg is stable since Rust 1.92 (merged into doc_cfg) — no
// feature gate needed. The `docsrs` cfg is set by docs.rs and by our
// `package.metadata.docs.rs` rustdoc-args.

// The `std` feature is a bare dependency on Symphonia; codec support is
// opt-in via the per-codec sub-features (`std-wav`, `std-mp3`, …) or the
// `all-codecs` feature (every codec at once). Enable one of them to get
// `audiofp::io`. Features that merely imply `std` (`neural`, `watermark`,
// `rayon`, `mimalloc`) are unaffected.
#[cfg(all(
    feature = "std",
    not(any(
        feature = "std-mp3",
        feature = "std-aac",
        feature = "std-flac",
        feature = "std-ogg",
        feature = "std-wav",
        feature = "std-mp4",
        feature = "std-aiff",
        feature = "std-mkv",
        feature = "std-adpcm",
        feature = "std-alac",
        feature = "all-codecs",
        feature = "neural",
        feature = "watermark",
        feature = "rayon",
        feature = "mimalloc"
    ))
))]
compile_error!(
    "the `std` feature enables no codecs by itself; enable at least one \
     per-codec feature (e.g. `std-wav`, `std-mp3`, `std-flac`, `std-ogg`, \
     `std-aac`, `std-mp4`, or the extended `std-aiff` / `std-mkv` / \
     `std-adpcm` / `std-alac`), or `all-codecs` for every codec, to use `audiofp::io`"
);

extern crate alloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(feature = "std")]
pub mod cache;
pub mod classical;
pub mod dsp;
#[cfg(any(
    feature = "std-mp3",
    feature = "std-aac",
    feature = "std-flac",
    feature = "std-ogg",
    feature = "std-wav",
    feature = "std-mp4",
    feature = "std-aiff",
    feature = "std-mkv",
    feature = "std-adpcm",
    feature = "std-alac",
    feature = "all-codecs"
))]
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

// Compile-time assertions: all public types must be Send + Sync so they are
// usable in async contexts and can be shared across threads.
#[cfg(test)]
mod send_sync_assertions {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_sync() {
        // Error type
        assert_send_sync::<AfpError>();

        // Value types
        assert_send_sync::<SampleRate>();
        assert_send_sync::<TimestampMs>();
        assert_send_sync::<matching::TimeOffset>();
        assert_send_sync::<matching::MatchResult>();

        // Hash/fingerprint types
        assert_send_sync::<classical::WangHash>();
        assert_send_sync::<classical::WangFingerprint>();
        assert_send_sync::<classical::PanakoHash>();
        assert_send_sync::<classical::PanakoFingerprint>();
        assert_send_sync::<classical::HaitsmaFingerprint>();

        // Extractors
        assert_send_sync::<classical::Wang>();
        assert_send_sync::<classical::Panako>();
        assert_send_sync::<classical::Haitsma>();

        // Streaming extractors
        assert_send_sync::<classical::StreamingWang>();
        assert_send_sync::<classical::StreamingPanako>();
        assert_send_sync::<classical::StreamingHaitsma>();

        // Matchers
        assert_send_sync::<matching::WangMatcher>();
        assert_send_sync::<matching::HaitsmaMatcher>();
        assert_send_sync::<matching::PanakoMatcher>();

        // Configs
        assert_send_sync::<classical::WangConfig>();
        assert_send_sync::<classical::PanakoConfig>();
        assert_send_sync::<classical::HaitsmaConfig>();
        assert_send_sync::<matching::WangMatchConfig>();
        assert_send_sync::<matching::HaitsmaMatchConfig>();
        assert_send_sync::<matching::PanakoMatchConfig>();
    }
}
