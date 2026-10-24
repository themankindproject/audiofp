//! Keyed audio watermarking.
//!
//! # Classical embed and detect
//!
//! Feature `classical-watermark`. A keyed embedder and detector at
//! `audiofp::watermark::classical`. FFT and `alloc` only: it does not imply
//! `std`.
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.4", features = ["classical-watermark"] }
//! ```

#[cfg(feature = "classical-watermark")]
pub mod classical;
