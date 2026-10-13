//! Audio watermarks.
//!
//! Two independent subsystems live here.
//!
//! # ONNX detector
//!
//! Feature `watermark` (implies `std` and `tract-onnx`). An AudioSeal-compatible
//! loader and decoder. The caller supplies the ONNX model. Types are
//! `WatermarkConfig`, `WatermarkDetector`, and `WatermarkResult`, both from
//! `audiofp::watermark::detector` and re-exported at `audiofp::watermark`.
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.4", features = ["watermark"] }
//! ```
//!
//! # Classical embed and detect
//!
//! Feature `classical-watermark`. A keyed embedder and detector at
//! `audiofp::watermark::classical`. FFT and `alloc` only: it does not imply
//! `std` and it does not pull in `tract-onnx`.
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.4", features = ["classical-watermark"] }
//! ```

#[cfg(feature = "watermark")]
pub mod detector;

#[cfg(feature = "classical-watermark")]
pub mod classical;

#[cfg(all(test, feature = "watermark"))]
mod test_fixture;

#[cfg(feature = "watermark")]
pub use detector::{WatermarkConfig, WatermarkDetector, WatermarkResult};
