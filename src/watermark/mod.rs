//! Audio watermarks.
//!
//! Two subsystems live here; only the classical one is supported going
//! forward.
//!
//! # Classical embed and detect (supported)
//!
//! Feature `classical-watermark`. A keyed embedder and detector at
//! `audiofp::watermark::classical`. FFT and `alloc` only: it does not imply
//! `std` and it does not pull in `tract-onnx`.
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.4", features = ["classical-watermark"] }
//! ```
//!
//! # ONNX detector (deprecated — removed in 0.5.0)
//!
//! Feature `watermark` (implies `std` and `tract-onnx`). An AudioSeal-compatible
//! loader and decoder. The caller supplies the ONNX model. Types are
//! `WatermarkConfig`, `WatermarkDetector`, and `WatermarkResult`, both from
//! `audiofp::watermark::detector` and re-exported at `audiofp::watermark`.
//!
//! Deprecated as of 0.4.5: this subsystem, its types, the `watermark` feature,
//! and the `tract-onnx` dependency it pulls in leave the crate in 0.5.0.
//! Every use emits a compiler deprecation warning. Migrate to
//! `audiofp::watermark::classical`.
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.4", features = ["watermark"] }
//! ```

// The deprecated marker lives on the types in `detector` (and the module docs
// above), not on the module declaration: a `#[deprecated]` module would also
// deprecate every item inside it, including items the test harness calls.
#[cfg(feature = "watermark")]
pub mod detector;

#[cfg(feature = "classical-watermark")]
pub mod classical;

#[cfg(all(test, feature = "watermark"))]
mod test_fixture;

// Re-exporting the deprecated types is deliberate: the paths stay available
// (with warnings at every call site) until the 0.5.0 removal.
#[cfg(feature = "watermark")]
#[allow(deprecated)]
pub use detector::{WatermarkConfig, WatermarkDetector, WatermarkResult};
