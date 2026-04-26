//! Audio watermark detection (AudioSeal-compatible).
//!
//! A thin wrapper around [`tract-onnx`] that loads a watermark detection
//! model, runs it on an audio buffer, and decodes the result into a
//! [`WatermarkResult`]. The crate ships only the loader and decoder — the
//! caller supplies the ONNX export themselves (Meta's AudioSeal v0.2 is
//! the reference target).
//!
//! Available only when the `watermark` feature is enabled.

pub mod detector;

pub use detector::{WatermarkConfig, WatermarkDetector, WatermarkResult};
