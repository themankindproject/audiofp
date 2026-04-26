//! Audio watermark detection (AudioSeal-compatible).
//!
//! A thin wrapper around [`tract-onnx`] that loads a watermark-detector
//! ONNX model, runs it on an audio buffer, and decodes the model's two
//! outputs (detection scores + message logits) into a
//! [`WatermarkResult`].
//!
//! The crate ships only the loader and decoder — the caller supplies
//! the ONNX export themselves. Meta's [AudioSeal v0.2] is the reference
//! target; any model that follows the contract documented on
//! [`WatermarkConfig`] will work.
//!
//! Available only when the `watermark` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! audiofp = { version = "0.1", features = ["watermark"] }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use audiofp::watermark::{WatermarkConfig, WatermarkDetector};
//! use audiofp::{AudioBuffer, SampleRate};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut det = WatermarkDetector::new(WatermarkConfig::new("audioseal_v0.2.onnx"))?;
//! let samples: Vec<f32> = vec![/* … 16 kHz mono PCM … */];
//! let buf = AudioBuffer { samples: &samples, rate: SampleRate::new(16_000).unwrap() };
//! let r = det.detect(buf)?;
//!
//! if r.detected {
//!     println!("watermarked, confidence {:.3}, message {:#018b}",
//!              r.confidence, r.message);
//! }
//! # Ok(()) }
//! ```
//!
//! [AudioSeal v0.2]: https://github.com/facebookresearch/audioseal
//! [`tract-onnx`]: https://docs.rs/tract-onnx

pub mod detector;

pub use detector::{WatermarkConfig, WatermarkDetector, WatermarkResult};
