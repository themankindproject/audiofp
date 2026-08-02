//! Convenience re-exports of the most commonly used types.
//!
//! ```ignore
//! use audiofp::prelude::*;
//!
//! let samples = vec![0.0_f32; 16_000];
//! let mut wang = Wang::default();
//! let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
//! assert_eq!(fp.frames_per_sec, 62.5);
//! ```
//!
//! Pulls in: error types, value types, the two core traits, and the three
//! classical fingerprinters (offline + streaming) with their config and
//! hash types.
//!
//! Sample-rate constants live on the [`SampleRate`] type itself
//! (e.g. `SampleRate::HZ_8000`) and do not need a separate import.
//!
//! Feature-gated types (`io`, `neural`, `watermark`) are **not** included
//! — import those from their respective modules.

pub use crate::classical::{
    Haitsma, HaitsmaConfig, HaitsmaFingerprint, Panako, PanakoConfig, PanakoFingerprint,
    PanakoHash, StreamingHaitsma, StreamingPanako, StreamingWang, Wang, WangConfig,
    WangFingerprint, WangHash,
};
pub use crate::error::{AfpError, Result};
pub use crate::fp::{Fingerprinter, StreamingFingerprinter};
pub use crate::serial::FingerprintEnvelope;
pub use crate::types::{SampleRate, TimestampMs};
