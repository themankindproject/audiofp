//! `afp` — audio fingerprinting SDK.
//!
//! Compiles `no_std + alloc` by default (when the `std` feature is disabled),
//! so the DSP primitives and classical fingerprinters can target embedded
//! systems such as `thumbv7em-none-eabihf`.
//!
//! # Quick tour
//!
//! - [`AfpError`] / [`Result`] — the crate's error model.
//! - [`SampleRate`], [`AudioBuffer`], [`TimestampMs`] — shared value types.
//! - [`Fingerprinter`], [`StreamingFingerprinter`] — the two traits every
//!   extractor implements.
//!
//! Concrete fingerprinters land in subsequent phases.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

extern crate alloc;

mod error;
mod fp;
mod types;

pub use error::{AfpError, Result};
pub use fp::{Fingerprinter, StreamingFingerprinter};
pub use types::{AudioBuffer, SampleRate, TimestampMs};

/// Crate version string, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
