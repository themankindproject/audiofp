//! `afp` — audio fingerprinting SDK.
//!
//! Compiles `no_std + alloc` by default (when `std` feature is disabled), so the
//! DSP primitives and classical fingerprinters can target embedded systems
//! such as `thumbv7em-none-eabihf`.
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Crate version string, sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
