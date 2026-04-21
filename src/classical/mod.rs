//! Classical (DSP-only) fingerprinters.
//!
//! Three independent extractors, all `no_std + alloc`:
//!
//! - [`Wang`] — Shazam-style anchor-target landmark pairs.
//! - `Panako` (Phase 3c) — temporal triplet hashes.
//! - `Haitsma` (Phase 3d) — Philips robust hash bands.

pub mod wang;

pub use wang::{StreamingWang, Wang, WangConfig, WangFingerprint, WangHash};
