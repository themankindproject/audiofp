//! Classical (DSP-only) fingerprinters.
//!
//! Three independent extractors, all `no_std + alloc`:
//!
//! - [`Wang`] — Shazam-style anchor-target landmark pairs.
//! - [`Panako`] — temporal triplet hashes with tempo-invariant β ratio.
//! - [`Haitsma`] — Philips robust hash over 33 logarithmic bands.

pub mod haitsma;
pub mod panako;
pub mod wang;

pub use haitsma::{Haitsma, HaitsmaConfig, HaitsmaFingerprint, StreamingHaitsma};
pub use panako::{Panako, PanakoConfig, PanakoFingerprint, PanakoHash, StreamingPanako};
pub use wang::{StreamingWang, Wang, WangConfig, WangFingerprint, WangHash};
