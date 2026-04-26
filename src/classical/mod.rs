//! Classical (DSP-only) fingerprinters.
//!
//! Three independent extractors, all `no_std + alloc`. Each makes a
//! different storage / robustness tradeoff; pick the one that matches
//! your workload:
//!
//! | Algorithm   | Output                  | Sample rate | Frame rate | Storage / sec       | Best for                                |
//! | ----------- | ----------------------- | ----------- | ---------- | ------------------- | --------------------------------------- |
//! | [`Wang`]    | Anchor-target landmarks | 8 kHz       | 62.5 fps   | ~2.4 KB (fan-out 10)| Music ID, Shazam-style                  |
//! | [`Panako`]  | Triplet hashes          | 8 kHz       | 62.5 fps   | ~2.0 KB (fan-out 5) | Tempo-robust music ID (±5 % stretch)    |
//! | [`Haitsma`] | 32 sign bits / frame    | 5 kHz       | 78.125 fps | 312 B               | Compact dense IDs, fastest extraction   |
//!
//! Each fingerprinter has an offline ([`crate::Fingerprinter`]) and a
//! streaming ([`crate::StreamingFingerprinter`]) variant. The streaming
//! variants emit hashes incrementally with **bit-exact** parity to the
//! offline extractor under arbitrary chunking — verified down to the
//! 1-sample-per-push pathological case.
//!
//! All hash structs are `bytemuck::Pod` (cast-to-bytes safe), so you can
//! persist them directly to mmap'd files or ship them across an FFI
//! boundary without serialisation.

pub mod haitsma;
pub mod panako;
pub mod wang;

pub use haitsma::{Haitsma, HaitsmaConfig, HaitsmaFingerprint, StreamingHaitsma};
pub use panako::{Panako, PanakoConfig, PanakoFingerprint, PanakoHash, StreamingPanako};
pub use wang::{StreamingWang, Wang, WangConfig, WangFingerprint, WangHash};
