//! Digital signal processing primitives used by every fingerprinter.
//!
//! All modules in `dsp` compile under `no_std + alloc` so they can be reused
//! on Cortex-M targets via `--no-default-features`.

pub mod mel;
pub mod peaks;
pub mod resample;
pub mod stft;
pub mod windows;
