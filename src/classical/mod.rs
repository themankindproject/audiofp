//! Classical (DSP-only) fingerprinters.
//!
//! Three independent extractors, all `no_std + alloc` in API shape.
//! Each makes a different storage / robustness tradeoff; pick the one
//! that matches your workload. The current FFT dependency chain still
//! keeps the no_std path host-only today:
//!
//! | Algorithm   | Output                  | Sample rate | Frame rate | Storage / sec       | Best for                                |
//! | ----------- | ----------------------- | ----------- | ---------- | ------------------- | --------------------------------------- |
//! | [`Wang`]    | Anchor-target landmarks | 8 kHz       | 62.5 fps   | ~2.4 KB (fan-out 10)| Music ID, Shazam-style                  |
//! | [`Panako`]  | Triplet hashes          | 8 kHz       | 62.5 fps   | ~2.0 KB (fan-out 5) | Tempo-robust music ID (±5 % stretch)    |
//! | [`Haitsma`] | 32 sign bits / frame    | 5 kHz       | 78.125 fps | 312 B               | Compact dense IDs, fastest extraction   |
//!
//! Each fingerprinter has an offline ([`crate::Fingerprinter`]) and a
//! streaming ([`crate::StreamingFingerprinter`]) variant. The streaming
//! variants emit hashes incrementally and match the offline extractor
//! under arbitrary chunking — verified down to the 1-sample-per-push
//! pathological case: Haitsma bit-exactly, Wang and Panako as hash
//! multisets when the stream ends with `flush_complete` (the legacy
//! `flush` can drop hashes near a fractional-second end; audit F01).
//!
//! All hash structs are `bytemuck::Pod` (cast-to-bytes safe), so you can
//! persist them directly to mmap'd files or ship them across an FFI
//! boundary without serialisation.

pub mod haitsma;
pub(crate) mod landmark_common;
pub mod panako;
pub(crate) mod stream;
#[cfg(test)]
pub(crate) mod testing;
pub mod wang;

pub use haitsma::{Haitsma, HaitsmaConfig, HaitsmaFingerprint, StreamingHaitsma};
pub use panako::{Panako, PanakoConfig, PanakoFingerprint, PanakoHash, StreamingPanako};
pub use wang::{StreamingWang, Wang, WangConfig, WangFingerprint, WangHash};

/// Clamp the shared limit fields of [`WangConfig`] / [`PanakoConfig`] to
/// safe bounds: min-1 floors prevent underflow/empty output, ceilings
/// prevent OOM from extreme values, and zero-value limits are rejected
/// (they would reject all inputs/outputs).
macro_rules! sanitize_cfg {
    ($cfg:expr) => {{
        $cfg.target_zone_t = $cfg.target_zone_t.clamp(1, 512);
        $cfg.fan_out = $cfg.fan_out.clamp(1, 64);
        $cfg.peaks_per_sec = $cfg.peaks_per_sec.clamp(1, 500);
        if $cfg.max_input_samples == Some(0) {
            $cfg.max_input_samples = Some(1);
        }
        if $cfg.max_hashes == Some(0) {
            $cfg.max_hashes = Some(1);
        }
        if $cfg.max_pending_anchors == Some(0) {
            $cfg.max_pending_anchors = Some(1);
        }
        if $cfg.max_push_samples == Some(0) {
            $cfg.max_push_samples = Some(1);
        }
        $cfg.target_zone_f = $cfg.target_zone_f.clamp(1, 512);
        // `f32::clamp` passes NaN through unchanged, and every peak
        // comparison against a NaN floor is false: extraction would drop
        // every peak yet still return `Ok` with an empty fingerprint.
        // Fall back to the documented default floor instead.
        if $cfg.min_anchor_mag_db.is_nan() {
            $cfg.min_anchor_mag_db = -50.0;
        }
        // The front-end clamps power at 1e-12 before the dB conversion, so
        // every cell is at least -120 dB and digital silence sits exactly
        // on that floor. A peak floor at or below it lets every silent
        // cell through as a plateau peak — silence then hashes, and
        // matches other silence. Keep the floor strictly above it.
        $cfg.min_anchor_mag_db = $cfg
            .min_anchor_mag_db
            .clamp(crate::classical::MIN_ANCHOR_MAG_DB_FLOOR, 0.0);
    }};
}
pub(crate) use sanitize_cfg;

/// Lowest `min_anchor_mag_db` the landmark extractors accept: 1 dB above
/// the -120 dB log floor (`10·log10(1e-12)`) of their dB spectrogram.
pub(crate) const MIN_ANCHOR_MAG_DB_FLOOR: f32 = -119.0;
