//! Minimal libchromaprint FFI — the stable C symbols, nothing more.
//!
//! We link the distro's libchromaprint (`libchromaprint-dev`) directly
//! instead of a crates.io wrapper: the reference implementation is the
//! point of the bakeoff, and a small `extern` block has zero wrapper risk.
//! Signatures verified against `/usr/include/chromaprint.h` (v1.6.0) on
//! 2026-08-30. See `BENCHMARKS.md` for the pinned version.
//!
//! Conventions (from the header):
//! - all `chromaprint_*` calls return `1` on success, `0` on error;
//! - raw fingerprints are arrays of `u32` items (NOT 64-bit words);
//! - the context resamples internally, so we feed PCM at its native rate.

use std::ffi::{c_char, c_int, c_void};

/// Opaque chromaprint context (`struct ChromaprintContextPrivate`).
#[repr(C)]
pub struct ChromaprintContext {
    _private: [u8; 0],
}

/// `CHROMAPRINT_ALGORITHM_DEFAULT == CHROMAPRINT_ALGORITHM_TEST2`.
const ALGO_DEFAULT: c_int = 1;

/// Maximum PCM samples accepted by the C API (`size` is a signed `c_int`).
const MAX_FEED_SAMPLES: usize = c_int::MAX as usize;

#[link(name = "chromaprint")]
unsafe extern "C" {
    fn chromaprint_get_version() -> *const c_char;
    fn chromaprint_new(algorithm: c_int) -> *mut ChromaprintContext;
    fn chromaprint_free(ctx: *mut ChromaprintContext);
    fn chromaprint_start(
        ctx: *mut ChromaprintContext,
        sample_rate: c_int,
        num_channels: c_int,
    ) -> c_int;
    fn chromaprint_feed(ctx: *mut ChromaprintContext, data: *const i16, size: c_int) -> c_int;
    fn chromaprint_finish(ctx: *mut ChromaprintContext) -> c_int;
    fn chromaprint_get_raw_fingerprint(
        ctx: *mut ChromaprintContext,
        fingerprint: *mut *mut u32,
        size: *mut c_int,
    ) -> c_int;
    fn chromaprint_encode_fingerprint(
        fp: *const u32,
        size: c_int,
        algorithm: c_int,
        encoded_fp: *mut *mut c_char,
        encoded_size: *mut c_int,
        base64: c_int,
    ) -> c_int;
    fn chromaprint_decode_fingerprint(
        encoded_fp: *const c_char,
        encoded_size: c_int,
        fp: *mut *mut u32,
        size: *mut c_int,
        algorithm: *mut c_int,
        base64: c_int,
    ) -> c_int;

    fn chromaprint_dealloc(ptr: *mut c_void);
}

/// RAII wrapper around a chromaprint context pointer.
struct CpContext(*mut ChromaprintContext);

impl CpContext {
    fn new() -> Option<Self> {
        // SAFETY: chromaprint_new is the documented constructor.
        let ctx = unsafe { chromaprint_new(ALGO_DEFAULT) };
        if ctx.is_null() { None } else { Some(Self(ctx)) }
    }

    fn start(&self, sample_rate: c_int, num_channels: c_int) -> bool {
        // SAFETY: ctx is a live chromaprint context.
        unsafe { chromaprint_start(self.0, sample_rate, num_channels) == 1 }
    }

    fn feed(&self, data: &[i16]) -> bool {
        if data.len() > MAX_FEED_SAMPLES {
            return false;
        }
        // SAFETY: data is a contiguous i16 slice; size fits c_int by guard above.
        unsafe { chromaprint_feed(self.0, data.as_ptr(), data.len() as c_int) == 1 }
    }

    fn finish(&self) -> bool {
        // SAFETY: ctx is a live chromaprint context.
        unsafe { chromaprint_finish(self.0) == 1 }
    }

    fn take_raw_fingerprint(self) -> Option<CpFingerprint> {
        let mut arr: *mut u32 = std::ptr::null_mut();
        let mut size: c_int = 0;
        // SAFETY: ctx is live until this call returns; out-pointers are valid.
        let ok = unsafe { chromaprint_get_raw_fingerprint(self.0, &mut arr, &mut size) } == 1;
        let _allocation = CpAlloc(arr.cast());
        if !ok || arr.is_null() || size <= 0 {
            return None;
        }
        let len = size as usize;
        // SAFETY: chromaprint returned `size` valid u32 items at `arr`.
        let raw = unsafe { std::slice::from_raw_parts(arr, len) }.to_vec();

        Some(CpFingerprint(raw))
    }
}

impl Drop for CpContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: self.0 was created by chromaprint_new and not yet freed.
            unsafe { chromaprint_free(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// Heap block returned by chromaprint and freed with chromaprint_dealloc.
struct CpAlloc(*mut c_void);

impl Drop for CpAlloc {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: pointer came from chromaprint and is freed exactly once here.
            unsafe { chromaprint_dealloc(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

/// Owned raw fingerprint (`u32` items).
struct CpFingerprint(Vec<u32>);

/// Convert mono f32 PCM in `[-1, 1]` to i16 for chromaprint_feed.
fn samples_to_i16(samples: &[f32]) -> Option<Vec<i16>> {
    if samples.len() > MAX_FEED_SAMPLES {
        return None;
    }
    Some(
        samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect(),
    )
}

fn sample_rate_as_c_int(sample_rate: u32) -> Option<c_int> {
    if sample_rate == 0 || sample_rate > c_int::MAX as u32 {
        None
    } else {
        Some(sample_rate as c_int)
    }
}

/// Version string, e.g. `"1.6.0"` — recorded in the report.
pub fn version() -> String {
    let p = unsafe { chromaprint_get_version() };
    if p.is_null() {
        return String::new();
    }
    // SAFETY: chromaprint_get_version returns a static NUL-terminated C string.
    let b = unsafe { std::ffi::CStr::from_ptr(p) }.to_bytes();
    String::from_utf8_lossy(b).into_owned()
}

/// Extract a chromaprint raw fingerprint from mono f32 PCM in `[-1, 1]`.
///
/// Returns the raw fingerprint as `u32` items — the native 32-bit word
/// layout chromaprint uses internally, which is what the overlap and
/// identification metrics compare. Samples are fed **at their native
/// rate**; `chromaprint_start` resamples internally, exactly as the CLI
/// does (no manual resampler).
///
/// Invalid inputs (zero rate, oversize buffers) yield an empty fingerprint
/// instead of panicking.
pub fn extract(samples: &[f32], sample_rate: u32) -> Vec<u32> {
    let Some(rate) = sample_rate_as_c_int(sample_rate) else {
        return Vec::new();
    };
    let Some(s16) = samples_to_i16(samples) else {
        return Vec::new();
    };
    let Some(ctx) = CpContext::new() else {
        return Vec::new();
    };
    if !ctx.start(rate, 1) || !ctx.feed(&s16) || !ctx.finish() {
        return Vec::new();
    }
    ctx.take_raw_fingerprint()
        .map(|CpFingerprint(v)| v)
        .unwrap_or_default()
}

/// Encode a raw fingerprint to the AcoustID base64 blob (for the report /
/// cross-checks). `base64 = 1` -> base64-encoded ASCII string.
pub fn to_base64(raw: &[u32]) -> String {
    if raw.len() > MAX_FEED_SAMPLES {
        return String::new();
    }
    let mut enc: *mut c_char = std::ptr::null_mut();
    let mut enc_size: c_int = 0;
    // SAFETY: raw is a contiguous u32 slice; out-pointers are valid stack refs.
    let ok = unsafe {
        chromaprint_encode_fingerprint(
            raw.as_ptr(),
            raw.len() as c_int,
            ALGO_DEFAULT,
            &mut enc,
            &mut enc_size,
            1,
        )
    } == 1;
    let _allocation = CpAlloc(enc.cast());
    if !ok || enc.is_null() || enc_size <= 0 {
        return String::new();
    }
    // SAFETY: the C API returned enc_size initialized bytes; the guard
    // retains ownership until the string has been copied.
    let bytes = unsafe { std::slice::from_raw_parts(enc.cast::<u8>(), enc_size as usize) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Roundtrip self-check (replaces the CLI cross-check, since no CLI is
/// installed): encode then decode must reproduce the raw fingerprint
/// exactly. This is the single most important FFI validation — if it
/// fails, the pointer/size/free bookkeeping above is wrong.
pub fn encode_decode_roundtrip(raw: &[u32]) -> bool {
    if raw.len() > MAX_FEED_SAMPLES {
        return false;
    }
    let b64 = to_base64(raw);
    if b64.len() > MAX_FEED_SAMPLES {
        return false;
    }
    if b64.is_empty() {
        return raw.is_empty();
    }
    let enc = b64.as_ptr() as *const c_char;
    let enc_len = b64.len() as c_int;
    // Decode only bytes produced by our own encoder. Avoid the newer
    // decode_fingerprint_header symbol, absent from distro chromaprint 1.5.
    let mut dec: *mut u32 = std::ptr::null_mut();
    let mut dec_size: c_int = 0;
    let mut algo: c_int = 0;
    let ok = unsafe {
        chromaprint_decode_fingerprint(enc, enc_len, &mut dec, &mut dec_size, &mut algo, 1)
    } == 1;
    let _allocation = CpAlloc(dec.cast());
    if !ok {
        return false;
    }
    if dec_size == 0 {
        return raw.is_empty();
    }
    if dec.is_null() || dec_size < 0 {
        return false;
    }
    let len = dec_size as usize;
    // SAFETY: chromaprint returned `dec_size` items at `dec`.
    let decoded = unsafe { std::slice::from_raw_parts(dec, len) };
    decoded.len() == raw.len() && decoded == raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }

    #[test]
    fn short_audio_yields_empty_or_short_fingerprint() {
        // 0.1 s of tone at 8 kHz: below chromaprint's internal minimum,
        // so we expect a small or empty raw fingerprint — but no crash.
        let samples: Vec<f32> = (0..800).map(|i| (i as f32 * 0.02).sin()).collect();
        let raw = extract(&samples, 8_000);
        assert!(raw.len() < 64);
    }

    #[test]
    fn zero_sample_rate_returns_empty() {
        let samples = vec![0.0_f32; 100];
        assert!(extract(&samples, 0).is_empty());
    }

    #[test]
    fn sample_rate_outside_c_int_range_is_rejected() {
        assert_eq!(sample_rate_as_c_int(u32::MAX), None);
        assert_eq!(sample_rate_as_c_int(8_000), Some(8_000));
    }

    #[test]
    fn empty_samples_are_handled_without_panicking() {
        assert!(extract(&[], 8_000).is_empty());
    }

    #[test]
    fn encode_decode_roundtrip_nonempty_fingerprint() {
        // ~16 s at 8 kHz — chromaprint's internal minimum for a non-empty raw fp.
        let samples: Vec<f32> = (0..128_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let raw = extract(&samples, 8_000);
        assert!(
            !raw.is_empty(),
            "expected a non-empty fingerprint from tone"
        );
        assert!(
            encode_decode_roundtrip(&raw),
            "encode/decode must reproduce a real extracted fingerprint"
        );
    }

    #[test]
    fn encode_decode_roundtrip_empty_fingerprint() {
        assert!(
            encode_decode_roundtrip(&[]),
            "empty raw fingerprint must roundtrip through encode/decode"
        );
    }
}
