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

/// All chromaprint C functions return `1` on success, `0` on error.
macro_rules! ok {
    ($call:expr, $what:expr) => {
        if $call != 1 {
            panic!("{0} failed", $what);
        }
    };
}

/// Version string, e.g. `"1.6.0"` — recorded in the report.
pub fn version() -> String {
    let p = unsafe { chromaprint_get_version() };
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
pub fn extract(samples: &[f32], sample_rate: u32) -> Vec<u32> {
    let s16: Vec<i16> = samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();
    let ctx = unsafe { chromaprint_new(ALGO_DEFAULT) };
    assert!(!ctx.is_null(), "chromaprint_new failed");
    ok!(
        unsafe { chromaprint_start(ctx, sample_rate as c_int, 1) },
        "chromaprint_start"
    );
    // Feed the whole buffer; chromaprint buffers internally.
    ok!(
        unsafe { chromaprint_feed(ctx, s16.as_ptr(), s16.len() as c_int) },
        "chromaprint_feed"
    );
    ok!(unsafe { chromaprint_finish(ctx) }, "chromaprint_finish");
    let mut arr: *mut u32 = std::ptr::null_mut();
    let mut size: c_int = 0;
    ok!(
        unsafe { chromaprint_get_raw_fingerprint(ctx, &mut arr, &mut size) },
        "chromaprint_get_raw_fingerprint"
    );
    unsafe { chromaprint_free(ctx) };
    if arr.is_null() || size <= 0 {
        return Vec::new(); // silent / short audio -> empty fingerprint
    }
    let raw = unsafe { std::slice::from_raw_parts(arr, size as usize) }.to_vec();
    unsafe { chromaprint_dealloc(arr as *mut c_void) };
    raw
}

/// Encode a raw fingerprint to the AcoustID base64 blob (for the report /
/// cross-checks). `base64 = 1` -> base64-encoded ASCII string.
pub fn to_base64(raw: &[u32]) -> String {
    let mut enc: *mut c_char = std::ptr::null_mut();
    let mut enc_size: c_int = 0;
    ok!(
        unsafe {
            chromaprint_encode_fingerprint(
                raw.as_ptr(),
                raw.len() as c_int,
                ALGO_DEFAULT,
                &mut enc,
                &mut enc_size,
                1,
            )
        },
        "chromaprint_encode_fingerprint"
    );
    let s = unsafe { std::ffi::CStr::from_ptr(enc) }.to_bytes().to_vec();
    unsafe { chromaprint_dealloc(enc as *mut c_void) };
    String::from_utf8_lossy(&s).into_owned()
}

/// Roundtrip self-check (replaces the CLI cross-check, since no CLI is
/// installed): encode then decode must reproduce the raw fingerprint
/// exactly. This is the single most important FFI validation — if it
/// fails, the pointer/size/free bookkeeping above is wrong.
pub fn encode_decode_roundtrip(raw: &[u32]) -> bool {
    let b64 = to_base64(raw);
    let enc = b64.as_ptr() as *const c_char;
    let mut dec: *mut u32 = std::ptr::null_mut();
    let mut dec_size: c_int = 0;
    let mut algo: c_int = 0;
    if unsafe {
        chromaprint_decode_fingerprint(
            enc,
            b64.len() as c_int,
            &mut dec,
            &mut dec_size,
            &mut algo,
            1,
        )
    } != 1
    {
        return false;
    }
    let decoded = unsafe { std::slice::from_raw_parts(dec, dec_size as usize) };
    let good = decoded.len() == raw.len() && decoded == raw;
    unsafe { chromaprint_dealloc(dec as *mut c_void) };
    good
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
}
