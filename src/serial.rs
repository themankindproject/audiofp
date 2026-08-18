//! Lightweight binary serialization for fingerprint types.
//!
//! Each fingerprint can be round-tripped through a compact binary format
//! via [`to_bytes`] / [`from_bytes`], and metadata about a fingerprint
//! blob is available without parsing the hash payload through
//! [`FingerprintEnvelope::peek`] (raw bytes) or [`envelope`] (on a
//! parsed fingerprint).
//!
//! # Wire format (v1)
//!
//! ```text
//! [magic: 8 bytes "AUDIOFP\0"] [version: u8 = 1] [algorithm_id: u8]
//! [hash_count: u32 LE] [fps: f32 LE] [hashes: Pod bytes]
//! ```
//!
//! The hash payload is the raw `bytemuck::cast_slice` representation of
//! each algorithm's Pod hash type, meaning zero-copy reads on
//! little-endian hosts.
//!
//! [`to_bytes`]: crate::classical::WangFingerprint::to_bytes
//! [`from_bytes`]: crate::classical::WangFingerprint::from_bytes
//! [`envelope`]: crate::classical::WangFingerprint::envelope
//! [`peek`]: FingerprintEnvelope::peek

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use bytemuck::cast_slice;

use crate::classical::{
    HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangFingerprint, WangHash,
};
use crate::{AfpError, Result};

/// Magic header identifying an `audiofp` binary fingerprint blob.
const MAGIC: [u8; 8] = *b"AUDIOFP\0";

/// Current serialization format version.
const FORMAT_VERSION: u8 = 1;

/// Fixed-size header: magic (8) + version (1) + algorithm_id (1) +
/// hash_count (4) + fps (4) = 18 bytes.
const HEADER_SIZE: usize = 8 + 1 + 1 + 4 + 4;

const ALG_WANG: u8 = 0;
const ALG_PANAKO: u8 = 1;
const ALG_HAITSMA: u8 = 2;

/// Metadata envelope for a serialized fingerprint.
///
/// Provides a quick summary of a fingerprint's provenance without
/// requiring full deserialization.
///
/// # Example
///
/// ```
/// use audiofp::classical::Wang;
/// use audiofp::{Fingerprinter, SampleRate};
///
/// let samples = vec![0.0_f32; 8_000 * 3];
/// let mut wang = Wang::default();
/// let fp = wang.extract(&samples, SampleRate::HZ_8000).unwrap();
/// let env = fp.envelope();
/// assert_eq!(env.algorithm, "wang-v1");
/// assert_eq!(env.sample_rate, 8_000);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FingerprintEnvelope {
    /// Algorithm name string (e.g. `"wang-v1"`, `"panako-v2"`, `"haitsma-v1"`).
    pub algorithm: &'static str,
    /// Version of the **reading** crate. The v1 wire format does not
    /// persist the producer's version, so after a
    /// [`from_bytes`](crate::classical::WangFingerprint::from_bytes)
    /// round-trip this reports the current crate, not necessarily the
    /// crate that wrote the blob.
    pub crate_version: &'static str,
    /// Sample rate the algorithm expects (Hz).
    pub sample_rate: u32,
    /// STFT frame rate (frames per second).
    pub frames_per_sec: f32,
    /// Number of hashes (or frames, for Haitsma) in the fingerprint.
    pub hash_count: usize,
}

impl FingerprintEnvelope {
    /// Read a blob's metadata without deserializing the hash payload.
    ///
    /// Parses and validates only the fixed 18-byte header — the payload
    /// can be arbitrarily large and is never touched. Useful for
    /// triaging mixed-format blobs before committing to a full decode.
    ///
    /// # Errors
    ///
    /// [`AfpError::Deserialize`] on short buffers, bad magic,
    /// unsupported format version, an unknown algorithm id, or a
    /// non-finite / non-positive frame rate.
    pub fn peek(bytes: &[u8]) -> Result<Self> {
        const fn alg_name(alg_id: u8) -> Option<&'static str> {
            match alg_id {
                ALG_WANG => Some("wang-v1"),
                ALG_PANAKO => Some("panako-v2"),
                ALG_HAITSMA => Some("haitsma-v1"),
                _ => None,
            }
        }
        const fn alg_sample_rate(alg_id: u8) -> u32 {
            match alg_id {
                ALG_PANAKO => 8_000,
                ALG_HAITSMA => 5_000,
                _ => 8_000,
            }
        }

        // No expected-algorithm check: peek validates the id against the
        // known-algorithm table itself so foreign ids get a precise error.
        let (alg_id, hash_count, fps) = read_header(bytes, None)?;
        let algorithm = alg_name(alg_id)
            .ok_or_else(|| AfpError::Deserialize(format!("unknown algorithm id: {alg_id}")))?;
        Ok(FingerprintEnvelope {
            algorithm,
            crate_version: crate::VERSION,
            sample_rate: alg_sample_rate(alg_id),
            frames_per_sec: fps,
            hash_count: hash_count as usize,
        })
    }
}

/// Write the fixed header into a pre-allocated `Vec<u8>`.
fn write_header(buf: &mut Vec<u8>, alg_id: u8, hash_count: u32, fps: f32) {
    buf.extend_from_slice(&MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(alg_id);
    buf.extend_from_slice(&hash_count.to_le_bytes());
    buf.extend_from_slice(&fps.to_le_bytes());
}

/// Parse and validate the fixed header, returning `(algorithm_id, hash_count, fps)`.
///
/// With `expected_alg = None` the algorithm id is accepted as-is (used
/// by [`FingerprintEnvelope::peek`], which validates the id against the
/// known-algorithm table itself).
///
/// [`FingerprintEnvelope::peek`]: FingerprintEnvelope::peek
fn read_header(bytes: &[u8], expected_alg: Option<u8>) -> Result<(u8, u32, f32)> {
    if bytes.len() < HEADER_SIZE {
        return Err(AfpError::Deserialize(format!(
            "buffer too short: {} bytes, need at least {}",
            bytes.len(),
            HEADER_SIZE
        )));
    }
    if bytes[..8] != MAGIC {
        return Err(AfpError::Deserialize(
            "invalid magic: not an audiofp binary blob".to_string(),
        ));
    }
    let version = bytes[8];
    if version != FORMAT_VERSION {
        return Err(AfpError::Deserialize(format!(
            "unsupported format version: got {version}, expected {FORMAT_VERSION}"
        )));
    }
    let alg_id = bytes[9];
    if let Some(expected) = expected_alg
        && alg_id != expected
    {
        return Err(AfpError::Deserialize(format!(
            "algorithm mismatch: blob has id {alg_id}, expected {expected}"
        )));
    }
    let hash_count = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
    let fps = f32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
    // Validate the frame rate here (in the header) rather than only in
    // `read_payload`, so `FingerprintEnvelope::peek` — which stops at the
    // header — honours its documented contract of rejecting a non-finite /
    // non-positive frame rate.
    if !fps.is_finite() || fps <= 0.0 {
        return Err(AfpError::Deserialize(format!(
            "invalid frame rate in header: {fps} (must be finite and > 0)"
        )));
    }
    Ok((alg_id, hash_count, fps))
}

/// Read a byte slice into a `Vec<T>` where `T: Pod`.
///
/// This handles potentially-unaligned input by allocating a properly
/// aligned `Vec<T>` and copying the raw bytes into it exactly once (no
/// intermediate zero-fill). `src` must be an exact multiple of
/// `size_of::<T>()`.
fn read_pod_vec<T: bytemuck::Pod>(src: &[u8]) -> Vec<T> {
    let elem_size = core::mem::size_of::<T>();
    if elem_size == 0 || src.is_empty() {
        return Vec::new();
    }
    debug_assert_eq!(src.len() % elem_size, 0);
    bytemuck::allocation::pod_collect_to_vec(src)
}

/// Serialize a Pod hash slice with the standard header into a `Vec<u8>`.
///
/// # Panics
///
/// Panics if `values.len()` exceeds `u32::MAX`. The header stores the hash
/// count as a `u32`, so a larger slice cannot be represented; silently
/// truncating the count would produce a corrupt blob. This is unreachable
/// in practice (billions of hashes) but must fail loudly if it ever occurs.
fn pod_blob<T: bytemuck::Pod>(values: &[T], alg_id: u8, fps: f32) -> Vec<u8> {
    let value_bytes: &[u8] = cast_slice(values);
    let count = u32::try_from(values.len())
        .expect("fingerprint hash count exceeds u32::MAX and cannot be serialized");
    let mut buf = Vec::with_capacity(HEADER_SIZE + value_bytes.len());
    write_header(&mut buf, alg_id, count, fps);
    buf.extend_from_slice(value_bytes);
    buf
}

/// Parse and validate a fingerprint payload, returning `(values, fps)`.
///
/// Trailing bytes beyond the payload are intentionally ignored (forward
/// compatibility for envelope extensions).
fn read_payload<T: bytemuck::Pod>(bytes: &[u8], alg_id: u8, kind: &str) -> Result<(Vec<T>, f32)> {
    // `read_header` already validates the frame rate (finite and > 0).
    let (_alg, hash_count, fps) = read_header(bytes, Some(alg_id))?;
    let payload = &bytes[HEADER_SIZE..];
    let expected_len = (hash_count as usize).checked_mul(core::mem::size_of::<T>());
    let expected_len = expected_len.filter(|&len| payload.len() >= len);
    let Some(expected_len) = expected_len else {
        return Err(AfpError::Deserialize(format!(
            "payload too short or hash count overflows: need {} bytes for {} {kind}, got {}",
            (hash_count as usize).saturating_mul(core::mem::size_of::<T>()),
            hash_count,
            payload.len()
        )));
    };
    let values = read_pod_vec::<T>(&payload[..expected_len]);
    Ok((values, fps))
}

/// Build a metadata envelope from the per-algorithm constants.
fn envelope(
    algorithm: &'static str,
    sample_rate: u32,
    fps: f32,
    hash_count: usize,
) -> FingerprintEnvelope {
    FingerprintEnvelope {
        algorithm,
        crate_version: crate::VERSION,
        sample_rate,
        frames_per_sec: fps,
        hash_count,
    }
}

impl WangFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    ///
    /// The format is documented in the [`serial`](crate::serial) module.
    pub fn to_bytes(&self) -> Vec<u8> {
        pod_blob(&self.hashes, ALG_WANG, self.frames_per_sec)
    }

    /// Deserialize a Wang fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (hashes, frames_per_sec) = read_payload::<WangHash>(bytes, ALG_WANG, "hashes")?;
        Ok(Self {
            hashes,
            frames_per_sec,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        envelope("wang-v1", 8_000, self.frames_per_sec, self.hashes.len())
    }
}

impl PanakoFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        pod_blob(&self.hashes, ALG_PANAKO, self.frames_per_sec)
    }

    /// Deserialize a Panako fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (hashes, frames_per_sec) = read_payload::<PanakoHash>(bytes, ALG_PANAKO, "hashes")?;
        Ok(Self {
            hashes,
            frames_per_sec,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        envelope("panako-v2", 8_000, self.frames_per_sec, self.hashes.len())
    }
}

impl HaitsmaFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        pod_blob(&self.frames, ALG_HAITSMA, self.frames_per_sec)
    }

    /// Deserialize a Haitsma fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (frames, frames_per_sec) = read_payload::<u32>(bytes, ALG_HAITSMA, "frames")?;
        Ok(Self {
            frames,
            frames_per_sec,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        envelope("haitsma-v1", 5_000, self.frames_per_sec, self.frames.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn wang_round_trip_empty() {
        let fp = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        let fp2 = WangFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
        assert_eq!(fp.frames_per_sec, fp2.frames_per_sec);
    }

    #[test]
    fn wang_round_trip_with_hashes() {
        let fp = WangFingerprint {
            hashes: vec![
                WangHash {
                    hash: 0xDEAD_BEEF,
                    t_anchor: 42,
                },
                WangHash {
                    hash: 0xCAFE_BABE,
                    t_anchor: 100,
                },
            ],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 2 * 8); // 2 hashes × 8 bytes
        let fp2 = WangFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
        assert_eq!(fp.frames_per_sec, fp2.frames_per_sec);
    }

    #[test]
    fn panako_round_trip() {
        let fp = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 0x1234_5678,
                t_anchor: 10,
                t_b: 15,
                t_c: 20,
            }],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 16); // 1 hash × 16 bytes
        let fp2 = PanakoFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
        assert_eq!(fp.frames_per_sec, fp2.frames_per_sec);
    }

    #[test]
    fn haitsma_round_trip() {
        let fp = HaitsmaFingerprint {
            frames: vec![0xAAAA_BBBB, 0xCCCC_DDDD, 0x1111_2222],
            frames_per_sec: 78.125,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 3 * 4); // 3 frames × 4 bytes
        let fp2 = HaitsmaFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.frames, fp2.frames);
        assert_eq!(fp.frames_per_sec, fp2.frames_per_sec);
    }

    #[test]
    fn reject_bad_magic() {
        let mut bytes = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        }
        .to_bytes();
        bytes[0] = b'X'; // corrupt magic
        let err = WangFingerprint::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn reject_bad_version() {
        let mut bytes = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        }
        .to_bytes();
        bytes[8] = 99; // future version
        let err = WangFingerprint::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported format version"));
    }

    #[test]
    fn reject_algorithm_mismatch() {
        let bytes = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        }
        .to_bytes();
        // Try to parse Wang blob as Panako
        let err = PanakoFingerprint::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("algorithm mismatch"));
    }

    #[test]
    fn reject_truncated_header() {
        let err = WangFingerprint::from_bytes(&[0u8; 5]).unwrap_err();
        assert!(err.to_string().contains("buffer too short"));
    }

    #[test]
    fn reject_truncated_payload() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 2,
            }],
            frames_per_sec: 62.5,
        };
        let mut bytes = fp.to_bytes();
        bytes.truncate(HEADER_SIZE + 4); // only 4 of 8 needed bytes
        let err = WangFingerprint::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("payload too short"));
    }

    #[test]
    fn wang_envelope() {
        let fp = WangFingerprint {
            hashes: vec![
                WangHash {
                    hash: 1,
                    t_anchor: 0,
                },
                WangHash {
                    hash: 2,
                    t_anchor: 1,
                },
            ],
            frames_per_sec: 62.5,
        };
        let env = fp.envelope();
        assert_eq!(env.algorithm, "wang-v1");
        assert_eq!(env.sample_rate, 8_000);
        assert_eq!(env.frames_per_sec, 62.5);
        assert_eq!(env.hash_count, 2);
        assert_eq!(env.crate_version, crate::VERSION);
    }

    #[test]
    fn panako_envelope() {
        let fp = PanakoFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let env = fp.envelope();
        assert_eq!(env.algorithm, "panako-v2");
        assert_eq!(env.sample_rate, 8_000);
        assert_eq!(env.hash_count, 0);
    }

    #[test]
    fn haitsma_envelope() {
        let fp = HaitsmaFingerprint {
            frames: vec![0; 100],
            frames_per_sec: 78.125,
        };
        let env = fp.envelope();
        assert_eq!(env.algorithm, "haitsma-v1");
        assert_eq!(env.sample_rate, 5_000);
        assert_eq!(env.hash_count, 100);
    }

    #[test]
    fn header_layout_is_correct_size() {
        // Verify the HEADER_SIZE constant matches what we write.
        let fp = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
    }

    #[test]
    fn extra_trailing_bytes_are_ignored() {
        // from_bytes should tolerate extra bytes after the payload
        // (forward compatibility for envelope extensions).
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xFF,
                t_anchor: 7,
            }],
            frames_per_sec: 62.5,
        };
        let mut bytes = fp.to_bytes();
        bytes.extend_from_slice(&[0xDE, 0xAD]); // extra junk
        let fp2 = WangFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
    }

    #[test]
    fn peek_reads_header_without_payload() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xAB,
                t_anchor: 3,
            }],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        let env = FingerprintEnvelope::peek(&bytes).unwrap();
        assert_eq!(env.algorithm, "wang-v1");
        assert_eq!(env.sample_rate, 8_000);
        assert_eq!(env.frames_per_sec, 62.5);
        assert_eq!(env.hash_count, 1);
        // peek must work on a header-only prefix: the payload is never
        // touched, so a truncated-tail blob still yields metadata.
        let mut header_only = bytes[..HEADER_SIZE].to_vec();
        header_only.truncate(HEADER_SIZE);
        let env2 = FingerprintEnvelope::peek(&header_only).unwrap();
        assert_eq!(env2.hash_count, 1);
    }

    #[test]
    fn peek_rejects_unknown_algorithm_id() {
        let fp = PanakoFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let mut bytes = fp.to_bytes();
        bytes[9] = 0x7F; // unknown algorithm id
        let err = FingerprintEnvelope::peek(&bytes).unwrap_err();
        assert!(err.to_string().contains("unknown algorithm id"));
    }

    #[test]
    fn peek_rejects_bad_header_like_from_bytes() {
        let mut bytes = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        }
        .to_bytes();
        bytes[0] = b'X';
        assert!(FingerprintEnvelope::peek(&bytes).is_err());
        assert!(FingerprintEnvelope::peek(&[0u8; 4]).is_err());
    }

    // Regression (A2/M3): `peek` documents that it rejects a non-finite /
    // non-positive frame rate, but the check used to live only in
    // `read_payload` (full parse). `peek` on a NaN/zero-fps blob returned
    // garbage metadata. Validation now lives in `read_header`, so `peek`
    // must reject these too.
    #[test]
    fn peek_rejects_non_finite_or_non_positive_fps() {
        for bad_fps in [0.0_f32, -62.5, f32::NAN, f32::INFINITY] {
            let fp = WangFingerprint {
                hashes: vec![],
                frames_per_sec: 62.5,
            };
            let mut bytes = fp.to_bytes();
            // fps is the last 4 header bytes (LE).
            let fps_bytes = bad_fps.to_le_bytes();
            bytes[HEADER_SIZE - 4..HEADER_SIZE].copy_from_slice(&fps_bytes);
            let err = FingerprintEnvelope::peek(&bytes).unwrap_err();
            assert!(
                err.to_string().contains("invalid frame rate"),
                "fps={bad_fps}: {err}"
            );
        }
    }

    #[test]
    fn from_bytes_rejects_non_finite_or_non_positive_fps() {
        for bad_fps in [0.0_f32, -62.5, f32::NAN, f32::INFINITY] {
            let fp = WangFingerprint {
                hashes: vec![],
                frames_per_sec: 62.5,
            };
            let mut bytes = fp.to_bytes();
            // fps is the last 4 header bytes (LE).
            let fps_bytes = bad_fps.to_le_bytes();
            bytes[HEADER_SIZE - 4..HEADER_SIZE].copy_from_slice(&fps_bytes);
            let err = WangFingerprint::from_bytes(&bytes).unwrap_err();
            assert!(
                err.to_string().contains("invalid frame rate"),
                "fps={bad_fps}: {err}"
            );
        }
    }
}
