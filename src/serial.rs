//! Lightweight binary serialization for fingerprint types.
//!
//! Each fingerprint can be round-tripped through a compact binary format
//! via [`to_bytes`] / [`from_bytes`], and metadata about a fingerprint
//! (without parsing the hash payload) is available through [`envelope`].
//!
//! # Wire format (v2)
//!
//! ```text
//! [magic: 8 bytes "AUDIOFP\0"] [version: u8 = 2] [algorithm_id: u8]
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

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use bytemuck::cast_slice;

use crate::classical::{
    HaitsmaFingerprint, PanakoFingerprint, PanakoHash, WangFingerprint, WangHash,
};
use crate::{AfpError, Result};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic header identifying an `audiofp` binary fingerprint blob.
const MAGIC: [u8; 8] = *b"AUDIOFP\0";

/// Current serialization format version.
///
/// v2 (0.4.0): `WangHash` timestamps became `TimestampMs` (8 → 12
/// bytes/hash), `PanakoHash` likewise (16 → 28 bytes/hash). v1 blobs
/// are rejected with [`AfpError::UnsupportedVersion`].
const FORMAT_VERSION: u8 = 2;

/// Fixed-size header: magic (8) + version (1) + algorithm_id (1) +
/// hash_count (4) + fps (4) = 18 bytes.
const HEADER_SIZE: usize = 8 + 1 + 1 + 4 + 4;

// Algorithm IDs.
const ALG_WANG: u8 = 0;
const ALG_PANAKO: u8 = 1;
const ALG_HAITSMA: u8 = 2;

// ---------------------------------------------------------------------------
// FingerprintEnvelope
// ---------------------------------------------------------------------------

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
    /// Crate version that produced the fingerprint.
    pub crate_version: &'static str,
    /// Sample rate the algorithm expects (Hz).
    pub sample_rate: u32,
    /// STFT frame rate (frames per second).
    pub frames_per_sec: f32,
    /// Number of hashes (or frames, for Haitsma) in the fingerprint.
    pub hash_count: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Write the fixed header into a pre-allocated `Vec<u8>`.
fn write_header(buf: &mut Vec<u8>, alg_id: u8, hash_count: u32, fps: f32) {
    buf.extend_from_slice(&MAGIC);
    buf.push(FORMAT_VERSION);
    buf.push(alg_id);
    buf.extend_from_slice(&hash_count.to_le_bytes());
    buf.extend_from_slice(&fps.to_le_bytes());
}

/// Parse and validate the fixed header, returning `(algorithm_id, hash_count, fps)`.
fn read_header(bytes: &[u8], expected_alg: u8) -> Result<(u8, u32, f32)> {
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
    if alg_id != expected_alg {
        return Err(AfpError::Deserialize(format!(
            "algorithm mismatch: blob has id {alg_id}, expected {expected_alg}"
        )));
    }
    let hash_count = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
    let fps = f32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
    Ok((alg_id, hash_count, fps))
}

/// Read a byte slice into a `Vec<T>` where `T: Pod`.
///
/// This handles potentially-unaligned input by allocating a fresh `Vec<T>`
/// (which is always properly aligned) and copying the raw bytes into it.
/// `src` must be an exact multiple of `size_of::<T>()`.
fn read_pod_vec<T: bytemuck::Pod>(src: &[u8]) -> Vec<T> {
    let elem_size = core::mem::size_of::<T>();
    let count = if elem_size == 0 {
        0
    } else {
        src.len() / elem_size
    };
    // Allocate a zeroed vec (Pod guarantees all-zeros is valid).
    let mut vec: Vec<T> = alloc::vec![T::zeroed(); count];
    if count > 0 {
        let dst: &mut [u8] = bytemuck::cast_slice_mut(&mut vec);
        dst.copy_from_slice(src);
    }
    vec
}

// ---------------------------------------------------------------------------
// WangFingerprint serialization
// ---------------------------------------------------------------------------

impl WangFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    ///
    /// The format is documented in the [`serial`](crate::serial) module.
    pub fn to_bytes(&self) -> Vec<u8> {
        let hash_bytes: &[u8] = cast_slice(&self.hashes);
        let mut buf = Vec::with_capacity(HEADER_SIZE + hash_bytes.len());
        write_header(
            &mut buf,
            ALG_WANG,
            self.hashes.len() as u32,
            self.frames_per_sec,
        );
        buf.extend_from_slice(hash_bytes);
        buf
    }

    /// Deserialize a Wang fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_alg, hash_count, fps) = read_header(bytes, ALG_WANG)?;
        let payload = &bytes[HEADER_SIZE..];
        let expected_len = (hash_count as usize) * core::mem::size_of::<WangHash>();
        if payload.len() < expected_len {
            return Err(AfpError::Deserialize(format!(
                "payload too short: need {} bytes for {} hashes, got {}",
                expected_len,
                hash_count,
                payload.len()
            )));
        }
        let hashes = read_pod_vec::<WangHash>(&payload[..expected_len]);
        Ok(Self {
            hashes,
            frames_per_sec: fps,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        FingerprintEnvelope {
            algorithm: "wang-v1",
            crate_version: crate::VERSION,
            sample_rate: 8_000,
            frames_per_sec: self.frames_per_sec,
            hash_count: self.hashes.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// PanakoFingerprint serialization
// ---------------------------------------------------------------------------

impl PanakoFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let hash_bytes: &[u8] = cast_slice(&self.hashes);
        let mut buf = Vec::with_capacity(HEADER_SIZE + hash_bytes.len());
        write_header(
            &mut buf,
            ALG_PANAKO,
            self.hashes.len() as u32,
            self.frames_per_sec,
        );
        buf.extend_from_slice(hash_bytes);
        buf
    }

    /// Deserialize a Panako fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_alg, hash_count, fps) = read_header(bytes, ALG_PANAKO)?;
        let payload = &bytes[HEADER_SIZE..];
        let expected_len = (hash_count as usize) * core::mem::size_of::<PanakoHash>();
        if payload.len() < expected_len {
            return Err(AfpError::Deserialize(format!(
                "payload too short: need {} bytes for {} hashes, got {}",
                expected_len,
                hash_count,
                payload.len()
            )));
        }
        let hashes = read_pod_vec::<PanakoHash>(&payload[..expected_len]);
        Ok(Self {
            hashes,
            frames_per_sec: fps,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        FingerprintEnvelope {
            algorithm: "panako-v2",
            crate_version: crate::VERSION,
            sample_rate: 8_000,
            frames_per_sec: self.frames_per_sec,
            hash_count: self.hashes.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// HaitsmaFingerprint serialization
// ---------------------------------------------------------------------------

impl HaitsmaFingerprint {
    /// Serialize this fingerprint to a compact binary blob.
    pub fn to_bytes(&self) -> Vec<u8> {
        let frame_bytes: &[u8] = cast_slice(&self.frames);
        let mut buf = Vec::with_capacity(HEADER_SIZE + frame_bytes.len());
        write_header(
            &mut buf,
            ALG_HAITSMA,
            self.frames.len() as u32,
            self.frames_per_sec,
        );
        buf.extend_from_slice(frame_bytes);
        buf
    }

    /// Deserialize a Haitsma fingerprint from a binary blob produced by
    /// [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_alg, hash_count, fps) = read_header(bytes, ALG_HAITSMA)?;
        let payload = &bytes[HEADER_SIZE..];
        let expected_len = (hash_count as usize) * core::mem::size_of::<u32>();
        if payload.len() < expected_len {
            return Err(AfpError::Deserialize(format!(
                "payload too short: need {} bytes for {} frames, got {}",
                expected_len,
                hash_count,
                payload.len()
            )));
        }
        let frames = read_pod_vec::<u32>(&payload[..expected_len]);
        Ok(Self {
            frames,
            frames_per_sec: fps,
        })
    }

    /// Return a metadata envelope describing this fingerprint.
    pub fn envelope(&self) -> FingerprintEnvelope {
        FingerprintEnvelope {
            algorithm: "haitsma-v1",
            crate_version: crate::VERSION,
            sample_rate: 5_000,
            frames_per_sec: self.frames_per_sec,
            hash_count: self.frames.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimestampMs;
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
                    t_anchor: TimestampMs(42),
                },
                WangHash {
                    hash: 0xCAFE_BABE,
                    t_anchor: TimestampMs(100),
                },
            ],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 2 * 12); // 2 hashes × 12 bytes
        let fp2 = WangFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
        assert_eq!(fp.frames_per_sec, fp2.frames_per_sec);
    }

    #[test]
    fn panako_round_trip() {
        let fp = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 0x1234_5678,
                t_anchor: TimestampMs(10),
                t_b: TimestampMs(15),
                t_c: TimestampMs(20),
            }],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE + 28); // 1 hash × 28 bytes
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
                t_anchor: TimestampMs(2),
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
                    t_anchor: TimestampMs(0),
                },
                WangHash {
                    hash: 2,
                    t_anchor: TimestampMs(1),
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
                t_anchor: TimestampMs(7),
            }],
            frames_per_sec: 62.5,
        };
        let mut bytes = fp.to_bytes();
        bytes.extend_from_slice(&[0xDE, 0xAD]); // extra junk
        let fp2 = WangFingerprint::from_bytes(&bytes).unwrap();
        assert_eq!(fp.hashes, fp2.hashes);
    }
}
