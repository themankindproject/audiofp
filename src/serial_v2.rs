//! Opt-in fingerprint serialization with extractor configuration provenance.
//!
//! The original [`serial`](crate::serial) v1 methods and bytes remain intact.
//! This format uses little-endian fields throughout and records a stable
//! 128-bit ID for the extraction settings that affect fingerprint content.
//! The producer crate version is stored for diagnostics.
//!
//! Pass the config used to build the extractor to `to_bytes_with_config`;
//! fingerprints themselves do not retain their extractor config. Output
//! truncation caps are part of the ID, while input rejection caps are not.
//!
//! ```
//! use audiofp::classical::{WangConfig, WangFingerprint};
//!
//! let cfg = WangConfig::default();
//! let fp = WangFingerprint { hashes: Vec::new(), frames_per_sec: 62.5 };
//! let bytes = fp.to_bytes_with_config(&cfg).unwrap();
//! let restored = WangFingerprint::from_bytes_with_config(&bytes, &cfg).unwrap();
//! assert!(restored.hashes.is_empty());
//! ```

use alloc::format;
use alloc::vec::Vec;

use crate::classical::{
    HaitsmaConfig, HaitsmaFingerprint, PanakoConfig, PanakoFingerprint, PanakoHash, WangConfig,
    WangFingerprint, WangHash,
};
use crate::{AfpError, Result};

const MAGIC: [u8; 8] = *b"AUDIOFP\0";
const VERSION: u8 = 2;
const HEADER: usize = 40;
const WANG: u8 = 0;
const PANAKO: u8 = 1;
const HAITSMA: u8 = 2;
const FNV_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

fn hash_part(hash: &mut u128, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u128::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_optional_limit(hash: &mut u128, limit: Option<usize>) {
    match limit {
        None => hash_part(hash, &[0]),
        Some(value) => {
            hash_part(hash, &[1]);
            hash_part(hash, &(value as u64).to_le_bytes());
        }
    }
}

/// Metadata persisted in a v2 fingerprint blob.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FingerprintProvenance {
    /// Stable algorithm and hash-layout identifier.
    pub algorithm: &'static str,
    /// ID of the settings that affect fingerprint bytes.
    pub config_id: u128,
    /// Producer crate version as `(major, minor, patch)`.
    pub producer_version: (u16, u16, u16),
    /// Fingerprint frames per second.
    pub frames_per_sec: f32,
    /// Number of hashes or frames.
    pub hash_count: u32,
}

impl FingerprintProvenance {
    /// Inspect a v2 header without reading the payload.
    pub fn peek(bytes: &[u8]) -> Result<Self> {
        let (_, metadata) = read_header(bytes)?;
        Ok(metadata)
    }
}

impl WangConfig {
    /// Stable ID of Wang settings that affect produced hash bytes.
    #[must_use]
    pub fn compatibility_id(&self) -> u128 {
        let mut cfg = self.clone();
        crate::classical::sanitize_cfg!(cfg);
        let mut hash = FNV_OFFSET;
        hash_part(&mut hash, b"wang-v1");
        hash_part(&mut hash, &cfg.fan_out.to_le_bytes());
        hash_part(&mut hash, &cfg.target_zone_t.to_le_bytes());
        hash_part(&mut hash, &cfg.target_zone_f.to_le_bytes());
        hash_part(&mut hash, &cfg.peaks_per_sec.to_le_bytes());
        hash_part(&mut hash, &cfg.min_anchor_mag_db.to_bits().to_le_bytes());
        hash_optional_limit(&mut hash, cfg.max_hashes);
        hash_optional_limit(&mut hash, cfg.max_pending_anchors);
        hash_optional_limit(&mut hash, cfg.max_push_samples);
        hash
    }
}

impl PanakoConfig {
    /// Stable ID of Panako settings that affect produced hash bytes.
    #[must_use]
    pub fn compatibility_id(&self) -> u128 {
        let mut cfg = self.clone();
        crate::classical::sanitize_cfg!(cfg);
        let mut hash = FNV_OFFSET;
        hash_part(&mut hash, b"panako-v2");
        hash_part(&mut hash, &cfg.fan_out.to_le_bytes());
        hash_part(&mut hash, &cfg.target_zone_t.to_le_bytes());
        hash_part(&mut hash, &cfg.target_zone_f.to_le_bytes());
        hash_part(&mut hash, &cfg.peaks_per_sec.to_le_bytes());
        hash_part(&mut hash, &cfg.min_anchor_mag_db.to_bits().to_le_bytes());
        hash_optional_limit(&mut hash, cfg.max_hashes);
        hash_optional_limit(&mut hash, cfg.max_pending_anchors);
        hash_optional_limit(&mut hash, cfg.max_push_samples);
        hash
    }
}

impl HaitsmaConfig {
    /// Stable ID of Haitsma band settings that affect produced frame bits.
    #[must_use]
    pub fn compatibility_id(&self) -> u128 {
        let mut hash = FNV_OFFSET;
        hash_part(&mut hash, b"haitsma-v1");
        hash_part(&mut hash, &self.fmin.to_bits().to_le_bytes());
        hash_part(&mut hash, &self.fmax.to_bits().to_le_bytes());
        hash_optional_limit(&mut hash, self.max_push_samples);
        hash
    }
}

fn producer_version() -> (u16, u16, u16) {
    (
        env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0),
    )
}

fn new_blob(count: usize, element_size: usize) -> Result<Vec<u8>> {
    u32::try_from(count)
        .map_err(|_| AfpError::Config("fingerprint hash count exceeds u32::MAX".into()))?;
    let capacity = count
        .checked_mul(element_size)
        .and_then(|bytes| bytes.checked_add(HEADER))
        .ok_or_else(|| AfpError::Config("fingerprint payload length overflow".into()))?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|_| AfpError::Config("fingerprint payload allocation failed".into()))?;
    Ok(out)
}

fn write_header(
    out: &mut Vec<u8>,
    algorithm: u8,
    count: usize,
    fps: f32,
    config_id: u128,
) -> Result<()> {
    let count = u32::try_from(count)
        .map_err(|_| AfpError::Config("fingerprint hash count exceeds u32::MAX".into()))?;
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(algorithm);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&fps.to_le_bytes());
    out.extend_from_slice(&config_id.to_le_bytes());
    let (major, minor, patch) = producer_version();
    out.extend_from_slice(&major.to_le_bytes());
    out.extend_from_slice(&minor.to_le_bytes());
    out.extend_from_slice(&patch.to_le_bytes());
    Ok(())
}

fn read_header(bytes: &[u8]) -> Result<(u8, FingerprintProvenance)> {
    if bytes.len() < HEADER {
        return Err(AfpError::Deserialize("v2 header too short".into()));
    }
    if bytes[..8] != MAGIC {
        return Err(AfpError::Deserialize("bad fingerprint magic".into()));
    }
    if bytes[8] != VERSION {
        return Err(AfpError::Deserialize(format!(
            "expected fingerprint format v2, got v{}",
            bytes[8]
        )));
    }
    let algorithm = match bytes[9] {
        WANG => "wang-v1",
        PANAKO => "panako-v2",
        HAITSMA => "haitsma-v1",
        id => return Err(AfpError::Deserialize(format!("unknown algorithm id: {id}"))),
    };
    let count = u32::from_le_bytes(bytes[10..14].try_into().expect("fixed header"));
    let fps = f32::from_le_bytes(bytes[14..18].try_into().expect("fixed header"));
    if !fps.is_finite() || fps <= 0.0 {
        return Err(AfpError::Deserialize("invalid frame rate".into()));
    }
    let config_id = u128::from_le_bytes(bytes[18..34].try_into().expect("fixed header"));
    let major = u16::from_le_bytes(bytes[34..36].try_into().expect("fixed header"));
    let minor = u16::from_le_bytes(bytes[36..38].try_into().expect("fixed header"));
    let patch = u16::from_le_bytes(bytes[38..40].try_into().expect("fixed header"));
    Ok((
        bytes[9],
        FingerprintProvenance {
            algorithm,
            config_id,
            producer_version: (major, minor, patch),
            frames_per_sec: fps,
            hash_count: count,
        },
    ))
}

fn checked_payload(
    bytes: &[u8],
    algorithm: u8,
    config_id: u128,
    fps: f32,
    element_size: usize,
) -> Result<(&[u8], usize)> {
    let (actual_algorithm, metadata) = read_header(bytes)?;
    if actual_algorithm != algorithm {
        return Err(AfpError::Deserialize(
            "fingerprint algorithm mismatch".into(),
        ));
    }
    if metadata.config_id != config_id {
        return Err(AfpError::Deserialize(
            "fingerprint configuration mismatch".into(),
        ));
    }
    if metadata.frames_per_sec != fps {
        return Err(AfpError::Deserialize("noncanonical frame rate".into()));
    }
    let count = metadata.hash_count as usize;
    let size = count
        .checked_mul(element_size)
        .ok_or_else(|| AfpError::Deserialize("fingerprint payload length overflow".into()))?;
    if bytes.len() - HEADER < size {
        return Err(AfpError::Deserialize(
            "fingerprint payload too short".into(),
        ));
    }
    Ok((&bytes[HEADER..HEADER + size], count))
}

impl WangFingerprint {
    /// Write portable v2 bytes tied to the supplied extractor settings.
    pub fn to_bytes_with_config(&self, cfg: &WangConfig) -> Result<Vec<u8>> {
        if self.frames_per_sec != 62.5 {
            return Err(AfpError::Config("noncanonical Wang frame rate".into()));
        }
        let mut out = new_blob(self.hashes.len(), 8)?;
        write_header(
            &mut out,
            WANG,
            self.hashes.len(),
            self.frames_per_sec,
            cfg.compatibility_id(),
        )?;
        for h in &self.hashes {
            out.extend_from_slice(&h.hash.to_le_bytes());
            out.extend_from_slice(&h.t_anchor.to_le_bytes());
        }
        Ok(out)
    }

    /// Read portable v2 bytes only when their extractor settings match.
    pub fn from_bytes_with_config(bytes: &[u8], cfg: &WangConfig) -> Result<Self> {
        let (payload, count) = checked_payload(bytes, WANG, cfg.compatibility_id(), 62.5, 8)?;
        let mut hashes = Vec::with_capacity(count);
        for row in payload.chunks_exact(8) {
            hashes.push(WangHash {
                hash: u32::from_le_bytes(row[..4].try_into().expect("fixed row")),
                t_anchor: u32::from_le_bytes(row[4..8].try_into().expect("fixed row")),
            });
        }
        Ok(Self {
            hashes,
            frames_per_sec: 62.5,
        })
    }
}

impl PanakoFingerprint {
    /// Write portable v2 bytes tied to the supplied extractor settings.
    pub fn to_bytes_with_config(&self, cfg: &PanakoConfig) -> Result<Vec<u8>> {
        if self.frames_per_sec != 62.5 {
            return Err(AfpError::Config("noncanonical Panako frame rate".into()));
        }
        let mut out = new_blob(self.hashes.len(), 16)?;
        write_header(
            &mut out,
            PANAKO,
            self.hashes.len(),
            self.frames_per_sec,
            cfg.compatibility_id(),
        )?;
        for h in &self.hashes {
            out.extend_from_slice(&h.hash.to_le_bytes());
            out.extend_from_slice(&h.t_anchor.to_le_bytes());
            out.extend_from_slice(&h.t_b.to_le_bytes());
            out.extend_from_slice(&h.t_c.to_le_bytes());
        }
        Ok(out)
    }

    /// Read portable v2 bytes only when their extractor settings match.
    pub fn from_bytes_with_config(bytes: &[u8], cfg: &PanakoConfig) -> Result<Self> {
        let (payload, count) = checked_payload(bytes, PANAKO, cfg.compatibility_id(), 62.5, 16)?;
        let mut hashes = Vec::with_capacity(count);
        for row in payload.chunks_exact(16) {
            hashes.push(PanakoHash {
                hash: u32::from_le_bytes(row[..4].try_into().expect("fixed row")),
                t_anchor: u32::from_le_bytes(row[4..8].try_into().expect("fixed row")),
                t_b: u32::from_le_bytes(row[8..12].try_into().expect("fixed row")),
                t_c: u32::from_le_bytes(row[12..16].try_into().expect("fixed row")),
            });
        }
        Ok(Self {
            hashes,
            frames_per_sec: 62.5,
        })
    }
}

impl HaitsmaFingerprint {
    /// Write portable v2 bytes tied to the supplied extractor settings.
    pub fn to_bytes_with_config(&self, cfg: &HaitsmaConfig) -> Result<Vec<u8>> {
        if self.frames_per_sec != 78.125 {
            return Err(AfpError::Config("noncanonical Haitsma frame rate".into()));
        }
        let mut out = new_blob(self.frames.len(), 4)?;
        write_header(
            &mut out,
            HAITSMA,
            self.frames.len(),
            self.frames_per_sec,
            cfg.compatibility_id(),
        )?;
        for &frame in &self.frames {
            out.extend_from_slice(&frame.to_le_bytes());
        }
        Ok(out)
    }

    /// Read portable v2 bytes only when their extractor settings match.
    pub fn from_bytes_with_config(bytes: &[u8], cfg: &HaitsmaConfig) -> Result<Self> {
        let (payload, count) = checked_payload(bytes, HAITSMA, cfg.compatibility_id(), 78.125, 4)?;
        let mut frames = Vec::with_capacity(count);
        for row in payload.chunks_exact(4) {
            frames.push(u32::from_le_bytes(row.try_into().expect("fixed row")));
        }
        Ok(Self {
            frames,
            frames_per_sec: 78.125,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn v2_round_trip_and_config_guard() {
        let cfg = WangConfig::default();
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x1234_5678,
                t_anchor: 42,
            }],
            frames_per_sec: 62.5,
        };
        let bytes = fp.to_bytes_with_config(&cfg).unwrap();
        assert_eq!(
            &bytes[HEADER..HEADER + 8],
            &[0x78, 0x56, 0x34, 0x12, 42, 0, 0, 0]
        );
        let restored = WangFingerprint::from_bytes_with_config(&bytes, &cfg).unwrap();
        assert_eq!(restored.hashes, fp.hashes);
        assert_eq!(
            FingerprintProvenance::peek(&bytes).unwrap().config_id,
            cfg.compatibility_id()
        );
        let changed = WangConfig {
            fan_out: cfg.fan_out + 1,
            ..cfg.clone()
        };
        assert!(matches!(
            WangFingerprint::from_bytes_with_config(&bytes, &changed),
            Err(AfpError::Deserialize(_))
        ));
        let changed_cap = WangConfig {
            max_hashes: Some(10),
            ..WangConfig::default()
        };
        assert_ne!(cfg.compatibility_id(), changed_cap.compatibility_id());
        let changed_reject_only = WangConfig {
            max_input_samples: None,
            ..WangConfig::default()
        };
        assert_eq!(
            cfg.compatibility_id(),
            changed_reject_only.compatibility_id()
        );
    }

    #[test]
    fn all_algorithms_round_trip_without_changing_v1_bytes() {
        let panako = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 1,
                t_anchor: 2,
                t_b: 3,
                t_c: 4,
            }],
            frames_per_sec: 62.5,
        };
        let p_cfg = PanakoConfig::default();
        let p_v1 = panako.to_bytes();
        let p_v2 = panako.to_bytes_with_config(&p_cfg).unwrap();
        assert_eq!(
            PanakoFingerprint::from_bytes_with_config(&p_v2, &p_cfg)
                .unwrap()
                .hashes,
            panako.hashes
        );
        assert_eq!(panako.to_bytes(), p_v1);

        let haitsma = HaitsmaFingerprint {
            frames: vec![0xAABB_CCDD, 7],
            frames_per_sec: 78.125,
        };
        let h_cfg = HaitsmaConfig::default();
        let h_v1 = haitsma.to_bytes();
        let h_v2 = haitsma.to_bytes_with_config(&h_cfg).unwrap();
        assert_eq!(
            HaitsmaFingerprint::from_bytes_with_config(&h_v2, &h_cfg)
                .unwrap()
                .frames,
            haitsma.frames
        );
        assert_eq!(haitsma.to_bytes(), h_v1);
    }
}
