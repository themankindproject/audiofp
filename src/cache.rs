//! File caching for fingerprints (`.afp` files).
//!
//! A `.afp` file is byte-for-byte the v1 serialization blob from
//! [`serial`](crate::serial) — `to_bytes()` written to disk. This module
//! adds only the file I/O: single-file read/write plus a bulk directory
//! scan for the *parallel extract → serial ingest* workflow (issue #119).
//!
//! # Workflow
//!
//! ```rust
//! use audiofp::cache::{CacheableFingerprint, cache_to_file, load_from_cache};
//! use audiofp::classical::Wang;
//! use audiofp::{Fingerprinter, SampleRate};
//!
//! # fn main() -> audiofp::Result<()> {
//! // 1. Parallel extraction (any rayon setup) writes .afp files:
//! let samples = vec![0.0_f32; 8_000 * 3];
//! let mut wang = Wang::default();
//! let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
//!
//! let path = std::env::temp_dir().join("audiofp_doc_cache.afp");
//! cache_to_file(&fp, &path)?;
//!
//! // 2. Serial ingest reads them back:
//! let restored: Wang::Output = load_from_cache(&path)?;
//! assert_eq!(restored.hashes, fp.hashes);
//! assert_eq!(restored.frames_per_sec, fp.frames_per_sec);
//! # std::fs::remove_file(&path).ok();
//! # Ok(())
//! # }
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, WangFingerprint};
use crate::{AfpError, Result};

/// Fingerprint file extension (without the leading dot).
pub const AFP_EXT: &str = "afp";

/// A fingerprint loaded from an arbitrary `.afp` blob.
///
/// [`load_all_cached`](self::load_all_cached) returns this enum because a
/// cache directory may mix algorithms.
#[derive(Clone, Debug)]
pub enum CachedFingerprint {
    /// A `wang-v1` fingerprint.
    Wang(WangFingerprint),
    /// A `panako-v2` fingerprint.
    Panako(PanakoFingerprint),
    /// A `haitsma-v1` fingerprint.
    Haitsma(HaitsmaFingerprint),
}

// The fingerprint structs themselves derive only `Clone, Debug`, so the
// enum can't derive `PartialEq`; compare field-wise instead (same shape
// as the `serial` round-trip tests). No `Eq` because of the `f32` frame
// rate.
impl PartialEq for CachedFingerprint {
    fn eq(&self, other: &Self) -> bool {
        use CachedFingerprint::{Haitsma, Panako, Wang};
        match (self, other) {
            (Wang(a), Wang(b)) => a.hashes == b.hashes && a.frames_per_sec == b.frames_per_sec,
            (Panako(a), Panako(b)) => a.hashes == b.hashes && a.frames_per_sec == b.frames_per_sec,
            (Haitsma(a), Haitsma(b)) => {
                a.frames == b.frames && a.frames_per_sec == b.frames_per_sec
            }
            _ => false,
        }
    }
}

impl CachedFingerprint {
    /// Parse any `.afp` blob, choosing the variant by header algorithm id.
    ///
    /// # Errors
    ///
    /// `AfpError::Deserialize` if the blob's algorithm id is unknown or the
    /// payload fails validation for its declared algorithm.
    pub fn from_blob(bytes: &[u8]) -> Result<Self> {
        // peek() validates magic/version/fps and resolves the algorithm.
        let env = crate::serial::FingerprintEnvelope::peek(bytes)?;
        match env.algorithm {
            "wang-v1" => Ok(Self::Wang(WangFingerprint::from_bytes(bytes)?)),
            "panako-v2" => Ok(Self::Panako(PanakoFingerprint::from_bytes(bytes)?)),
            "haitsma-v1" => Ok(Self::Haitsma(HaitsmaFingerprint::from_bytes(bytes)?)),
            other => Err(AfpError::Deserialize(format!(
                "unknown algorithm tag: {other}"
            ))),
        }
    }

    /// The envelope of the cached fingerprint.
    #[must_use]
    pub fn envelope(&self) -> crate::serial::FingerprintEnvelope {
        match self {
            Self::Wang(fp) => fp.envelope(),
            Self::Panako(fp) => fp.envelope(),
            Self::Haitsma(fp) => fp.envelope(),
        }
    }
}

/// A fingerprint type that can be persisted to a `.afp` file.
pub trait CacheableFingerprint: Sized {
    /// Serialize to the v1 blob (see [`serial`](crate::serial)).
    fn to_cache_bytes(&self) -> Vec<u8>;
    /// Parse from the v1 blob.
    ///
    /// # Errors
    ///
    /// `AfpError::Deserialize` on any blob defect (inherited from
    /// `from_bytes`).
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self>;
}

impl CacheableFingerprint for WangFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

impl CacheableFingerprint for PanakoFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

impl CacheableFingerprint for HaitsmaFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
}

/// Write a fingerprint to a `.afp` cache file (the v1 blob).
///
/// Parent directories are **not** created (caller's job) — matching the
/// std-lib `fs::write` contract. Overwrites any existing file.
///
/// # Errors
///
/// `AfpError::Io` with the path attached on any filesystem failure.
pub fn cache_to_file<T: CacheableFingerprint>(fp: &T, path: &Path) -> Result<()> {
    fs::write(path, fp.to_cache_bytes()).map_err(|e| AfpError::io_with_path(path, e))
}

/// Load a fingerprint from a `.afp` cache file.
///
/// # Errors
///
/// - `AfpError::Io` if the file cannot be read.
/// - `AfpError::Deserialize` if the contents are not a valid v1 blob for `T`.
pub fn load_from_cache<T: CacheableFingerprint>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| AfpError::io_with_path(path, e))?;
    T::from_cache_bytes(&bytes)
}

/// Load every `*.afp` file in a directory (non-recursive).
///
/// Files with other extensions (and subdirectories) are ignored. Entries
/// are sorted by path for deterministic ingest order. An empty directory
/// yields `Ok(vec![])`.
///
/// **Fails on the first invalid `.afp` file** (error carries the path via
/// [`AfpError::Io`] or a `Deserialize` message naming it) — bulk ingest
/// must not silently drop catalog entries. To skip bad files, iterate
/// `fs::read_dir` and call [`load_from_cache`] per entry yourself.
///
/// # Errors
///
/// `AfpError::Io` if the directory cannot be read or any `.afp` file
/// fails to load/parse.
pub fn load_all_cached(dir: &Path) -> Result<Vec<(PathBuf, CachedFingerprint)>> {
    let entries = fs::read_dir(dir).map_err(|e| AfpError::io_with_path(dir.to_path_buf(), e))?;
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| AfpError::io_with_path(dir.to_path_buf(), e))?;
        let path = entry.path();
        if path.is_dir() {
            continue;
        }
        let is_afp = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("afp"));
        if is_afp {
            paths.push(path);
        }
    }
    paths.sort();
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(&path).map_err(|e| AfpError::io_with_path(&path, e))?;
        let fp = CachedFingerprint::from_blob(&bytes).map_err(|e| match e {
            AfpError::Deserialize(msg) => {
                AfpError::Deserialize(format!("{}: {msg}", path.display()))
            }
            other => other,
        })?;
        out.push((path, fp));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{WangFingerprint, WangHash};

    /// Unique temp dir per test, cleaned up on drop.
    struct TempDir(std::path::PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "audiofp_cache_test_{tag}_{}_{}_{n}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0),
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn roundtrip_wang_cache_file() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xDEAD_BEEF,
                t_anchor: 42,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("roundtrip");
        let path = dir.0.join("t.afp");
        cache_to_file(&fp, &path).unwrap();
        let restored: WangFingerprint = load_from_cache(&path).unwrap();
        assert_eq!(restored.hashes, fp.hashes);
        assert_eq!(restored.frames_per_sec, fp.frames_per_sec);
    }
}
