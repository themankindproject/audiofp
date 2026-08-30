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
//! let restored: <Wang as Fingerprinter>::Output =
//!     load_from_cache(&path)?;
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
/// [`load_all_cached`] returns this enum because a
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
            // Unreachable today: `peek` validates the algorithm id against
            // the known table first. Defensive: a future `serial` version
            // adding an algorithm must not be misparsed here.
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
        // `from_blob` yields `Deserialize` on every failure path today
        // (header + payload validation); name the offending file. The
        // `other` arm is defensive — it cannot fire while `from_blob`
        // only constructs `Deserialize`, but if a future error variant
        // is added there it must still propagate with its own context.
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

    #[test]
    fn roundtrip_panako_and_haitsma_cache_files() {
        use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, PanakoHash};

        let panako = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 0x1234_5678,
                t_anchor: 10,
                t_b: 15,
                t_c: 20,
            }],
            frames_per_sec: 62.5,
        };
        let haitsma = HaitsmaFingerprint {
            frames: vec![0xAAAA_BBBB, 0x1111_2222],
            frames_per_sec: 78.125,
        };

        let dir = TempDir::new("roundtrip_pa");
        let p_path = dir.0.join("p.afp");
        let h_path = dir.0.join("h.afp");
        cache_to_file(&panako, &p_path).unwrap();
        cache_to_file(&haitsma, &h_path).unwrap();

        let p_restored: PanakoFingerprint = load_from_cache(&p_path).unwrap();
        assert_eq!(p_restored.hashes, panako.hashes);
        assert_eq!(p_restored.frames_per_sec, panako.frames_per_sec);

        let h_restored: HaitsmaFingerprint = load_from_cache(&h_path).unwrap();
        assert_eq!(h_restored.frames, haitsma.frames);
        assert_eq!(h_restored.frames_per_sec, haitsma.frames_per_sec);
    }

    #[test]
    fn load_all_cached_sorts_and_ignores_other_extensions() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("scan");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();
        cache_to_file(&fp, &dir.0.join("c.afp")).unwrap();
        std::fs::write(dir.0.join("notes.txt"), "not a fingerprint").unwrap();
        // Subdirectory with an .afp inside — must be skipped (non-recursive).
        let sub = dir.0.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        cache_to_file(&fp, &sub.join("b.afp")).unwrap();

        let loaded = load_all_cached(&dir.0).unwrap();
        let names: Vec<&str> = loaded
            .iter()
            .map(|(p, _)| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["a.afp", "c.afp"]);
        // Sortedness (not a fixed global order — Windows sorts differently).
        let sorted = {
            let mut v = loaded.clone();
            v.sort_by(|a, b| a.0.cmp(&b.0));
            v
        };
        assert_eq!(loaded, sorted);
        // Every entry is the fingerprint we cached.
        for (_, cached) in &loaded {
            assert_eq!(cached, &CachedFingerprint::Wang(fp.clone()));
        }
    }

    #[test]
    fn load_all_cached_empty_dir_is_ok() {
        let dir = TempDir::new("empty");
        let loaded = load_all_cached(&dir.0).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn load_all_cached_missing_dir_is_io_error() {
        let missing = TempDir::new("missing").0.join("does_not_exist");
        let err = load_all_cached(&missing).unwrap_err();
        assert!(err.to_string().contains("does_not_exist"), "got: {err}");
    }

    #[test]
    fn load_all_cached_corrupt_file_fails_with_path_in_error() {
        let fp = WangFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("corrupt");
        cache_to_file(&fp, &dir.0.join("good.afp")).unwrap();
        std::fs::write(dir.0.join("bad.afp"), b"garbage bytes, no magic").unwrap();

        let err = load_all_cached(&dir.0).unwrap_err();
        assert!(
            err.to_string().contains("bad.afp"),
            "error must name the corrupt file: {err}"
        );
    }

    #[test]
    fn load_all_cached_mixed_algorithms() {
        use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, PanakoHash};

        let wang = WangFingerprint {
            hashes: vec![WangHash {
                hash: 9,
                t_anchor: 3,
            }],
            frames_per_sec: 62.5,
        };
        let panako = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 5,
                t_anchor: 1,
                t_b: 2,
                t_c: 3,
            }],
            frames_per_sec: 62.5,
        };
        let haitsma = HaitsmaFingerprint {
            frames: vec![7, 8],
            frames_per_sec: 78.125,
        };

        let dir = TempDir::new("mixed");
        cache_to_file(&wang, &dir.0.join("w.afp")).unwrap();
        cache_to_file(&panako, &dir.0.join("p.afp")).unwrap();
        cache_to_file(&haitsma, &dir.0.join("h.afp")).unwrap();

        let loaded = load_all_cached(&dir.0).unwrap();
        assert_eq!(loaded.len(), 3);
        for (path, cached) in &loaded {
            let name = path.file_name().unwrap().to_str().unwrap();
            match cached {
                CachedFingerprint::Wang(fp) => {
                    assert_eq!(name, "w.afp");
                    assert_eq!(fp.hashes, wang.hashes);
                }
                CachedFingerprint::Panako(fp) => {
                    assert_eq!(name, "p.afp");
                    assert_eq!(fp.hashes, panako.hashes);
                }
                CachedFingerprint::Haitsma(fp) => {
                    assert_eq!(name, "h.afp");
                    assert_eq!(fp.frames, haitsma.frames);
                }
            }
        }
    }

    #[test]
    fn load_from_cache_missing_file_is_io_error_with_path() {
        let dir = TempDir::new("missing_file");
        let missing = dir.0.join("nope.afp");
        let err = load_from_cache::<WangFingerprint>(&missing).unwrap_err();
        assert!(err.to_string().contains("nope.afp"), "got: {err}");
    }

    #[test]
    fn cache_file_bytes_identical_to_to_bytes() {
        // Pins the "no extra wrapper bytes" contract: a .afp file on disk
        // is exactly fp.to_bytes().
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xFF,
                t_anchor: 7,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("identical");
        let path = dir.0.join("bytes.afp");
        cache_to_file(&fp, &path).unwrap();
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk, fp.to_bytes());
    }

    #[test]
    fn cached_fingerprint_partial_eq_per_variant() {
        use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, PanakoHash};

        let panako = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 5,
                t_anchor: 1,
                t_b: 2,
                t_c: 3,
            }],
            frames_per_sec: 62.5,
        };
        let haitsma = HaitsmaFingerprint {
            frames: vec![7, 8],
            frames_per_sec: 78.125,
        };

        // Same-variant equality (Panako and Haitsma arms)…
        assert_eq!(
            CachedFingerprint::Panako(panako.clone()),
            CachedFingerprint::Panako(panako.clone())
        );
        assert_eq!(
            CachedFingerprint::Haitsma(haitsma.clone()),
            CachedFingerprint::Haitsma(haitsma.clone())
        );
        // …and field-wise inequality inside each variant.
        let panako_other = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 6,
                t_anchor: 1,
                t_b: 2,
                t_c: 3,
            }],
            frames_per_sec: 62.5,
        };
        assert_ne!(
            CachedFingerprint::Panako(panako),
            CachedFingerprint::Panako(panako_other)
        );
        let haitsma_other = HaitsmaFingerprint {
            frames: vec![7, 9],
            frames_per_sec: 78.125,
        };
        assert_ne!(
            CachedFingerprint::Haitsma(haitsma.clone()),
            CachedFingerprint::Haitsma(haitsma_other)
        );
        // Cross-variant inequality (the `_ => false` arm).
        assert_ne!(
            CachedFingerprint::Wang(WangFingerprint {
                hashes: vec![],
                frames_per_sec: 62.5
            }),
            CachedFingerprint::Haitsma(haitsma)
        );
    }

    #[test]
    fn envelope_per_variant() {
        use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, PanakoHash};

        let panako = PanakoFingerprint {
            hashes: vec![PanakoHash {
                hash: 5,
                t_anchor: 1,
                t_b: 2,
                t_c: 3,
            }],
            frames_per_sec: 62.5,
        };
        let haitsma = HaitsmaFingerprint {
            frames: vec![7, 8, 9],
            frames_per_sec: 78.125,
        };

        let p_env = CachedFingerprint::Panako(panako).envelope();
        assert_eq!(p_env.algorithm, "panako-v2");
        assert_eq!(p_env.sample_rate, 8_000);
        assert_eq!(p_env.hash_count, 1);
        assert_eq!(p_env.frames_per_sec, 62.5);

        let h_env = CachedFingerprint::Haitsma(haitsma).envelope();
        assert_eq!(h_env.algorithm, "haitsma-v1");
        assert_eq!(h_env.sample_rate, 5_000);
        assert_eq!(h_env.hash_count, 3);
        assert_eq!(h_env.frames_per_sec, 78.125);
    }
}
