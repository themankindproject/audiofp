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

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

use crate::classical::{HaitsmaFingerprint, PanakoFingerprint, WangFingerprint};
use crate::{AfpError, Result};

static ATOMIC_TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Default cap on the size of a `.afp` file that will be read.
///
/// The v1 blob is ~8 bytes per Wang/Panako hash and 4 bytes per Haitsma
/// frame, so 256 MiB is far beyond any realistic fingerprint (tens of
/// millions of hashes). The cap exists because [`load_from_cache`] and
/// [`load_all_cached`] accept caller-supplied paths — without it a
/// corrupt or adversarial file (including a symlink to `/dev/zero`) makes
/// `fs::read` allocate until the process is killed.
pub const MAX_CACHE_FILE_BYTES: u64 = 256 * 1024 * 1024;
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

    fn from_file(path: &Path, file: &mut File, cap: u64) -> Result<Self> {
        checked_cache_len(path, file, cap)?;
        let mut header = [0u8; crate::serial::HEADER_SIZE];
        file.read_exact(&mut header).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                AfpError::Deserialize(format!("{}: short fingerprint header", path.display()))
            } else {
                AfpError::io_with_path(path, e)
            }
        })?;
        let algorithm = crate::serial::FingerprintEnvelope::peek(&header)?.algorithm;
        file.rewind().map_err(|e| AfpError::io_with_path(path, e))?;
        match algorithm {
            "wang-v1" => WangFingerprint::read_cache_from(path, file, cap).map(Self::Wang),
            "panako-v2" => PanakoFingerprint::read_cache_from(path, file, cap).map(Self::Panako),
            "haitsma-v1" => HaitsmaFingerprint::read_cache_from(path, file, cap).map(Self::Haitsma),
            other => Err(AfpError::Deserialize(format!(
                "{}: unknown algorithm tag: {other}",
                path.display()
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

    /// Write v1 bytes to a stream. Built-in fingerprints avoid a full blob
    /// allocation; third-party implementations retain the existing default.
    fn write_cache_to(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        writer.write_all(&self.to_cache_bytes())
    }

    /// Read from a checked file handle. Built-in fingerprints parse in
    /// bounded blocks; third-party implementations retain the v1 default.
    fn read_cache_from(path: &Path, file: &mut File, cap: u64) -> Result<Self> {
        let bytes = read_handle_with_limit(path, file, cap)?;
        Self::from_cache_bytes(&bytes)
    }
}

impl CacheableFingerprint for WangFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
    fn write_cache_to(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::serial::write_pod_blob(
            writer,
            &self.hashes,
            crate::serial::ALG_WANG,
            self.frames_per_sec,
        )
    }
    fn read_cache_from(path: &Path, file: &mut File, cap: u64) -> Result<Self> {
        let (hashes, frames_per_sec) = read_pod_cache(path, file, cap, crate::serial::ALG_WANG)?;
        Ok(Self {
            hashes,
            frames_per_sec,
        })
    }
}

impl CacheableFingerprint for PanakoFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
    fn write_cache_to(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::serial::write_pod_blob(
            writer,
            &self.hashes,
            crate::serial::ALG_PANAKO,
            self.frames_per_sec,
        )
    }
    fn read_cache_from(path: &Path, file: &mut File, cap: u64) -> Result<Self> {
        let (hashes, frames_per_sec) = read_pod_cache(path, file, cap, crate::serial::ALG_PANAKO)?;
        Ok(Self {
            hashes,
            frames_per_sec,
        })
    }
}

impl CacheableFingerprint for HaitsmaFingerprint {
    fn to_cache_bytes(&self) -> Vec<u8> {
        self.to_bytes()
    }
    fn from_cache_bytes(bytes: &[u8]) -> Result<Self> {
        Self::from_bytes(bytes)
    }
    fn write_cache_to(&self, writer: &mut dyn Write) -> std::io::Result<()> {
        crate::serial::write_pod_blob(
            writer,
            &self.frames,
            crate::serial::ALG_HAITSMA,
            self.frames_per_sec,
        )
    }
    fn read_cache_from(path: &Path, file: &mut File, cap: u64) -> Result<Self> {
        let (frames, frames_per_sec) = read_pod_cache(path, file, cap, crate::serial::ALG_HAITSMA)?;
        Ok(Self {
            frames,
            frames_per_sec,
        })
    }
}

/// Optional aggregate limits for directory cache ingestion.
///
/// Per-file size is always capped by [`MAX_CACHE_FILE_BYTES`]; these
/// options bound how many files and how many total on-disk bytes a bulk
/// scan may retain in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheLoadLimits {
    /// Stop after this many `.afp` files. `None` means no file-count cap.
    pub max_files: Option<usize>,
    /// Stop after this many bytes of successfully loaded cache files,
    /// including headers and any trailing extension bytes. `None`
    /// means no aggregate byte cap.
    pub max_total_bytes: Option<u64>,
}

/// Open a cache file for reading without following symlinks.
///
/// On Unix the open uses `O_NOFOLLOW` and `O_NONBLOCK` so a writable
/// parent directory cannot substitute a FIFO/device between validation
/// and read. On Windows the open uses `FILE_FLAG_OPEN_REPARSE_POINT` so
/// final-component reparse points are not followed and are rejected via
/// file attributes. Parent directories must be trusted on every platform.
/// Other platforms use a `symlink_metadata`
/// guard before a bounded read — callers must place cache files in a
/// **trusted parent directory** they control.
fn open_cache_file_for_read(path: &Path) -> Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        use rustix::fs::OFlags;

        let flags = (OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK).bits();
        OpenOptions::new()
            .read(true)
            .custom_flags(flags as i32)
            .open(path)
            .map_err(|e| AfpError::io_with_path(path, e))
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::{MetadataExt, OpenOptionsExt};

        // FILE_FLAG_OPEN_REPARSE_POINT (0x0020_0000): do not follow symlinks.
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;

        let file = OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .map_err(|e| AfpError::io_with_path(path, e))?;
        let attrs = file
            .metadata()
            .map_err(|e| AfpError::io_with_path(path, e))?
            .file_attributes();
        if attrs & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(AfpError::io_with_path(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "not a regular file (reparse point)",
                ),
            ));
        }
        Ok(file)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let meta = fs::symlink_metadata(path).map_err(|e| AfpError::io_with_path(path, e))?;
        if meta.file_type().is_symlink() {
            return Err(AfpError::io_with_path(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "not a regular file (symlink)",
                ),
            ));
        }
        if !meta.is_file() {
            return Err(AfpError::io_with_path(
                path,
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "not a regular file (FIFO or device)",
                ),
            ));
        }
        OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|e| AfpError::io_with_path(path, e))
    }
}

fn read_handle_with_limit(path: &Path, file: &mut File, cap: u64) -> Result<Vec<u8>> {
    checked_cache_len(path, file, cap)?;
    read_limited(path, file, cap)
}

fn checked_cache_len(path: &Path, file: &File, cap: u64) -> Result<u64> {
    let meta = file
        .metadata()
        .map_err(|e| AfpError::io_with_path(path, e))?;
    let file_type = meta.file_type();
    if file_type.is_symlink() {
        return Err(AfpError::io_with_path(
            path,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "not a regular file (symlink)",
            ),
        ));
    }
    if !file_type.is_file() {
        return Err(AfpError::io_with_path(
            path,
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "not a regular file (FIFO or device)",
            ),
        ));
    }
    if meta.len() > cap {
        return Err(AfpError::InputTooLarge {
            limit: usize::try_from(cap).unwrap_or(usize::MAX),
            provided: usize::try_from(meta.len()).unwrap_or(usize::MAX),
        });
    }
    Ok(meta.len())
}

/// Parse a built-in v1 payload into its final typed allocation in fixed
/// blocks, avoiding a second allocation proportional to file size.
fn read_pod_cache<T: bytemuck::Pod>(
    path: &Path,
    file: &mut File,
    cap: u64,
    alg_id: u8,
) -> Result<(Vec<T>, f32)> {
    let file_len = checked_cache_len(path, file, cap)?;
    let mut header = [0u8; crate::serial::HEADER_SIZE];
    file.read_exact(&mut header).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            AfpError::Deserialize(format!("{}: short fingerprint header", path.display()))
        } else {
            AfpError::io_with_path(path, e)
        }
    })?;
    let (_, count, fps) = crate::serial::read_header(&header, Some(alg_id))?;
    let elem_size = core::mem::size_of::<T>();
    if elem_size == 0 || elem_size > 64 * 1024 {
        return Err(AfpError::Deserialize(
            "unsupported cache element size".into(),
        ));
    }
    let payload_len = (count as usize)
        .checked_mul(elem_size)
        .ok_or_else(|| AfpError::Deserialize("fingerprint payload length overflow".into()))?;
    let required_len = crate::serial::HEADER_SIZE
        .checked_add(payload_len)
        .ok_or_else(|| AfpError::Deserialize("fingerprint file length overflow".into()))?;
    if required_len as u64 > file_len {
        return Err(AfpError::Deserialize(format!(
            "{}: payload too short: need {required_len} bytes, got {file_len}",
            path.display()
        )));
    }
    let mut values = Vec::with_capacity(count as usize);
    let mut block = [0u8; 64 * 1024];
    let per_block = block.len() / elem_size;
    let mut remaining = count as usize;
    while remaining > 0 {
        let n = remaining.min(per_block);
        let bytes = &mut block[..n * elem_size];
        file.read_exact(bytes).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                AfpError::Deserialize(format!("{}: fingerprint payload truncated", path.display()))
            } else {
                AfpError::io_with_path(path, e)
            }
        })?;
        values.extend(
            bytes
                .chunks_exact(elem_size)
                .map(bytemuck::pod_read_unaligned::<T>),
        );
        remaining -= n;
    }
    Ok((values, fps))
}

fn read_limited(path: &Path, reader: impl Read, cap: u64) -> Result<Vec<u8>> {
    let mut limited = reader.take(cap.saturating_add(1));
    let mut bytes = Vec::new();
    limited
        .read_to_end(&mut bytes)
        .map_err(|e| AfpError::io_with_path(path, e))?;
    if bytes.len() as u64 > cap {
        return Err(AfpError::InputTooLarge {
            limit: usize::try_from(cap).unwrap_or(usize::MAX),
            provided: bytes.len(),
        });
    }
    Ok(bytes)
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
    let mut file = File::create(path).map_err(|e| AfpError::io_with_path(path, e))?;
    fp.write_cache_to(&mut file)
        .map_err(|e| AfpError::io_with_path(path, e))
}

/// RAII guard: removes the temp file on drop unless disarmed after a
/// successful rename.
struct AtomicTempGuard(PathBuf);

impl AtomicTempGuard {
    fn disarm(&mut self) {
        self.0 = PathBuf::new();
    }
}

impl Drop for AtomicTempGuard {
    fn drop(&mut self) {
        if !self.0.as_os_str().is_empty() {
            let _ = fs::remove_file(&self.0);
        }
    }
}

/// Write a fingerprint atomically: create a private temporary sibling,
/// write the full v1 blob, then rename into place.
///
/// On failure the previous file at `path` (if any) is preserved. The
/// temporary file is removed when possible. Parent directories are **not**
/// created (caller's job), matching [`cache_to_file`].
///
/// Durability: the temp file is `fsync`ed before `rename`. Parent-directory
/// `fsync` is **not** performed — on Windows (and some network filesystems)
/// a crash immediately after `rename` can still leave the directory entry
/// invisible until the volume journal replays. Callers needing strict
/// crash safety should fsync the parent directory themselves after this
/// returns `Ok`.
///
/// # Errors
///
/// `AfpError::Io` with the path attached on any filesystem failure.
pub fn cache_to_file_atomic<T: CacheableFingerprint>(fp: &T, path: &Path) -> Result<()> {
    let base = ATOMIC_TEMP_COUNTER.fetch_add(8, std::sync::atomic::Ordering::Relaxed);
    cache_to_file_atomic_at(fp, path, base)
}

fn cache_to_file_atomic_at<T: CacheableFingerprint>(fp: &T, path: &Path, base: u64) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_name()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "cache".into());

    let mut temp_path = PathBuf::new();
    let mut file = None;
    for attempt in 0..8u64 {
        let candidate = parent.join(format!(
            ".{}.afp.tmp.{}_{}",
            stem,
            std::process::id(),
            base.wrapping_add(attempt),
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(f) => {
                temp_path = candidate;
                file = Some(f);
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(AfpError::io_with_path(&candidate, e)),
        }
    }
    let mut file = file.ok_or_else(|| {
        AfpError::io_with_path(
            parent,
            std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not allocate a unique temporary cache file name",
            ),
        )
    })?;
    let mut guard = AtomicTempGuard(temp_path.clone());

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|e| AfpError::io_with_path(&temp_path, e))?;
    }

    fp.write_cache_to(&mut file)
        .map_err(|e| AfpError::io_with_path(&temp_path, e))?;
    file.sync_all()
        .map_err(|e| AfpError::io_with_path(&temp_path, e))?;
    drop(file);

    match fs::rename(&temp_path, path) {
        Ok(()) => {
            guard.disarm();
            Ok(())
        }
        Err(e) => Err(AfpError::io_with_path(path, e)),
    }
}

/// Load a fingerprint from a `.afp` cache file.
///
/// The file must be a regular file no larger than
/// [`MAX_CACHE_FILE_BYTES`]; symlinks are rejected rather than followed.
///
/// # Errors
///
/// - `AfpError::Io` if the file cannot be read or is not a regular file.
/// - `AfpError::InputTooLarge` if the file exceeds [`MAX_CACHE_FILE_BYTES`].
/// - `AfpError::Deserialize` if the contents are not a valid v1 blob for `T`.
pub fn load_from_cache<T: CacheableFingerprint>(path: &Path) -> Result<T> {
    let mut file = open_cache_file_for_read(path)?;
    T::read_cache_from(path, &mut file, MAX_CACHE_FILE_BYTES)
}

/// Load every `*.afp` file in a directory (non-recursive).
///
/// Only **regular files** whose extension is `.afp` are read —
/// subdirectories, symlinks, FIFOs, and other devices are skipped, so a
/// dangling or `/dev/*` symlink cannot abort or hang the scan. Entries
/// are sorted by path for deterministic ingest order. An empty directory
/// yields `Ok(vec![])`.
///
/// Files larger than [`MAX_CACHE_FILE_BYTES`] are rejected
/// ([`AfpError::InputTooLarge`]).
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
    load_all_cached_limited(dir, CacheLoadLimits::default())
}

/// Like [`load_all_cached`] with optional aggregate file-count and
/// total-byte caps.
///
/// # Errors
///
/// Same as [`load_all_cached`], plus [`AfpError::InputTooLarge`] when an
/// aggregate limit is exceeded (the error message names the limit).
pub fn load_all_cached_limited(
    dir: &Path,
    limits: CacheLoadLimits,
) -> Result<Vec<(PathBuf, CachedFingerprint)>> {
    let mut out = Vec::new();
    for item in iter_cached(dir, limits)? {
        out.push(item?);
    }
    Ok(out)
}

/// Iterator over sorted `.afp` files in `dir` with optional aggregate
/// limits applied as entries are loaded.
///
/// # Errors
///
/// Returns `Err` when the directory cannot be read. Individual load
/// failures are returned as `Item = Err(...)`.
pub fn iter_cached(dir: &Path, limits: CacheLoadLimits) -> Result<CacheDirIter> {
    let entries = fs::read_dir(dir).map_err(|e| AfpError::io_with_path(dir.to_path_buf(), e))?;
    let mut paths: Vec<PathBuf> = Vec::new();
    let mut afp_count = 0usize;
    for entry in entries {
        let entry = entry.map_err(|e| AfpError::io_with_path(dir.to_path_buf(), e))?;
        let file_type = entry
            .file_type()
            .map_err(|e| AfpError::io_with_path(entry.path(), e))?;
        if !file_type.is_file() {
            continue;
        }
        let path = entry.path();
        let is_afp = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case(AFP_EXT));
        if is_afp {
            afp_count += 1;
            if let Some(max_files) = limits.max_files {
                if paths.len() < max_files {
                    paths.push(path);
                }
            } else {
                paths.push(path);
            }
        }
    }
    paths.sort();
    let excess_files = limits
        .max_files
        .is_some_and(|max_files| afp_count > max_files);
    Ok(CacheDirIter {
        paths,
        index: 0,
        limits,
        bytes_loaded: 0,
        files_loaded: 0,
        excess_files,
        afp_count,
    })
}

/// Lazy directory scan for `.afp` fingerprints.
pub struct CacheDirIter {
    paths: Vec<PathBuf>,
    index: usize,
    limits: CacheLoadLimits,
    bytes_loaded: u64,
    files_loaded: usize,
    excess_files: bool,
    afp_count: usize,
}

impl Iterator for CacheDirIter {
    type Item = Result<(PathBuf, CachedFingerprint)>;

    fn next(&mut self) -> Option<Self::Item> {
        let path = match self.paths.get(self.index) {
            Some(p) => {
                self.index += 1;
                p.clone()
            }
            None => {
                if self.excess_files {
                    self.excess_files = false;
                    let max_files = self.limits.max_files.unwrap_or(0);
                    return Some(Err(AfpError::InputTooLarge {
                        limit: max_files,
                        provided: self.afp_count,
                    }));
                }
                return None;
            }
        };
        if let Some(max_total) = self.limits.max_total_bytes {
            let remaining = max_total.saturating_sub(self.bytes_loaded);
            if remaining == 0 {
                return Some(Err(AfpError::InputTooLarge {
                    limit: usize::try_from(max_total).unwrap_or(usize::MAX),
                    provided: usize::try_from(self.bytes_loaded)
                        .unwrap_or(usize::MAX)
                        .saturating_add(1),
                }));
            }
            let meta = match fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(e) => return Some(Err(AfpError::io_with_path(&path, e))),
            };
            if meta.is_file() && meta.len() > remaining {
                return Some(Err(AfpError::InputTooLarge {
                    limit: usize::try_from(max_total).unwrap_or(usize::MAX),
                    provided: usize::try_from(self.bytes_loaded.saturating_add(meta.len()))
                        .unwrap_or(usize::MAX),
                }));
            }
        }
        let cap = self
            .limits
            .max_total_bytes
            .map_or(MAX_CACHE_FILE_BYTES, |max| {
                max.saturating_sub(self.bytes_loaded)
                    .min(MAX_CACHE_FILE_BYTES)
            });
        let (fp, file_len) = match open_cache_file_for_read(&path)
            .and_then(|mut file| {
                let file_len = checked_cache_len(&path, &file, cap)?;
                let fp = CachedFingerprint::from_file(&path, &mut file, cap)?;
                Ok((fp, file_len))
            })
            .map_err(|e| match e {
                AfpError::Deserialize(msg) => {
                    AfpError::Deserialize(format!("{}: {msg}", path.display()))
                }
                AfpError::InputTooLarge { limit, provided } => AfpError::Deserialize(format!(
                    "{}: payload exceeds limit (limit {limit}, provided {provided})",
                    path.display()
                )),
                other => other,
            }) {
            Ok(result) => result,
            Err(e) => return Some(Err(e)),
        };
        self.bytes_loaded = self.bytes_loaded.saturating_add(file_len);
        self.files_loaded += 1;
        Some(Ok((path, fp)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classical::{WangFingerprint, WangHash};

    #[test]
    fn streamed_builtin_writes_are_byte_identical_to_v1_blobs() {
        let wang = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x1234_5678,
                t_anchor: 42,
            }],
            frames_per_sec: 62.5,
        };
        let panako = PanakoFingerprint {
            hashes: vec![],
            frames_per_sec: 62.5,
        };
        let haitsma = HaitsmaFingerprint {
            frames: vec![0xAABB_CCDD, 0x0123_4567],
            frames_per_sec: 78.125,
        };
        fn assert_same<T: CacheableFingerprint>(fp: &T) {
            let mut actual = Vec::new();
            fp.write_cache_to(&mut actual).unwrap();
            assert_eq!(actual, fp.to_cache_bytes());
        }
        assert_same(&wang);
        assert_same(&panako);
        assert_same(&haitsma);
    }

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

    #[cfg(unix)]
    #[test]
    fn symlinked_afp_files_are_skipped_not_followed() {
        // A `*.afp` symlink could point at /dev/zero (unbounded read) or
        // dangle (would abort the whole scan). Both must be skipped.
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("symlink");
        cache_to_file(&fp, &dir.0.join("real.afp")).unwrap();
        std::os::unix::fs::symlink("/dev/zero", dir.0.join("zero.afp")).unwrap();
        std::os::unix::fs::symlink(dir.0.join("nope"), dir.0.join("dead.afp")).unwrap();

        let loaded = load_all_cached(&dir.0).expect("symlinks must not abort the scan");
        let names: Vec<&str> = loaded
            .iter()
            .map(|(p, _)| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["real.afp"], "only regular files are loaded");
    }

    #[cfg(unix)]
    #[test]
    fn load_from_cache_rejects_a_symlink_pointing_at_a_device() {
        let dir = TempDir::new("symlink_device");
        let link = dir.0.join("zero.afp");
        std::os::unix::fs::symlink("/dev/zero", &link).unwrap();
        let err = load_from_cache::<WangFingerprint>(&link).unwrap_err();
        assert!(
            matches!(err, AfpError::Io(_)),
            "device symlink must be an Io error, got {err:?}"
        );
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
    fn cache_to_file_atomic_roundtrip() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xABCD,
                t_anchor: 1,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("atomic_ok");
        let path = dir.0.join("atomic.afp");
        cache_to_file_atomic(&fp, &path).unwrap();
        let restored: WangFingerprint = load_from_cache(&path).unwrap();
        assert_eq!(restored.hashes, fp.hashes);
    }

    #[cfg(unix)]
    #[test]
    fn atomic_write_failure_preserves_existing_cache() {
        use std::os::unix::fs::PermissionsExt;

        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x11,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("atomic_preserve");
        let path = dir.0.join("keep.afp");
        cache_to_file(&fp, &path).unwrap();
        let before = std::fs::read(&path).unwrap();

        let mut perms = std::fs::metadata(&dir.0).unwrap().permissions();
        perms.set_mode(0o555);
        std::fs::set_permissions(&dir.0, perms).unwrap();

        let fp2 = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x22,
                t_anchor: 1,
            }],
            frames_per_sec: 62.5,
        };
        assert!(cache_to_file_atomic(&fp2, &path).is_err());
        let after = std::fs::read(&path).unwrap();
        assert_eq!(before, after, "failed atomic write must not truncate cache");
    }

    #[test]
    fn iter_cached_respects_aggregate_limits() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("iter_limits");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();
        cache_to_file(&fp, &dir.0.join("b.afp")).unwrap();

        let limits = CacheLoadLimits {
            max_files: Some(1),
            max_total_bytes: None,
        };
        let mut iter = iter_cached(&dir.0, limits).unwrap();
        assert!(iter.next().unwrap().is_ok());
        match iter.next() {
            Some(Err(AfpError::InputTooLarge {
                limit: 1,
                provided: 2,
            })) => {}
            other => panic!("expected InputTooLarge when more .afp files remain, got {other:?}"),
        }

        let one = load_from_cache::<WangFingerprint>(&dir.0.join("a.afp")).unwrap();
        let bytes_one = one.to_cache_bytes().len() as u64;
        let limits = CacheLoadLimits {
            max_files: None,
            max_total_bytes: Some(bytes_one),
        };
        let err = load_all_cached_limited(&dir.0, limits).unwrap_err();
        assert!(
            matches!(err, AfpError::InputTooLarge { .. }),
            "byte budget exhaustion must error, got {err:?}"
        );
    }

    #[test]
    fn max_files_zero_on_nonempty_dir_errors() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("max_files_zero");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();

        let limits = CacheLoadLimits {
            max_files: Some(0),
            max_total_bytes: None,
        };
        let mut iter = iter_cached(&dir.0, limits).unwrap();
        match iter.next() {
            Some(Err(AfpError::InputTooLarge {
                limit: 0,
                provided: 1,
            })) => {}
            other => panic!("expected InputTooLarge for max_files=0, got {other:?}"),
        }
    }

    #[test]
    fn max_total_bytes_rejects_before_reading_oversized_file() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("max_bytes");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();
        cache_to_file(&fp, &dir.0.join("b.afp")).unwrap();
        let one_file_bytes = fp.to_cache_bytes().len() as u64;

        let limits = CacheLoadLimits {
            max_files: None,
            max_total_bytes: Some(one_file_bytes),
        };
        let mut iter = iter_cached(&dir.0, limits).unwrap();
        assert!(iter.next().unwrap().is_ok(), "first file fits byte cap");
        match iter.next() {
            Some(Err(AfpError::InputTooLarge { .. })) => {}
            other => panic!("expected InputTooLarge when byte budget exhausted, got {other:?}"),
        }
    }

    #[test]
    fn corrupt_file_errors_and_does_not_fuse_with_limits() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("corrupt_iter");
        cache_to_file(&fp, &dir.0.join("good.afp")).unwrap();
        std::fs::write(dir.0.join("bad.afp"), b"not a fingerprint").unwrap();

        let limits = CacheLoadLimits {
            max_files: Some(10),
            max_total_bytes: None,
        };
        let iter = iter_cached(&dir.0, limits).unwrap();
        let mut saw_good = false;
        let mut saw_bad_err = false;
        for item in iter {
            match item {
                Ok((path, _)) => {
                    assert_eq!(path.file_name().unwrap(), "good.afp");
                    saw_good = true;
                }
                Err(e) => {
                    assert!(
                        e.to_string().contains("bad.afp"),
                        "corrupt file must surface its path: {e}"
                    );
                    saw_bad_err = true;
                }
            }
        }
        assert!(saw_good, "good file must load before corrupt entry fails");
        assert!(saw_bad_err, "corrupt file must fail with path in error");
    }

    #[test]
    fn atomic_create_new_collision_preserves_existing_temp() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0xBB,
                t_anchor: 1,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("atomic_collision");
        let target = dir.0.join("keep.afp");
        let stem = target.file_name().unwrap().to_string_lossy();
        let base = 0;
        let colliding = dir
            .0
            .join(format!(".{}.afp.tmp.{}_{}", stem, std::process::id(), base,));
        std::fs::write(&colliding, b"unrelated stale temp").unwrap();

        cache_to_file_atomic_at(&fp, &target, base).unwrap();
        assert_eq!(
            std::fs::read(&colliding).unwrap(),
            b"unrelated stale temp",
            "pre-existing colliding temp path must not be deleted on create_new failure"
        );
        let restored: WangFingerprint = load_from_cache(&target).unwrap();
        assert_eq!(restored.hashes, fp.hashes);
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

    #[test]
    fn wang_envelope_from_cached_fingerprint() {
        let wang = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x42,
                t_anchor: 3,
            }],
            frames_per_sec: 62.5,
        };
        let env = CachedFingerprint::Wang(wang).envelope();
        assert_eq!(env.algorithm, "wang-v1");
        assert_eq!(env.sample_rate, 8_000);
        assert_eq!(env.hash_count, 1);
        assert_eq!(env.frames_per_sec, 62.5);
    }

    #[test]
    fn actual_read_rejects_growth_without_trusting_metadata() {
        let path = Path::new("growing.afp");
        let mut source = std::io::Cursor::new(vec![7; 64]);
        let result = read_limited(path, &mut source, 8);
        assert!(matches!(
            result,
            Err(AfpError::InputTooLarge {
                limit: 8,
                provided: 9
            })
        ));
        assert_eq!(source.position(), 9, "read only one byte beyond budget");
        let exact = read_limited(path, std::io::Cursor::new(vec![7; 8]), 8).unwrap();
        assert_eq!(exact, vec![7; 8]);
    }

    #[test]
    fn iterator_handles_files_removed_after_enumeration() {
        for budget in [None, Some(1000)] {
            let dir = TempDir::new("removed_after_list");
            let path = dir.0.join("gone.afp");
            std::fs::write(&path, b"placeholder").unwrap();
            let mut iter = iter_cached(
                &dir.0,
                CacheLoadLimits {
                    max_files: None,
                    max_total_bytes: budget,
                },
            )
            .unwrap();
            std::fs::remove_file(&path).unwrap();
            assert!(matches!(iter.next(), Some(Err(AfpError::Io(_)))));
            assert!(iter.next().is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn handle_reader_rejects_directories_and_devices() {
        let dir = TempDir::new("non_regular_handle");
        for path in [dir.0.as_path(), Path::new("/dev/null")] {
            let mut file = File::open(path).unwrap();
            let result = read_handle_with_limit(path, &mut file, 32);
            assert!(matches!(result, Err(AfpError::Io(_))));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn handle_reader_rejects_a_symlink_handle_without_following() {
        use rustix::fs::OFlags;
        use std::os::unix::fs::OpenOptionsExt;
        let dir = TempDir::new("symlink_handle");
        let path = dir.0.join("link.afp");
        std::os::unix::fs::symlink("missing.afp", &path).unwrap();
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags((OFlags::PATH | OFlags::NOFOLLOW).bits() as i32)
            .open(&path)
            .unwrap();
        let result = read_handle_with_limit(&path, &mut file, 32);
        assert!(matches!(result, Err(AfpError::Io(ref err))
            if err.source.to_string().contains("symlink")));
    }

    #[cfg(unix)]
    #[test]
    fn load_from_cache_rejects_file_over_max_bytes() {
        let dir = TempDir::new("oversized");
        let path = dir.0.join("huge.afp");
        {
            let mut f = std::fs::File::create(&path).unwrap();
            use std::io::Write;
            f.write_all(b"AUDIOFP\0").unwrap();
            f.set_len(MAX_CACHE_FILE_BYTES + 1).unwrap();
        }
        let err = load_from_cache::<WangFingerprint>(&path).unwrap_err();
        match err {
            AfpError::InputTooLarge { limit, provided } => {
                assert_eq!(limit, MAX_CACHE_FILE_BYTES as usize);
                assert_eq!(provided, MAX_CACHE_FILE_BYTES as usize + 1);
            }
            other => panic!("expected InputTooLarge for oversized cache file, got {other:?}"),
        }
    }

    #[test]
    fn atomic_rename_to_directory_fails_and_drops_temp() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x99,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("atomic_rename_fail");
        let target = dir.0.join("blocked.afp");
        std::fs::create_dir(&target).unwrap();
        let err = cache_to_file_atomic(&fp, &target).unwrap_err();
        assert!(
            err.to_string().contains("blocked.afp"),
            "rename failure must name the destination: {err}"
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir.0)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| n.to_string_lossy().contains(".afp.tmp."))
            .collect();
        assert!(
            leftovers.is_empty(),
            "failed atomic write must remove its temp file, found {leftovers:?}"
        );
    }

    #[test]
    fn atomic_temp_name_exhaustion_returns_error() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 0x77,
                t_anchor: 1,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("temp_exhaust");
        let target = dir.0.join("out.afp");
        let stem = target.file_name().unwrap().to_string_lossy();
        let attempt_base = 0_u64;
        for attempt in 0..8u64 {
            let colliding = dir.0.join(format!(
                ".{}.afp.tmp.{}_{}",
                stem,
                std::process::id(),
                attempt_base.wrapping_add(attempt),
            ));
            std::fs::write(&colliding, b"occupied").unwrap();
        }
        let err = cache_to_file_atomic_at(&fp, &target, attempt_base).unwrap_err();
        assert!(
            err.to_string().contains("unique temporary cache file"),
            "expected temp allocation failure, got {err}"
        );
    }

    #[test]
    fn iter_byte_budget_rejects_second_file_from_metadata() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("byte_meta");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();
        cache_to_file(&fp, &dir.0.join("b.afp")).unwrap();
        let one = fp.to_cache_bytes().len() as u64;
        let limits = CacheLoadLimits {
            max_files: None,
            max_total_bytes: Some(one + one / 2),
        };
        let mut iter = iter_cached(&dir.0, limits).unwrap();
        assert!(iter.next().unwrap().is_ok(), "first file must fit");
        match iter.next() {
            Some(Err(AfpError::InputTooLarge { limit, provided })) => {
                assert_eq!(limit, (one + one / 2) as usize);
                assert_eq!(provided, (one + one) as usize);
            }
            other => panic!("expected metadata byte-budget rejection, got {other:?}"),
        }
    }

    #[test]
    fn iter_byte_budget_zero_remaining_after_first_file() {
        let fp = WangFingerprint {
            hashes: vec![WangHash {
                hash: 1,
                t_anchor: 0,
            }],
            frames_per_sec: 62.5,
        };
        let dir = TempDir::new("byte_zero_remain");
        cache_to_file(&fp, &dir.0.join("a.afp")).unwrap();
        cache_to_file(&fp, &dir.0.join("b.afp")).unwrap();
        let one = fp.to_cache_bytes().len() as u64;
        let limits = CacheLoadLimits {
            max_files: None,
            max_total_bytes: Some(one),
        };
        let mut iter = iter_cached(&dir.0, limits).unwrap();
        assert!(iter.next().unwrap().is_ok());
        match iter.next() {
            Some(Err(AfpError::InputTooLarge { limit, provided })) => {
                assert_eq!(limit, one as usize);
                assert_eq!(provided, one as usize + 1);
            }
            other => panic!("expected zero-remaining byte budget error, got {other:?}"),
        }
    }
}
