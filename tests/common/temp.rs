//! RAII scratch paths for integration tests (collision-safe).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Unique file or directory under the system temp dir, removed on drop.
pub struct TempPath(PathBuf, bool);

impl TempPath {
    /// Create a unique path (optionally as a directory).
    pub fn new(label: &str, is_dir: bool) -> Self {
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "audiofp_test_{label}_{}_{}.tmp",
            std::process::id(),
            n
        ));
        if is_dir {
            std::fs::create_dir_all(&path).expect("temp dir");
        }
        Self(path, is_dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        if self.1 {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}
