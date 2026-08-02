//! Error type for the `audiofp` crate.
//!
//! Every fallible API in `audiofp` returns [`Result<T>`], a type alias for
//! `core::result::Result<T, AfpError>`.

use alloc::string::String;
use thiserror::Error;

/// All errors surfaced by `audiofp`.
///
/// Marked `#[non_exhaustive]` so that adding a new variant in a future
/// version is not a breaking change. Match exhaustively only inside the
/// crate.
///
/// # Example
///
/// ```
/// use audiofp::AfpError;
///
/// let err = AfpError::AudioTooShort { needed: 16_000, got: 8_000 };
/// assert_eq!(
///     err.to_string(),
///     "audio too short: needed at least 16000 samples, got 8000",
/// );
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AfpError {
    /// The caller-supplied audio buffer is shorter than the fingerprinter's
    /// minimum window.
    #[error("audio too short: needed at least {needed} samples, got {got}")]
    AudioTooShort {
        /// Minimum required sample count.
        needed: usize,
        /// Sample count actually supplied.
        got: usize,
    },

    /// The audio's sample rate does not match the rate the fingerprinter
    /// expects. Each fingerprinter has a single required rate; consult
    /// [`Fingerprinter::required_sample_rate`](crate::Fingerprinter::required_sample_rate)
    /// or [`StreamingFingerprinter::required_sample_rate`](crate::StreamingFingerprinter::required_sample_rate)
    /// to learn the value.
    #[error("unsupported sample rate: {0} Hz")]
    UnsupportedSampleRate(u32),

    /// The audio has a channel count `audiofp` cannot consume (must be mono).
    #[error("unsupported channel count: {0}")]
    UnsupportedChannels(u16),

    /// A model file was expected at the given path but was not found.
    #[error("model not found at {0}")]
    ModelNotFound(String),

    /// The model file was found but failed to load (corrupt, wrong format, …).
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Inference against a loaded model failed at runtime.
    #[error("inference failed: {0}")]
    Inference(String),

    /// The input exceeds the configured maximum. Raised early in decode
    /// and extract paths to prevent OOM from untrusted audio. The limit
    /// can be raised or disabled entirely via the config struct.
    ///
    /// `limit` / `provided` share the same unit as the check that failed
    /// (samples, bytes, or hash count — see the call site).
    #[error(
        "input too large: {provided} exceeds maximum {limit}; \
         raise the limit or set it to None to disable"
    )]
    InputTooLarge {
        /// Configured limit that was exceeded.
        limit: usize,
        /// Actual size that exceeded the limit (same unit as `limit`).
        provided: usize,
    },

    /// A streaming pipeline dropped samples because the consumer fell behind.
    ///
    /// Reserved for bounded real-time capture (e.g. mic ring buffer). Not
    /// emitted by classical/neural extractors today; see issue tracking the
    /// mic orchestrator.
    #[error("buffer overrun: dropped {dropped} samples")]
    BufferOverrun {
        /// Number of samples dropped.
        dropped: usize,
    },

    /// Input PCM contained NaN or ±Inf at `index`.
    ///
    /// Offline `extract` / watermark `detect` reject non-finite samples.
    /// Streaming `push` sanitizes them to `0.0` instead (infallible API).
    #[error("audio contains non-finite sample (NaN or Inf) at index {index}")]
    NonFiniteSample {
        /// Index of the first non-finite sample.
        index: usize,
    },

    /// Deserialization of a fingerprint binary blob failed (magic mismatch,
    /// unsupported version, truncated payload, wrong algorithm, …).
    #[error("deserialize: {0}")]
    Deserialize(String),

    /// A configuration value was rejected (out of range, mutually exclusive, …).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// An I/O failure surfaced through `audiofp`.
    ///
    /// When the `std` feature is enabled, this carries a structured
    /// [`IoError`] with path, kind, and source. Without `std`, it
    /// carries only a string description.
    #[cfg(feature = "std")]
    #[error("{0}")]
    Io(IoError),

    /// An I/O failure (no_std fallback — string only).
    #[cfg(not(feature = "std"))]
    #[error("io: {0}")]
    Io(String),
}

// ---------------------------------------------------------------------------
// IoError — structured I/O error (std only)
// ---------------------------------------------------------------------------

/// Structured I/O error with path and source.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct IoError {
    /// The file path where the error occurred, if known.
    pub path: Option<std::path::PathBuf>,
    /// The kind of I/O error.
    pub kind: std::io::ErrorKind,
    /// The underlying error.
    pub source: std::io::Error,
}

#[cfg(feature = "std")]
impl IoError {
    /// Create a new `IoError` with a path.
    pub fn new(path: impl Into<std::path::PathBuf>, source: std::io::Error) -> Self {
        let kind = source.kind();
        Self {
            path: Some(path.into()),
            kind,
            source,
        }
    }

    /// Create a new `IoError` without a path.
    pub fn without_path(source: std::io::Error) -> Self {
        let kind = source.kind();
        Self {
            path: None,
            kind,
            source,
        }
    }
}

#[cfg(feature = "std")]
impl core::fmt::Display for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.path {
            Some(p) => write!(f, "io error at {}: {}", p.display(), self.source),
            None => write!(f, "io error: {}", self.source),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for AfpError {
    fn from(e: std::io::Error) -> Self {
        AfpError::Io(IoError::without_path(e))
    }
}

#[cfg(feature = "std")]
impl AfpError {
    /// Create an `Io` error with a path context.
    pub fn io_with_path(path: impl Into<std::path::PathBuf>, source: std::io::Error) -> Self {
        AfpError::Io(IoError::new(path, source))
    }
}

// ---------------------------------------------------------------------------
// Model-loading helpers (shared by `neural` and `watermark` features)
// ---------------------------------------------------------------------------

/// Map a failed filesystem open into the appropriate [`AfpError`] variant.
///
/// Used by both the neural embedder and watermark detector when opening
/// ONNX model files.
#[cfg(feature = "std")]
pub(crate) fn map_model_open_io(path: &str, e: std::io::Error) -> AfpError {
    use alloc::string::ToString;
    if e.kind() == std::io::ErrorKind::NotFound {
        AfpError::ModelNotFound(path.to_string())
    } else {
        AfpError::ModelLoad(alloc::format!("open: {e}"))
    }
}

/// Map any `Display`-able model-load error into [`AfpError::ModelLoad`].
#[cfg(feature = "std")]
pub(crate) fn map_model_load_err(e: impl core::fmt::Display) -> AfpError {
    AfpError::ModelLoad(alloc::format!("load: {e}"))
}

/// Shorthand for `core::result::Result<T, AfpError>`.
///
/// # Example
///
/// ```
/// use audiofp::{AfpError, Result};
///
/// fn at_least_one_second(samples: &[f32]) -> Result<()> {
///     if samples.len() < 16_000 {
///         return Err(AfpError::AudioTooShort { needed: 16_000, got: samples.len() });
///     }
///     Ok(())
/// }
/// # at_least_one_second(&vec![0.0; 16_000]).unwrap();
/// ```
pub type Result<T> = core::result::Result<T, AfpError>;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn audio_too_short_displays_both_numbers() {
        let e = AfpError::AudioTooShort {
            needed: 16_000,
            got: 8_000,
        };
        let s = e.to_string();
        assert!(s.contains("16000"), "got: {s}");
        assert!(s.contains("8000"), "got: {s}");
    }

    #[test]
    fn unsupported_sample_rate_displays_both_rates() {
        // The message must NOT claim a global "supported" list — each
        // fingerprinter has its own required rate.
        let s = AfpError::UnsupportedSampleRate(7_000).to_string();
        assert!(s.contains("7000"), "must contain the offending rate: {s}");
        assert!(
            !s.contains("(supported"),
            "must not advertise a hardcoded supported list: {s}",
        );
    }

    #[test]
    fn non_finite_sample_displays_index() {
        let s = AfpError::NonFiniteSample { index: 42 }.to_string();
        assert!(s.contains("42"));
        assert!(s.contains("non-finite"));
    }

    #[test]
    fn buffer_overrun_reports_drop_count() {
        let s = AfpError::BufferOverrun { dropped: 1024 }.to_string();
        assert!(s.contains("1024"));
    }

    #[test]
    fn input_too_large_displays_both_limit_and_provided() {
        let err = AfpError::InputTooLarge {
            limit: 1_000_000,
            provided: 5_000_000,
        };
        let s = err.to_string();
        assert!(s.contains("1000000"), "got: {s}");
        assert!(s.contains("5000000"), "got: {s}");
        assert!(s.contains("exceeds maximum"), "got: {s}");
    }

    #[test]
    fn result_ok_path() {
        let f = |x: u32| -> Result<u32> { Ok(x * 2) };
        assert_eq!(f(21).unwrap(), 42);
    }

    // -----------------------------------------------------------------
    // Display formatting for variants that were previously untested.
    //
    // Each `assert_eq!` pins the exact `Display` text so a future
    // `thiserror` annotation change is caught. The `to_string` of
    // every public error variant is part of the contract.
    // -----------------------------------------------------------------

    #[test]
    fn unsupported_channels_displays_the_count() {
        let err = AfpError::UnsupportedChannels(7);
        assert_eq!(err.to_string(), "unsupported channel count: 7");
    }

    #[test]
    #[cfg(feature = "std")]
    fn io_displays_path_and_source() {
        let source = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = AfpError::io_with_path("/some/path.wav", source);
        let s = err.to_string();
        assert!(s.contains("/some/path.wav"), "got: {s}");
        assert!(s.contains("file missing"), "got: {s}");
    }

    #[test]
    #[cfg(feature = "std")]
    fn io_without_path_displays_source() {
        let source = std::io::Error::other("disk full");
        let err = AfpError::from(source);
        let s = err.to_string();
        assert!(s.contains("disk full"), "got: {s}");
    }

    #[test]
    #[cfg(not(feature = "std"))]
    fn io_displays_the_inner_message() {
        let err = AfpError::Io("disk full".to_string());
        assert_eq!(err.to_string(), "io: disk full");
    }
}
