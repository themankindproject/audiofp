//! Error type for the `afp` crate.
//!
//! Every fallible API in `afp` returns [`Result<T>`], a type alias for
//! `core::result::Result<T, AfpError>`.

use alloc::string::String;
use thiserror::Error;

/// All errors surfaced by `afp`.
///
/// Marked `#[non_exhaustive]` so that adding a new variant in a future
/// version is not a breaking change. Match exhaustively only inside the
/// crate.
///
/// # Example
///
/// ```
/// use afp::AfpError;
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

    /// The audio's sample rate is not one of the supported rates.
    #[error(
        "unsupported sample rate: {0} Hz (supported: 8000, 11025, 16000, 22050, 44100, 48000)"
    )]
    UnsupportedSampleRate(u32),

    /// The audio has a channel count `afp` cannot consume (must be mono).
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

    /// A streaming pipeline dropped samples because the consumer fell behind.
    #[error("buffer overrun: dropped {dropped} samples")]
    BufferOverrun {
        /// Number of samples dropped.
        dropped: usize,
    },

    /// A configuration value was rejected (out of range, mutually exclusive, …).
    #[error("invalid configuration: {0}")]
    Config(String),

    /// An I/O failure surfaced through `afp`.
    #[error("io: {0}")]
    Io(String),
}

/// Shorthand for `core::result::Result<T, AfpError>`.
///
/// # Example
///
/// ```
/// use afp::{AfpError, Result};
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
        let e = AfpError::AudioTooShort { needed: 16_000, got: 8_000 };
        let s = e.to_string();
        assert!(s.contains("16000"), "got: {s}");
        assert!(s.contains("8000"), "got: {s}");
    }

    #[test]
    fn unsupported_sample_rate_shows_value_and_supported_list() {
        let s = AfpError::UnsupportedSampleRate(7_000).to_string();
        assert!(s.contains("7000"));
        // The supported list mentions all six canonical rates.
        for rate in ["8000", "11025", "16000", "22050", "44100", "48000"] {
            assert!(s.contains(rate), "missing {rate} in: {s}");
        }
    }

    #[test]
    fn buffer_overrun_reports_drop_count() {
        let s = AfpError::BufferOverrun { dropped: 1024 }.to_string();
        assert!(s.contains("1024"));
    }

    #[test]
    fn result_ok_path() {
        let f = |x: u32| -> Result<u32> { Ok(x * 2) };
        assert_eq!(f(21).unwrap(), 42);
    }
}
