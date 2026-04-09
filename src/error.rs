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
