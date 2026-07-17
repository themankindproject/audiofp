//! Shared PCM validation helpers (finite checks, push truncation).

use alloc::vec::Vec;

use crate::{AfpError, Result};

/// Reject buffers that contain NaN or ±Inf.
///
/// Offline `extract` paths call this so non-finite PCM cannot poison
/// peaks/hashes with an `Ok` result.
#[inline]
pub(crate) fn reject_non_finite(samples: &[f32]) -> Result<()> {
    if let Some(index) = samples.iter().position(|s| !s.is_finite()) {
        return Err(AfpError::NonFiniteSample { index });
    }
    Ok(())
}

/// Truncate a streaming `push` chunk to `max` samples when configured.
#[inline]
pub(crate) fn truncate_push(samples: &[f32], max: Option<usize>) -> &[f32] {
    match max {
        Some(limit) if samples.len() > limit => &samples[..limit],
        _ => samples,
    }
}

/// Append `samples` to `carry`, replacing any non-finite value with `0.0`.
///
/// Streaming `push` is infallible, so sanitize rather than error. Clean
/// (all-finite) chunks take the fast `extend_from_slice` path.
#[inline]
pub(crate) fn extend_sanitized(carry: &mut Vec<f32>, samples: &[f32]) {
    if samples.iter().all(|s| s.is_finite()) {
        carry.extend_from_slice(samples);
        return;
    }
    carry.reserve(samples.len());
    for &s in samples {
        carry.push(if s.is_finite() { s } else { 0.0 });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn reject_ok_on_finite() {
        assert!(reject_non_finite(&[0.0, 1.0, -0.5]).is_ok());
    }

    #[test]
    fn reject_reports_first_nan_index() {
        let err = reject_non_finite(&[0.0, f32::NAN, f32::INFINITY]).unwrap_err();
        assert!(matches!(err, AfpError::NonFiniteSample { index: 1 }));
    }

    #[test]
    fn truncate_push_respects_limit() {
        let s = [1.0_f32; 10];
        assert_eq!(truncate_push(&s, Some(4)).len(), 4);
        assert_eq!(truncate_push(&s, None).len(), 10);
    }

    #[test]
    fn extend_sanitized_replaces_nan() {
        let mut carry = vec![];
        extend_sanitized(&mut carry, &[1.0, f32::NAN, 2.0]);
        assert_eq!(carry, vec![1.0, 0.0, 2.0]);
    }
}
