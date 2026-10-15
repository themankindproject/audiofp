//! Payload framing: raw bytes followed by an IEEE CRC-32.

use alloc::vec::Vec;

/// Width of the CRC trailer, in bits.
pub(crate) const CRC_BITS: usize = 32;

/// Bit length of `payload_len` bytes plus the CRC trailer.
pub(crate) fn framed_bit_len(payload_len: usize) -> Option<usize> {
    payload_len.checked_mul(8)?.checked_add(CRC_BITS)
}

/// IEEE CRC-32 (polynomial 0xEDB88320, init and xor-out all ones).
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Payload bytes, MSB first, then the CRC-32, MSB first.
pub(crate) fn frame(payload: &[u8]) -> Vec<u8> {
    let mut bits = Vec::with_capacity(payload.len() * 8 + CRC_BITS);
    for &byte in payload {
        for shift in (0..8).rev() {
            bits.push((byte >> shift) & 1);
        }
    }
    let crc = crc32(payload);
    for shift in (0..32).rev() {
        bits.push(((crc >> shift) & 1) as u8);
    }
    bits
}

/// Hard decisions from soft bit votes: positive is 1, negative is 0.
///
/// A vote that is exactly zero (a tie, or a bit nothing voted on) or not
/// finite is an erasure, and a single erasure fails the whole decision.
/// Resolving it to a bit instead would let evidence-free input decode to a
/// fixed frame, and `crc32(0xFF * 4) == 0xFFFF_FFFF` makes the all-ones
/// frame a valid 4-byte payload.
pub(crate) fn hard_bits(votes: &[f64]) -> Option<Vec<u8>> {
    votes
        .iter()
        .map(|&v| match v {
            v if !v.is_finite() => None,
            v if v > 0.0 => Some(1),
            v if v < 0.0 => Some(0),
            _ => None,
        })
        .collect()
}

/// [`unframe`] of `bits`, then of their complement.
///
/// QIM cannot tell a 3 dB level shift from inverted bits: the two lattices
/// are 3 dB apart. Each polarity is a separate CRC trial.
pub(crate) fn unframe_either_polarity(bits: &[u8], payload_len: usize) -> Option<Vec<u8>> {
    unframe(bits, payload_len).or_else(|| {
        let inverted: Vec<u8> = bits.iter().map(|&b| b ^ 1).collect();
        unframe(&inverted, payload_len)
    })
}

/// Inverse of [`frame`]. `None` when the CRC does not match.
pub(crate) fn unframe(bits: &[u8], payload_len: usize) -> Option<Vec<u8>> {
    let n_bits = framed_bit_len(payload_len)?;
    if bits.len() != n_bits {
        return None;
    }
    let mut payload = alloc::vec![0u8; payload_len];
    for (i, byte) in payload.iter_mut().enumerate() {
        let mut value = 0u8;
        for bit in 0..8 {
            value = (value << 1) | bits[i * 8 + bit];
        }
        *byte = value;
    }
    let mut crc = 0u32;
    let base = payload_len * 8;
    for bit in 0..CRC_BITS {
        crc = (crc << 1) | u32::from(bits[base + bit]);
    }
    if crc == crc32(&payload) {
        Some(payload)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{crc32, frame, hard_bits, unframe, unframe_either_polarity};

    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn ieee_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// The root cause behind the silence false positive: the all-ones frame
    /// is a *valid* 4-byte frame, because `crc32([0xFF; 4]) == 0xFFFF_FFFF`.
    /// Nothing else up to 4096 bytes collides, which is why only
    /// `payload_len = 4` ever returned a payload for silence.
    #[test]
    fn degenerate_frames_are_valid_only_for_four_bytes() {
        let ones: Vec<usize> = (1..=4096)
            .filter(|&len| unframe(&vec![1u8; len * 8 + 32], len).is_some())
            .collect();
        assert_eq!(ones, [4]);
        assert_eq!(unframe(&[1u8; 64], 4), Some(vec![0xFF; 4]));
        let zeros: Vec<usize> = (1..=4096)
            .filter(|&len| unframe(&vec![0u8; len * 8 + 32], len).is_some())
            .collect();
        assert!(zeros.is_empty(), "{zeros:?}");
    }

    #[test]
    fn hard_bits_turns_ties_and_non_finite_votes_into_erasures() {
        assert_eq!(
            hard_bits(&[0.5, -0.5, 1e-300, -1e-300]),
            Some(vec![1, 0, 1, 0])
        );
        assert_eq!(hard_bits(&[0.5, 0.0, -0.5]), None);
        assert_eq!(hard_bits(&[-0.0]), None);
        assert_eq!(hard_bits(&[1.0, f64::NAN]), None);
        assert_eq!(hard_bits(&[1.0, f64::INFINITY]), None);
        assert_eq!(hard_bits(&[f64::NEG_INFINITY]), None);
        assert_eq!(hard_bits(&[0.0; 64]), None);
    }

    #[test]
    fn either_polarity_accepts_a_frame_and_its_complement() {
        let payload = [0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let bits = frame(&payload);
        let inverted: Vec<u8> = bits.iter().map(|&b| b ^ 1).collect();
        assert_eq!(
            unframe_either_polarity(&bits, 8).as_deref(),
            Some(&payload[..])
        );
        assert_eq!(
            unframe_either_polarity(&inverted, 8).as_deref(),
            Some(&payload[..])
        );
        let mut broken = bits.clone();
        broken[3] ^= 1;
        assert_eq!(unframe_either_polarity(&broken, 8), None);
    }
}
