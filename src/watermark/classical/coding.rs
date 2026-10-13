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
    use super::crc32;

    #[test]
    fn ieee_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }
}
