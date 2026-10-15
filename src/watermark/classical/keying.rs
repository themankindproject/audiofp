//! ChaCha20 keystream and keyed bin permutation.
//!
//! Integer-only, so the chip streams match the bakeoff on every target.
//! Domain tags are part of the measured algorithms and stay byte-for-byte.

use alloc::vec::Vec;

/// ChaCha20-based keystream. Deterministic across platforms.
pub(crate) struct ChaCha {
    state: [u32; 16],
    block: [u8; 64],
    idx: usize,
}

impl ChaCha {
    pub(crate) fn new(key: &[u8], nonce: u64) -> Self {
        let mut k = [0u32; 8];
        let mut acc = 0x9E37_79B9u32;
        for (i, &b) in key.iter().enumerate() {
            let j = i % 8;
            acc = acc.rotate_left(5) ^ (b as u32).wrapping_mul(0x0100_0193);
            k[j] = k[j].wrapping_add(acc ^ (i as u32).wrapping_mul(0x85EB_CA6B));
        }
        let mut state = [0u32; 16];
        state[0] = 0x6170_7865;
        state[1] = 0x3320_646e;
        state[2] = 0x7962_2d32;
        state[3] = 0x6b20_6574;
        state[4..12].copy_from_slice(&k);
        state[12] = 0;
        state[13] = 0;
        state[14] = nonce as u32;
        state[15] = (nonce >> 32) as u32;
        let mut s = Self {
            state,
            block: [0u8; 64],
            idx: 64,
        };
        s.fill_block();
        s
    }

    fn fill_block(&mut self) {
        let mut x = self.state;
        for _ in 0..10 {
            quarter(&mut x, 0, 4, 8, 12);
            quarter(&mut x, 1, 5, 9, 13);
            quarter(&mut x, 2, 6, 10, 14);
            quarter(&mut x, 3, 7, 11, 15);
            quarter(&mut x, 0, 5, 10, 15);
            quarter(&mut x, 1, 6, 11, 12);
            quarter(&mut x, 2, 7, 8, 13);
            quarter(&mut x, 3, 4, 9, 14);
        }
        for (i, (word, state)) in x.iter().zip(self.state).enumerate() {
            let v = word.wrapping_add(state);
            self.block[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }
        self.idx = 0;
    }

    pub(crate) fn u32(&mut self) -> u32 {
        if self.idx >= 64 {
            self.fill_block();
        }
        let b = &self.block[self.idx..self.idx + 4];
        self.idx += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Uniform `f32` in [-1, 1). Used by tests to build a deterministic host.
    #[cfg(test)]
    pub(crate) fn bipolar(&mut self) -> f32 {
        let x = (self.u32() >> 8) as f32 / (1u32 << 23) as f32;
        x - 1.0
    }
}

fn quarter(x: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(16);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(12);
    x[a] = x[a].wrapping_add(x[b]);
    x[d] = (x[d] ^ x[a]).rotate_left(8);
    x[c] = x[c].wrapping_add(x[d]);
    x[b] = (x[b] ^ x[c]).rotate_left(7);
}

/// Derive a subkey from a master key and a domain tag.
pub(crate) fn derive_key(master: &[u8], tag: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(master.len() + tag.len() + 1);
    out.extend_from_slice(master);
    out.push(0);
    out.extend_from_slice(tag);
    out
}

/// Keyed permutation of `lo..hi`, truncated to `count` entries.
pub(crate) fn keyed_bins(key: &[u8], tag: &[u8], lo: usize, hi: usize, count: usize) -> Vec<usize> {
    let mut rng = ChaCha::new(&derive_key(key, tag), 0);
    let mut order: Vec<usize> = (lo..hi).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    order.truncate(count);
    order
}
