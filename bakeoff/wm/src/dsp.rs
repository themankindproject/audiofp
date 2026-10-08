//! Core DSP for the watermark bakeoff: STFT / inverse-STFT with
//! overlap-add, and the keyed PRNG. Dev-only, no dependency on audiofp so
//! candidate algorithms can be prototyped without touching the crate.

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner, FftPlannerScalar};
use std::sync::Arc;

/// Analysis window (periodic Hann, matching audiofp's convention).
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Forward STFT: real frames -> complex spectra.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Vec<f32>,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let fft = FftPlannerScalar::new().plan_fft_forward(n_fft);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        Self {
            n_fft,
            hop,
            fft,
            win: hann(n_fft),
            buf: vec![Complex::new(0.0, 0.0); n_fft],
            scratch,
        }
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
    pub fn hop(&self) -> usize {
        self.hop
    }
    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Number of frames for `n_samples` with centred padding.
    pub fn n_frames(&self, n_samples: usize) -> usize {
        if n_samples == 0 {
            return 0;
        }
        1 + n_samples / self.hop
    }

    /// Analyse the whole buffer into a flat `(n_frames, n_bins)` complex
    /// spectrogram (centred, Hann-windowed).
    pub fn analyze(&mut self, samples: &[f32]) -> (Vec<Complex<f32>>, usize) {
        let n_frames = self.n_frames(samples.len());
        let nb = self.n_bins();
        let n = samples.len() as isize;
        let center = (self.n_fft / 2) as isize;
        let mut out = vec![Complex::new(0.0, 0.0); n_frames * nb];
        for f in 0..n_frames {
            let start = f as isize * self.hop as isize - center;
            for k in 0..self.n_fft {
                let idx = start + k as isize;
                let s = if idx < 0 || idx >= n {
                    0.0
                } else {
                    samples[idx as usize]
                };
                self.buf[k] = Complex::new(s * self.win[k], 0.0);
            }
            self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
            out[f * nb..(f + 1) * nb].copy_from_slice(&self.buf[..nb]);
        }
        (out, n_frames)
    }
}

/// Inverse STFT: complex spectra -> real samples via weighted overlap-add.
pub struct Istft {
    n_fft: usize,
    hop: usize,
    fft: Arc<dyn Fft<f32>>,
    win: Vec<f32>,
    buf: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
}

impl Istft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let fft = FftPlannerScalar::new().plan_fft_inverse(n_fft);
        let scratch = vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()];
        Self {
            n_fft,
            hop,
            fft,
            win: hann(n_fft),
            buf: vec![Complex::new(0.0, 0.0); n_fft],
            scratch,
        }
    }

    /// Reconstruct `n_samples` samples from a flat `(n_frames, n_bins)`
    /// spectrogram produced by [`Stft::analyze`].
    pub fn synthesize(
        &mut self,
        spec: &[Complex<f32>],
        n_frames: usize,
        n_samples: usize,
    ) -> Vec<f32> {
        let nb = self.n_fft / 2 + 1;
        let center = (self.n_fft / 2) as isize;
        let total = n_samples as isize + 2 * center;
        let mut acc = vec![0.0f64; total as usize];
        let mut wsum = vec![0.0f64; total as usize];
        let scale = 1.0 / self.n_fft as f32;

        for f in 0..n_frames {
            // Build the conjugate-symmetric spectrum.
            for k in 0..nb {
                self.buf[k] = spec[f * nb + k];
            }
            for k in nb..self.n_fft {
                self.buf[k] = spec[f * nb + (self.n_fft - k)].conj();
            }
            self.fft.process_with_scratch(&mut self.buf, &mut self.scratch);
            let start = f as isize * self.hop as isize;
            for k in 0..self.n_fft {
                let p = start + k as isize;
                if p < 0 || p >= total {
                    continue;
                }
                let w = self.win[k] as f64;
                acc[p as usize] += self.buf[k].re as f64 * scale as f64 * w;
                wsum[p as usize] += w * w;
            }
        }
        let mut out = vec![0.0f32; n_samples];
        for (i, o) in out.iter_mut().enumerate() {
            let p = (i as isize + center) as usize;
            if wsum[p] > 1e-8 {
                *o = (acc[p] / wsum[p]) as f32;
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Keyed PRNG: a small ChaCha20-based keystream. Deterministic across
// platforms (integer-only core), keyed by an arbitrary byte slice.
// ---------------------------------------------------------------------------

pub struct ChaCha {
    state: [u32; 16],
    block: [u8; 64],
    idx: usize,
}

impl ChaCha {
    pub fn new(key: &[u8], nonce: u64) -> Self {
        // 32-byte key expansion (SHA-ish fold for arbitrary key length).
        let mut k = [0u32; 8];
        let mut acc = 0x9E3779B9u32;
        for (i, &b) in key.iter().enumerate() {
            let j = i % 8;
            acc = acc.rotate_left(5) ^ (b as u32).wrapping_mul(0x01000193);
            k[j] = k[j].wrapping_add(acc ^ (i as u32).wrapping_mul(0x85EBCA6B));
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
        for i in 0..16 {
            let v = x[i].wrapping_add(self.state[i]);
            self.block[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.state[12] = self.state[12].wrapping_add(1);
        if self.state[12] == 0 {
            self.state[13] = self.state[13].wrapping_add(1);
        }
        self.idx = 0;
    }

    pub fn u32(&mut self) -> u32 {
        if self.idx >= 64 {
            self.fill_block();
        }
        let b = &self.block[self.idx..self.idx + 4];
        self.idx += 4;
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Uniform f32 in [-1, 1).
    pub fn bipolar(&mut self) -> f32 {
        // 24-bit mantissa worth of randomness for exact f32.
        let x = (self.u32() >> 8) as f32 / (1u32 << 23) as f32; // [0,2)
        x - 1.0
    }

    /// Uniform f32 in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.u32() >> 8) as f32 / (1u32 << 24) as f32
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

/// Derive a subkey/stream from a master key and a domain tag.
pub fn derive_key(master: &[u8], tag: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(master.len() + tag.len());
    out.extend_from_slice(master);
    out.extend_from_slice(&[0u8]);
    out.extend_from_slice(tag);
    out
}

/// Log-magnitude compression helper used by QIM / masking.
pub fn to_db(p: f32) -> f32 {
    10.0 * libm::log10f(p.max(1e-12))
}
