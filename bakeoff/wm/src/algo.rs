//! Candidate classical watermarking algorithms for the bakeoff.
//!
//! Each candidate exposes a uniform `embed` / `detect` interface so the same
//! attack matrix can be run against all of them. Payload is carried as a
//! `Vec<u8>` of bits; the detector is blind (needs only the secret key).

use crate::dsp::{ChaCha, Istft, Stft, derive_key};
use rustfft::num_complex::Complex;

/// Uniform interface for a classical candidate.
pub trait Candidate {
    fn name(&self) -> &'static str;
    /// Embed `bits` (one payload bit per byte, values 0/1) into `samples`,
    /// returning the watermarked signal and a nominal embedding SNR (dB).
    fn embed(&mut self, samples: &[f32], bits: &[u8], key: &[u8]) -> (Vec<f32>, f64);
    /// Blind detection: return the recovered payload bits and a
    /// detection score (higher = stronger evidence the key is present).
    /// The score is calibrated later; raw correlation/normalised statistic.
    fn detect(&mut self, samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64);
    /// Distortion (RMS) helper target.
    fn sample_rate(&self) -> u32 {
        16_000
    }
}

// ===========================================================================
// WINNING CANDIDATE: magnitude-domain DSSS with per-bin host-mean removal.
//
// Each STFT frame carries one chip per keyed bin, and the payload bit index
// cycles at the frame rate (chip rate = frame rate). Embedding modulates the
// complex coefficient's magnitude along its own phase by `sign * chip *
// alpha * mag`, so it is relative (gain-invariant) and preserves phase
// (codec-friendly). Blind detection removes the host by subtracting each
// bin's long-term mean magnitude over the clip; the chip sequence is
// zero-mean so it averages out, leaving the accumulated watermark residual.
// ===========================================================================

pub struct MagDsss {
    pub n_fft: usize,
    pub hop: usize,
    /// Embedding strength as a fraction of local bin magnitude.
    pub alpha: f32,
    pub bin_lo: usize,
    pub bin_hi_frac: f32,
    pub bins_per_frame: usize,
    /// When true the detector searches all bit phases and picks the
    /// strongest, which makes detection independent of the window start
    /// (leading trim / arbitrary offset / mid-extract).
    pub sync: bool,
}

impl MagDsss {
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop: 512,
            alpha: 0.35,
            bin_lo: 10,
            bin_hi_frac: 0.75,
            bins_per_frame: 300,
            sync: true,
        }
    }

    fn order(&self, key: &[u8]) -> Vec<usize> {
        let nb = self.n_fft / 2 + 1;
        let bin_hi = ((nb as f32) * self.bin_hi_frac) as usize;
        keyed_bins(key, b"mag-bins", self.bin_lo, bin_hi, self.bins_per_frame)
    }
}

impl Default for MagDsss {
    fn default() -> Self { Self::new() }
}

impl Candidate for MagDsss {
    fn name(&self) -> &'static str { "mag_dsss" }

    fn embed(&mut self, samples: &[f32], bits: &[u8], key: &[u8]) -> (Vec<f32>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (mut spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let order = self.order(key);
        let nb_bits = bits.len().max(1);

        let mut total_mod = 0.0f64;
        for f in 0..n_frames {
            let b = f % nb_bits;
            let sign = if bits[b] == 0 { -1.0f32 } else { 1.0f32 };
            let mut r = ChaCha::new(&derive_key(key, b"mag-spread"), b as u64);
            for &bin in order.iter() {
                let chip = r.bipolar();
                let cell = &mut spec[f * nb + bin];
                let mag = cell.norm();
                let delta = sign * chip * self.alpha * mag;
                let phase = cell.arg();
                *cell = Complex::from_polar((mag + delta).max(1e-7), phase);
                total_mod += (delta as f64) * (delta as f64);
            }
        }
        let total_ref: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let mut istft = Istft::new(self.n_fft, self.hop);
        let out = istft.synthesize(&spec, n_frames, samples.len());
        let snr = if total_mod > 0.0 { 10.0 * (total_ref / total_mod).log10() } else { f64::INFINITY };
        (out, snr)
    }

    fn detect(&mut self, samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let order = self.order(key);
        if n_frames == 0 {
            return (vec![0; n_bits], 0.0);
        }

        // Per-bin long-term mean magnitude = host estimate. The chip sequence
        // is zero-mean, so it averages out over the clip and what remains is
        // the accumulated watermark residual.
        let mut bin_mean = vec![0.0f64; nb];
        for f in 0..n_frames {
            for &bin in order.iter() {
                bin_mean[bin] += spec[f * nb + bin].norm() as f64;
            }
        }
        let inv_nf = 1.0 / n_frames as f64;
        for m in bin_mean.iter_mut() {
            *m *= inv_nf;
        }

        // Correlation for one trial bit-phase `ph0`: frame f carries bit
        // `(f + ph0) % n_bits`.
        let correlate = |ph0: usize| -> Vec<f64> {
            let mut corr = vec![0.0f64; n_bits];
            for f in 0..n_frames {
                let b = (f + ph0) % n_bits;
                let mut r = ChaCha::new(&derive_key(key, b"mag-spread"), b as u64);
                for &bin in order.iter() {
                    let chip = r.bipolar() as f64;
                    let resid = spec[f * nb + bin].norm() as f64 - bin_mean[bin];
                    corr[b] += (resid / bin_mean[bin].max(1e-9)) * chip;
                }
            }
            corr
        };

        let (corr, phase) = if self.sync {
            // Search every phase; the correct one accumulates coherently and
            // has by far the largest total correlation energy.
            let mut best = correlate(0);
            let mut best_e: f64 = best.iter().map(|c| c.abs()).sum();
            for ph0 in 1..n_bits {
                let c = correlate(ph0);
                let e: f64 = c.iter().map(|x| x.abs()).sum();
                if e > best_e {
                    best_e = e;
                    best = c;
                }
            }
            (best, 0)
        } else {
            (correlate(0), 0)
        };
        let _ = phase;

        let bits: Vec<u8> = corr.iter().map(|&c| if c >= 0.0 { 1 } else { 0 }).collect();
        // Detection statistic: mean |normalised correlation| per bit. Under
        // the correct key this is large; under a wrong key / no watermark it
        // collapses to the noise floor.
        let mean = corr.iter().map(|c| c.abs()).sum::<f64>() / corr.len().max(1) as f64;
        (bits, mean)
    }
}

// ===========================================================================
// Candidate B: log-magnitude DSSS (embed in log domain — robust to gain &
// codecs because it targets relative, not absolute, energy)
// ===========================================================================

pub struct LogDsss {
    pub n_fft: usize,
    pub hop: usize,
    pub frames_per_bit: usize,
    /// dB perturbation amplitude.
    pub db_alpha: f32,
    pub bin_lo: usize,
    pub bin_hi_frac: f32,
    pub bins_per_frame: usize,
}

impl LogDsss {
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop: 512,
            frames_per_bit: 32,
            db_alpha: 0.5,
            bin_lo: 8,
            bin_hi_frac: 0.45,
            bins_per_frame: 24,
        }
    }
}

impl Default for LogDsss {
    fn default() -> Self {
        Self::new()
    }
}

impl Candidate for LogDsss {
    fn name(&self) -> &'static str {
        "log_dsss"
    }

    fn embed(&mut self, samples: &[f32], bits: &[u8], key: &[u8]) -> (Vec<f32>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (mut spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let bin_hi = ((nb as f32) * self.bin_hi_frac) as usize;
        let mut bin_rng = ChaCha::new(&derive_key(key, b"dsss-bins"), 0);
        let mut order: Vec<usize> = (self.bin_lo..bin_hi).collect();
        for i in (1..order.len()).rev() {
            let j = (bin_rng.u32() as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut total_mod = 0.0f64;
        for (b, &bit) in bits.iter().enumerate() {
            let s = spread_seq(key, b, self.frames_per_bit * self.bins_per_frame);
            let sign = if bit == 0 { -1.0f32 } else { 1.0f32 };
            let f0 = b * self.frames_per_bit;
            for k in 0..self.frames_per_bit {
                let f = f0 + k;
                if f >= n_frames {
                    break;
                }
                for j in 0..self.bins_per_frame {
                    let bin = order[(k * self.bins_per_frame + j) % order.len()];
                    let cell = &mut spec[f * nb + bin];
                    let mag = cell.norm();
                    if mag < 1e-7 {
                        continue;
                    }
                    // log-domain: mag_new = mag * 10^(delta/20)
                    let idx = k * self.bins_per_frame + j;
                    let db_delta = sign * s[idx] * self.db_alpha;
                    let lin = libm::powf(10.0, db_delta / 20.0);
                    let new_mag = mag * lin;
                    let phase = cell.arg();
                    *cell = Complex::from_polar(new_mag, phase);
                    let dm = (new_mag - mag) as f64;
                    total_mod += dm * dm;
                }
            }
        }
        let total_ref: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let mut istft = Istft::new(self.n_fft, self.hop);
        let out = istft.synthesize(&spec, n_frames, samples.len());
        let snr = if total_mod > 0.0 {
            10.0 * (total_ref / total_mod).log10()
        } else {
            f64::INFINITY
        };
        (out, snr)
    }

    fn detect(&mut self, samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let bin_hi = ((nb as f32) * self.bin_hi_frac) as usize;
        let mut bin_rng = ChaCha::new(&derive_key(key, b"dsss-bins"), 0);
        let mut order: Vec<usize> = (self.bin_lo..bin_hi).collect();
        for i in (1..order.len()).rev() {
            let j = (bin_rng.u32() as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut bits = Vec::with_capacity(n_bits);
        let mut scores = Vec::with_capacity(n_bits);
        for b in 0..n_bits {
            let s = spread_seq(key, b, self.frames_per_bit * self.bins_per_frame);
            let f0 = b * self.frames_per_bit;
            let mut corr = 0.0f64;
            let mut norm = 0.0f64;
            for k in 0..self.frames_per_bit {
                let f = f0 + k;
                if f >= n_frames {
                    break;
                }
                for j in 0..self.bins_per_frame {
                    let bin = order[(k * self.bins_per_frame + j) % order.len()];
                    let mag = spec[f * nb + bin].norm().max(1e-7);
                    let db = libm::log10f(mag);
                    let idx = k * self.bins_per_frame + j;
                    corr += (db as f64) * (s[idx] as f64);
                    norm += 1.0;
                }
            }
            let score = if norm > 0.0 { corr / norm } else { 0.0 };
            bits.push(if score >= 0.0 { 1 } else { 0 });
            scores.push(score.abs());
        }
        let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        (bits, mean)
    }
}

// ===========================================================================
// Candidate C: QIM (quantisation index modulation) in log-magnitude
// ===========================================================================

pub struct Qim {
    pub n_fft: usize,
    pub hop: usize,
    /// Quantisation step in dB (log domain).
    pub q_db: f32,
    pub bin_lo: usize,
    pub bin_hi_frac: f32,
    pub bins_per_bit: usize,
}

impl Qim {
    pub fn new() -> Self {
        Self {
            n_fft: 2048,
            hop: 512,
            q_db: 3.0,
            bin_lo: 8,
            bin_hi_frac: 0.45,
            bins_per_bit: 64,
        }
    }
}

impl Default for Qim {
    fn default() -> Self {
        Self::new()
    }
}

impl Candidate for Qim {
    fn name(&self) -> &'static str {
        "qim_logmag"
    }

    fn embed(&mut self, samples: &[f32], bits: &[u8], key: &[u8]) -> (Vec<f32>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (mut spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let bin_hi = ((nb as f32) * self.bin_hi_frac) as usize;
        let mut bin_rng = ChaCha::new(&derive_key(key, b"qim-bins"), 0);
        let mut order: Vec<usize> = (self.bin_lo..bin_hi).collect();
        for i in (1..order.len()).rev() {
            let j = (bin_rng.u32() as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut total_mod = 0.0f64;
        let q = self.q_db;
        for (b, &bit) in bits.iter().enumerate() {
            for j in 0..self.bins_per_bit {
                // Distribute each bit's bins across frames.
                let flat = b * self.bins_per_bit + j;
                let f = flat / nb.max(1);
                if f >= n_frames {
                    break;
                }
                let bin = order[flat % order.len()];
                let cell = &mut spec[f * nb + bin];
                let mag = cell.norm();
                if mag < 1e-6 {
                    continue;
                }
                let db = libm::log10f(mag) * 20.0; // dB
                // Two interleaved quantiser lattices per bit value.
                let target_parity = if bit == 0 { 0.0 } else { q / 2.0 };
                let k = ((db - target_parity) / q).round();
                let new_db = k * q + target_parity;
                let new_mag = libm::powf(10.0, new_db / 20.0);
                let phase = cell.arg();
                *cell = Complex::from_polar(new_mag.max(1e-7), phase);
                let dm = (new_mag - mag) as f64;
                total_mod += dm * dm;
            }
        }
        let total_ref: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let mut istft = Istft::new(self.n_fft, self.hop);
        let out = istft.synthesize(&spec, n_frames, samples.len());
        let snr = if total_mod > 0.0 {
            10.0 * (total_ref / total_mod).log10()
        } else {
            f64::INFINITY
        };
        (out, snr)
    }

    fn detect(&mut self, samples: &[f32], n_bits: usize, key: &[u8]) -> (Vec<u8>, f64) {
        let mut stft = Stft::new(self.n_fft, self.hop);
        let (spec, n_frames) = stft.analyze(samples);
        let nb = stft.n_bins();
        let bin_hi = ((nb as f32) * self.bin_hi_frac) as usize;
        let mut bin_rng = ChaCha::new(&derive_key(key, b"qim-bins"), 0);
        let mut order: Vec<usize> = (self.bin_lo..bin_hi).collect();
        for i in (1..order.len()).rev() {
            let j = (bin_rng.u32() as usize) % (i + 1);
            order.swap(i, j);
        }
        let q = self.q_db;
        let mut bits = Vec::with_capacity(n_bits);
        let mut scores = Vec::with_capacity(n_bits);
        for b in 0..n_bits {
            let mut acc = 0.0f64;
            let mut n = 0.0f64;
            for j in 0..self.bins_per_bit {
                let flat = b * self.bins_per_bit + j;
                let f = flat / nb.max(1);
                if f >= n_frames {
                    break;
                }
                let bin = order[flat % order.len()];
                let mag = spec[f * nb + bin].norm().max(1e-7);
                let db = libm::log10f(mag) * 20.0;
                // Distance to even vs odd lattice.
                let r0 = (db / q - (db / q).round()).abs();
                let r1 = ((db - q / 2.0) / q - ((db - q / 2.0) / q).round()).abs();
                acc += (r1 - r0) as f64;
                n += 1.0;
            }
            let score = if n > 0.0 { acc / n } else { 0.0 };
            bits.push(if score >= 0.0 { 1 } else { 0 });
            scores.push(score.abs());
        }
        let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        (bits, mean)
    }
}

/// Keyed permutation of the embeddable bins, truncated to `count` entries.
fn keyed_bins(key: &[u8], tag: &[u8], lo: usize, hi: usize, count: usize) -> Vec<usize> {
    let mut rng = ChaCha::new(&derive_key(key, tag), 0);
    let mut order: Vec<usize> = (lo..hi).collect();
    for i in (1..order.len()).rev() {
        let j = (rng.u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    order.truncate(count);
    order
}

/// Keyed bipolar spreading sequence of length `len` for bit `index`.
fn spread_seq(key: &[u8], index: usize, len: usize) -> Vec<f32> {
    let mut rng = ChaCha::new(&derive_key(key, b"spread"), index as u64);
    (0..len).map(|_| rng.bipolar()).collect()
}
