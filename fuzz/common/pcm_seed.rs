// Compact seed-driven PCM synthesis for libFuzzer targets.
//
// Arbitrary `Vec<f32>` inputs from `arbitrary` rarely reach algorithm
// minimum lengths under the default 4096-byte `max_len`. Every extraction
// target instead derives deterministic two-tone audio (plus light noise)
// from the raw byte seed, with variable fractional duration between
// `min_samples` and `max_samples`.

/// Wang / Panako minimum at 8 kHz (2 s).
pub const WANG_MIN: usize = 16_000;
/// Haitsma minimum at 5 kHz (2 s).
pub const HAITSMA_MIN: usize = 10_000;

struct Rng(u64);

impl Rng {
    fn from_bytes(data: &[u8]) -> Self {
        let mut s = 0u64;
        for (i, &b) in data.iter().take(8).enumerate() {
            s |= (b as u64) << (i * 8);
        }
        Self(s.max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / ((1u64 << 53) as f32)
    }

    fn bipolar(&mut self) -> f32 {
        self.unit_f32() * 2.0 - 1.0
    }
}

// Synthesize mono PCM between `min_samples` and `max_samples` inclusive.
// Duration includes a fractional component so fuzzing exercises remainders
// that trip streaming bucket boundaries (not only whole-second lengths).
pub fn synth_pcm(data: &[u8], sr: u32, min_samples: usize, max_samples: usize) -> Vec<f32> {
    assert!(min_samples <= max_samples);
    let mut rng = Rng::from_bytes(data);
    let span = max_samples - min_samples;
    // 0..999 ms fractional tail on top of the minimum length.
    let frac_ms = (rng.next() % 1000) as f32 / 1000.0;
    let extra = ((span as f32) * frac_ms) as usize;
    let len = min_samples + extra;

    let f1 = 440.0 + (rng.next() % 400) as f32;
    let f2 = 880.0 + (rng.next() % 400) as f32;
    let sr_f = sr as f32;

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let t = i as f32 / sr_f;
        let tone = 0.45 * (2.0 * std::f32::consts::PI * f1 * t).sin()
            + 0.35 * (2.0 * std::f32::consts::PI * f2 * t).sin();
        out.push((tone + rng.bipolar() * 0.02).clamp(-1.0, 1.0));
    }
    out
}

// Derive a streaming chunk size from fuzz bytes (always >= 1, bounded).
pub fn chunk_size(data: &[u8], max_chunk: usize) -> usize {
    let b = data.get(8).copied().unwrap_or(1);
    (b as usize).max(1).min(max_chunk)
}
