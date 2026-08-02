//! Realistic deterministic audio synthesis for test fixtures (`pub(crate)`).
//!
//! All generators are fully deterministic (seeded xorshift) and produce
//! representative audio without requiring external CC0 files. This lets
//! the codec round-trip and pipeline tests run in CI without downloads.
#![allow(dead_code)]
//!
//! # Generators
//!
//! | Function | Style | Duration | Sample Rate |
//! |---|---|---|---|
//! | [`multi_instrument`] | Bass + chords + melody with ADSR, reverb | param | 48 kHz |
//! | [`speech_like`] | Glottal pulse + formant synthesis | param | 16 kHz |
//! | [`ambient_pad`] | Filtered noise + evolving sine clusters | param | 48 kHz |
//! | [`percussion`] | Kick, snare, hi-hat pattern | param | 48 kHz |
//!
//! Resamplers: [`resample_48k_to_8k`], [`resample_48k_to_5k`], [`resample_48k_to_16k`].

use std::f32::consts::PI;

const SR48K: u32 = 48_000;
const SR8K: u32 = 8_000;
const SR5K: u32 = 5_000;
const SR16K: u32 = 16_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f32(&mut self) -> f32 {
        (self.next() as i64 as f32) / (i64::MAX as f32)
    }

    fn f32_bipolar(&mut self) -> f32 {
        self.f32() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// ADSR envelope
// ---------------------------------------------------------------------------

struct Envelope {
    attack: usize,
    decay: usize,
    sustain: f32,
    release: usize,
    total: usize,
}

impl Envelope {
    fn value(&self, t: usize) -> f32 {
        if t < self.attack {
            t as f32 / self.attack as f32
        } else if t < self.attack + self.decay {
            let dt = (t - self.attack) as f32 / self.decay as f32;
            1.0 - (1.0 - self.sustain) * dt
        } else if t < self.total - self.release {
            self.sustain
        } else {
            let dt = (t - (self.total - self.release)) as f32 / self.release as f32;
            self.sustain * (1.0 - dt)
        }
    }
}

// ---------------------------------------------------------------------------
// Band-limited oscillator (additive sine with harmonics, no aliasing)
// ---------------------------------------------------------------------------

struct Oscillator {
    phase: f32,
    freq: f32,
    harmonics: Vec<(f32, f32)>, // (amplitude, phase_offset)
}

impl Oscillator {
    fn new(rng: &mut Rng, base_freq: f32, num_harmonics: usize) -> Self {
        let mut harmonics = Vec::with_capacity(num_harmonics);
        for i in 1..=num_harmonics {
            let amp = 1.0 / i as f32 * (0.6 + rng.f32() * 0.8);
            harmonics.push((amp, rng.f32() * PI * 2.0));
        }
        Self {
            phase: 0.0,
            freq: base_freq,
            harmonics,
        }
    }

    fn sample(&mut self, sr: f32) -> f32 {
        let mut out = 0.0;
        for (i, &(amp, phase_off)) in self.harmonics.iter().enumerate() {
            let h = (i + 1) as f32;
            out += amp * (self.phase * h + phase_off).sin();
        }
        self.phase += 2.0 * PI * self.freq / sr;
        if self.phase > 2.0 * PI {
            self.phase -= 2.0 * PI;
        }
        out * 0.3 // prevent clipping from stacked harmonics
    }
}

// ---------------------------------------------------------------------------
// Public generators
// ---------------------------------------------------------------------------

/// A realistic multi-instrument piece: bass, chords, melody with ADSR
/// envelopes, harmonic structure, and light reverb.
///
/// ~12 seconds, 48 kHz, deterministic from `seed`.
pub fn multi_instrument(seed: u64, secs: f32) -> Vec<f32> {
    let sr = SR48K as f32;
    let n = (sr * secs) as usize;
    let mut rng = Rng(seed.max(1));
    let mut out = vec![0.0f32; n];

    // Chord progression: I → IV → V → I at given bpm
    let bpm = 120.0;
    let beat_samples = (sr * 60.0 / bpm) as usize;
    let bar_samples = beat_samples * 4;
    let bass_notes = [55.0, 73.42, 82.41, 55.0]; // A2, D3, E3, A2
    let chord_roots = [110.0, 146.83, 164.81, 110.0]; // A2, D3, E3, A2
    let chord_thirds = [138.59, 185.00, 207.65, 138.59]; // C#3, F#3, G#3, C#3
    let chord_fifths = [164.81, 220.0, 246.94, 164.81]; // E3, A3, B3, E3
    let melody_notes = [220.0, 277.18, 329.63, 293.66, 246.94, 220.0, 196.0, 220.0];

    let mut bass_osc = Oscillator::new(&mut rng, bass_notes[0], 4);
    let mut chord1 = Oscillator::new(&mut rng, chord_roots[0], 3);
    let mut chord2 = Oscillator::new(&mut rng, chord_thirds[0], 3);
    let mut chord3 = Oscillator::new(&mut rng, chord_fifths[0], 3);
    let mut melody = Oscillator::new(&mut rng, melody_notes[0], 5);

    let mut delay_buf = vec![0.0f32; (0.12 * sr) as usize];
    let mut delay_pos = 0usize;

    for (i, sample) in out.iter_mut().enumerate() {
        let bar = i / bar_samples;
        let _bar_frac = (i % bar_samples) as f32 / bar_samples as f32;

        // Update note on bar boundaries
        if i % bar_samples == 0 && bar < 4 {
            bass_osc.freq = bass_notes[bar % 4];
            chord1.freq = chord_roots[bar % 4];
            chord2.freq = chord_thirds[bar % 4];
            chord3.freq = chord_fifths[bar % 4];
        }
        if i % beat_samples == 0 {
            melody.freq = melody_notes[(i / beat_samples) % melody_notes.len()];
        }

        let bass_env = envelope_value(bar_samples, 0.01, 0.05, 0.6, 0.1, i % bar_samples);
        let chord_env = envelope_value(bar_samples, 0.02, 0.1, 0.4, 0.2, i % bar_samples);
        let mel_env = envelope_value(beat_samples, 0.005, 0.15, 0.3, 0.08, i % beat_samples);

        let s = bass_osc.sample(sr) * bass_env * 0.6
            + chord1.sample(sr) * chord_env * 0.25
            + chord2.sample(sr) * chord_env * 0.2
            + chord3.sample(sr) * chord_env * 0.15
            + melody.sample(sr) * mel_env * 0.35
            + rng.f32_bipolar() * 0.003; // noise floor

        // Simple reverb: feedback delay
        let delayed = delay_buf[delay_pos];
        delay_buf[delay_pos] = s + delayed * 0.3;
        delay_pos = (delay_pos + 1) % delay_buf.len();

        *sample = (s + delayed * 0.3).clamp(-1.0, 1.0);
    }

    out
}

/// Speech-like formant synthesis: vowel-like resonances modulated
/// by a glottal pulse train with formant shifting.
pub fn speech_like(seed: u64, secs: f32) -> Vec<f32> {
    let sr = SR16K as f32;
    let n = (sr * secs) as usize;
    let mut rng = Rng(seed.max(1));
    let mut out = vec![0.0f32; n];

    let f0 = 120.0; // typical male fundamental
    let pulse_period = (sr / f0) as usize;

    // Formant frequencies (F1, F2, F3 for /a/ like "father")
    let f1 = 730.0;
    let f2 = 1090.0;
    let f3 = 2440.0;
    let bw1 = 50.0;
    let bw2 = 70.0;
    let bw3 = 110.0;

    // Formant filter state (2nd-order resonators)
    let mut s1 = [0.0f32; 2];
    let mut s2 = [0.0f32; 2];
    let mut s3 = [0.0f32; 2];

    let glottal_shape = |t: f32| -> f32 {
        if !(0.0..=1.0).contains(&t) {
            return 0.0;
        }
        // LF model glottal pulse approximation
        let open = (t * PI).sin().powf(2.0);
        let close = if t > 0.7 {
            (-(t - 0.7) * 20.0).exp()
        } else {
            0.0
        };
        open * 0.8 + close
    };

    for (i, sample) in out.iter_mut().enumerate() {
        let t_in_period = (i % pulse_period) as f32 / pulse_period as f32;
        let pulse = glottal_shape(t_in_period);

        let progress = i as f32 / n as f32;
        let modulated_f1 = f1 * (1.0 - progress * 0.5);
        let modulated_f2 = f2 * (1.0 + progress * 0.8);
        let modulated_f3 = f3 * (1.0 + progress * 0.1);

        let r1 = (-PI * bw1 / sr).exp();
        let r2 = (-PI * bw2 / sr).exp();
        let r3 = (-PI * bw3 / sr).exp();

        let theta1 = 2.0 * PI * modulated_f1 / sr;
        let theta2 = 2.0 * PI * modulated_f2 / sr;
        let theta3 = 2.0 * PI * modulated_f3 / sr;

        let res1 = resonator(pulse, &mut s1, r1, theta1);
        let res2 = resonator(pulse, &mut s2, r2, theta2);
        let res3 = resonator(pulse, &mut s3, r3, theta3);

        *sample =
            (res1 * 0.5 + res2 * 0.3 + res3 * 0.15 + rng.f32_bipolar() * 0.002).clamp(-1.0, 1.0);
    }

    out
}

/// Ambient / pad: slow-attack filtered noise + evolving sine clusters.
pub fn ambient_pad(seed: u64, secs: f32) -> Vec<f32> {
    let sr = SR48K as f32;
    let n = (sr * secs) as usize;
    let mut rng = Rng(seed.max(1));
    let mut out = vec![0.0f32; n];

    // Filtered noise state
    let mut lp = 0.0f32;
    let alpha = 0.01; // ~80 Hz cutoff @ 48 kHz

    let mut osc1 = Oscillator::new(&mut rng, 130.81, 6); // C3
    let mut osc2 = Oscillator::new(&mut rng, 196.00, 6); // G3
    let mut osc3 = Oscillator::new(&mut rng, 261.63, 8); // C4

    for (i, sample) in out.iter_mut().enumerate() {
        let t = i as f32 / sr;

        // Slow LFO on oscillator frequencies
        let lfo = (t * 0.15).sin();
        osc1.freq = 130.81 + lfo * 2.0;
        osc2.freq = 196.00 + lfo * 1.5;
        osc3.freq = 261.63 + lfo * 3.0;

        let env = (t * 0.3).min(1.0); // slow fade-in

        // Filtered noise
        let noise = rng.f32_bipolar() * 0.15;
        lp += alpha * (noise - lp);

        let s =
            (osc1.sample(sr) * 0.35 + osc2.sample(sr) * 0.3 + osc3.sample(sr) * 0.25 + lp * 0.4)
                * env;

        *sample = s.clamp(-1.0, 1.0);
    }

    out
}

/// Percussive / transient-rich loop: hi-hat-like noise bursts, kick drum,
/// snare-like tonal hits. Good for testing landmark density.
pub fn percussion(seed: u64, secs: f32) -> Vec<f32> {
    let sr = SR48K as f32;
    let n = (sr * secs) as usize;
    let mut rng = Rng(seed.max(1));
    let mut out = vec![0.0f32; n];

    let bpm = 140.0;
    let beat_samples = (sr * 60.0 / bpm) as usize;

    for (i, sample) in out.iter_mut().enumerate() {
        let beat_pos = i % beat_samples;
        let beat_frac = beat_pos as f32 / beat_samples as f32;

        // Kick on every beat
        let kick = if beat_frac < 0.12 {
            let env = 1.0 - beat_frac / 0.12;
            let freq = 150.0 * (1.0 - beat_frac * 5.0).max(0.3);
            (2.0 * PI * freq * beat_frac * 50.0).sin() * env * 0.7
        } else {
            0.0
        };

        // Snare on beats 2 and 4
        let snare = if (i / beat_samples) % 4 == 1 || (i / beat_samples) % 4 == 3 {
            if beat_frac < 0.08 {
                let env = 1.0 - beat_frac / 0.08;
                let tone = (2.0 * PI * 200.0 * beat_frac * 40.0).sin();
                let noise = rng.f32_bipolar() * 0.3;
                (tone * 0.5 + noise) * env * 0.4
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Hi-hat on 8th notes
        let hat = if beat_pos % (beat_samples / 2) < (beat_samples / 16) {
            rng.f32_bipolar()
                * 0.15
                * (1.0 - (beat_pos % (beat_samples / 2)) as f32 / (beat_samples / 16) as f32)
        } else {
            0.0
        };

        *sample = (kick + snare + hat).clamp(-1.0, 1.0);
    }

    out
}

/// Resample a 48 kHz signal to 8 kHz for Wang/Panako.
pub fn resample_48k_to_8k(samples: &[f32]) -> Vec<f32> {
    let ratio = SR48K as f32 / SR8K as f32;
    let out_len = (samples.len() as f32 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = (i as f32 * ratio) as usize;
        let frac = i as f32 * ratio - src as f32;
        let a = samples[src.min(samples.len() - 1)];
        let b = samples[(src + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Resample 48 kHz to 5 kHz for Haitsma.
pub fn resample_48k_to_5k(samples: &[f32]) -> Vec<f32> {
    let ratio = SR48K as f32 / SR5K as f32;
    let out_len = (samples.len() as f32 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = (i as f32 * ratio) as usize;
        let frac = i as f32 * ratio - src as f32;
        let a = samples[src.min(samples.len() - 1)];
        let b = samples[(src + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Resample 48 kHz to 16 kHz for Neural.
pub fn resample_48k_to_16k(samples: &[f32]) -> Vec<f32> {
    let ratio = SR48K as f32 / SR16K as f32;
    let out_len = (samples.len() as f32 / ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = (i as f32 * ratio) as usize;
        let frac = i as f32 * ratio - src as f32;
        let a = samples[src.min(samples.len() - 1)];
        let b = samples[(src + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn envelope_value(
    total: usize,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    t: usize,
) -> f32 {
    let atk = (total as f32 * attack) as usize;
    let dcy = (total as f32 * decay) as usize;
    let rel = (total as f32 * release) as usize;

    if t < atk {
        t as f32 / atk as f32
    } else if t < atk + dcy {
        let dt = (t - atk) as f32 / dcy as f32;
        1.0 - (1.0 - sustain) * dt
    } else if t < total - rel {
        sustain
    } else if total > rel {
        let dt = (t - (total - rel)) as f32 / rel as f32;
        sustain * (1.0 - dt)
    } else {
        sustain
    }
}

fn resonator(x: f32, state: &mut [f32; 2], r: f32, theta: f32) -> f32 {
    let y = x + 2.0 * r * theta.cos() * state[0] - r * r * state[1];
    state[1] = state[0];
    state[0] = y;
    y * (1.0 - r * r) // normalize gain
}
