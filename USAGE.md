# audiofp Usage Guide

> Complete API reference and examples for `audiofp`, the Rust audio fingerprinting SDK.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Core API](#core-api)
  - [Fingerprinter trait](#fingerprinter-trait)
  - [StreamingFingerprinter trait](#streamingfingerprinter-trait)
  - [Shared value types](#shared-value-types)
- [Classical Fingerprinters](#classical-fingerprinters)
  - [Wang (landmark pairs)](#wang-landmark-pairs)
  - [Panako (triplet hashes)](#panako-triplet-hashes)
  - [Haitsma–Kalker (band-power sign bits)](#haitsmakalker-band-power-sign-bits)
- [Streaming Fingerprinters](#streaming-fingerprinters)
- [Audio File Decoding](#audio-file-decoding)
- [Watermark Detection](#watermark-detection)
- [DSP Primitives](#dsp-primitives)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)
- [Feature Flags](#feature-flags)
- [no_std / Embedded](#no_std--embedded)

---

## Quick Start

Add the dependency:

```toml
[dependencies]
audiofp = "0.2"
```

### Basic example: fingerprint an MP3 with Wang

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode to mono f32 at the rate Wang requires.
    let samples = decode_to_mono_at("song.mp3", 8_000)?;

    let mut wang = Wang::default();
    let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
    let fp = wang.extract(buf)?;

    println!("{} hashes, {:.1} fps", fp.hashes.len(), fp.frames_per_sec);
    for h in fp.hashes.iter().take(5) {
        println!("  t_anchor={} hash={:08x}", h.t_anchor, h.hash);
    }
    Ok(())
}
```

### Detect duplicate songs across re-encodings

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};
use std::collections::HashSet;

fn fingerprint(path: &str) -> Result<HashSet<u32>, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    let mut wang = Wang::default();
    let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
    Ok(wang.extract(buf)?.hashes.into_iter().map(|h| h.hash).collect())
}

let original = fingerprint("song.flac")?;
let mp3 = fingerprint("song_128kbps.mp3")?;
let overlap = original.intersection(&mp3).count();
let pct = 100.0 * overlap as f64 / original.len().max(mp3.len()) as f64;
println!("{overlap} hashes shared ({pct:.1} %)");
```

---

## Core Concepts

### What is an audio fingerprint?

A **perceptual hash** of an audio recording — small enough to store and search at scale, yet stable across re-encoding, modest noise, and (for some algorithms) tempo or pitch changes. Two recordings of the same song will share many hashes; two unrelated recordings won't.

`audiofp` ships three classical fingerprinters, each making different tradeoffs:

| Algorithm  | Output          | Sample rate | Frame rate  | Storage / sec       | When to use                          |
| ---------- | --------------- | ----------- | ----------- | ------------------- | ------------------------------------ |
| `Wang`     | Landmark pairs  | 8 kHz       | 62.5 fps    | ~2.4 KB (fan-out 10)| Music ID; "Shazam-style" matching    |
| `Panako`   | Triplet hashes  | 8 kHz       | 62.5 fps    | ~2.0 KB (fan-out 5) | Tempo-robust music ID (±5 % stretch) |
| `Haitsma`  | 32-bit/frame    | 5 kHz       | 78.125 fps  | 312 B               | Compact dense IDs; fastest extraction|

All three:
- accept mono `f32` PCM in `[-1.0, 1.0]`
- **require** their native sample rate (resample first if your source differs — see [Audio File Decoding](#audio-file-decoding))
- need at least **2 seconds** of audio
- produce hash structs that are `bytemuck::Pod` — castable directly to bytes for storage / IPC

### Indexing is out of scope

`audiofp` extracts fingerprints. **Storage, ANN search, and scoring are the caller's responsibility.** A typical pipeline is:

1. `audiofp` → `Vec<WangHash>` per song
2. Your indexer (e.g. RocksDB, FAISS, custom hash table) → "songs that share hash X at offset Y"
3. Your scorer → "song A has 47 same-offset matches with query, song B has 3 → A wins"

---

## Core API

### `Fingerprinter` trait

Offline (whole-buffer) extraction. Implementors are stateful only insofar as they may reuse scratch buffers — `extract(a)` does not depend on any previous call.

```rust
pub trait Fingerprinter {
    type Output;
    type Config: Clone + Send + Sync;

    fn name(&self) -> &'static str;
    fn config(&self) -> &Self::Config;
    fn required_sample_rate(&self) -> u32;
    fn min_samples(&self) -> usize;
    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output>;
}
```

Stable algorithm IDs (`name()`):

| Type      | `name()`     |
| --------- | ------------ |
| `Wang`    | `"wang-v1"`  |
| `Panako`  | `"panako-v2"`|
| `Haitsma` | `"haitsma-v1"` |

Persist these alongside hashes if you ever plan to mix algorithm versions in one database.

### `StreamingFingerprinter` trait

Incremental, low-latency extraction.

```rust
pub trait StreamingFingerprinter {
    type Frame;

    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)>;
    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)>;
    fn latency_ms(&self) -> u32;
}
```

`push()` is non-blocking and returns any frames whose anchors are *fully observable* (their full lookahead has elapsed). `flush()` drains everything still pending — call it at end-of-stream. `latency_ms()` is a conservative upper bound from sample-in to hash-out.

> **Bit-exact guarantee.** Feeding the same audio in any chunking pattern (including 1-sample-per-push) produces the identical hash multiset as a single `Fingerprinter::extract` over the full buffer.
>
> **0.2.0 incremental streaming.** Wang/Panako keep a rolling
> `2·neighborhood_t + 1`-row spectrogram window and detect peaks
> frame-by-frame as each ripens; Haitsma keeps a single previous-frame
> band-energy vector. Per-push CPU is proportional to the new samples
> only — independent of total stream length.

### Shared value types

#### `SampleRate`

Newtype around `NonZeroU32`. Construct from one of the canonical constants or via `new`:

```rust
use audiofp::SampleRate;

let r = SampleRate::HZ_44100;        // 44_100
let r = SampleRate::new(32_000).unwrap();
assert!(SampleRate::new(0).is_none());
```

| Constant            | Hz     |
| ------------------- | ------ |
| `HZ_8000`           | 8 000  |
| `HZ_11025`          | 11 025 |
| `HZ_16000`          | 16 000 |
| `HZ_22050`          | 22 050 |
| `HZ_44100`          | 44 100 |
| `HZ_48000`          | 48 000 |

#### `AudioBuffer`

A borrowed mono PCM view:

```rust
pub struct AudioBuffer<'a> {
    pub samples: &'a [f32],
    pub rate: SampleRate,
}
```

#### `TimestampMs`

```rust
pub struct TimestampMs(pub u64);
```

Milliseconds since stream start. `u64` gives ≈ 584 million years of headroom.

---

## Classical Fingerprinters

### Wang (landmark pairs)

Avery Wang's "Shazam paper" algorithm: peaks in a log-mag spectrogram are paired into anchor-target landmarks; each pair packs into a 32-bit hash.

#### Hash layout

```text
[31..23]  f_a_q  9 bits, anchor frequency (quantised to 512 buckets)
[22..14]  f_b_q  9 bits, target frequency (same quantisation)
[13.. 0]  Δt    14 bits, frames between anchor and target (clamped 1..=16383)
```

#### `WangConfig`

```rust
pub struct WangConfig {
    pub fan_out: u16,            // default 10
    pub target_zone_t: u16,      // default 63 frames
    pub target_zone_f: u16,      // default 64 bins
    pub peaks_per_sec: u16,      // default 30
    pub min_anchor_mag_db: f32,  // default -50.0
}
```

| Field               | Default | Effect                                                     |
| ------------------- | ------- | ---------------------------------------------------------- |
| `fan_out`           | 10      | Targets paired with each anchor. Lower → smaller fingerprint, weaker recall |
| `target_zone_t`     | 63      | Maximum Δt (frames) for valid pairs                        |
| `target_zone_f`     | 64      | Maximum |Δf| (FFT bins) for valid pairs                    |
| `peaks_per_sec`     | 30      | Peaks the picker keeps per 1 s bucket                      |
| `min_anchor_mag_db` | -50.0   | Magnitude floor: peaks below this dB level are ignored     |

#### Output: `WangFingerprint`

```rust
pub struct WangFingerprint {
    pub hashes: Vec<WangHash>,    // sorted by (t_anchor, hash)
    pub frames_per_sec: f32,      // always 62.5 for wang-v1
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct WangHash {
    pub hash: u32,
    pub t_anchor: u32,
}
```

#### Example: custom config

```rust
use audiofp::classical::{Wang, WangConfig};
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

let cfg = WangConfig {
    fan_out: 5,             // smaller fingerprint
    peaks_per_sec: 20,      // fewer peaks → faster matching
    ..Default::default()
};
let mut wang = Wang::new(cfg);

let samples = vec![0.0_f32; 8_000 * 4];
let fp = wang.extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })?;
```

### Panako (triplet hashes)

Joren Six's Panako algorithm: anchors are paired with **two** targets each; the ratio of their offsets gives a tempo-invariant β value robust to ±5 % time stretch.

#### Hash layout

```text
[31..30]  sign       2 bits (sign of Δf_ab and Δf_bc)
[29..28]  mag_order  2 bits (which of {a, b, c} has largest magnitude)
[27..23]  β          5 bits, round((t_c − t_b) / (t_c − t_a) · 31)
[22..15]  Δf_ab      8 bits signed, clamped to ±127
[14.. 7]  Δf_bc      8 bits signed, clamped to ±127
[ 6.. 0]  reserved   7 bits, zero
```

#### `PanakoConfig`

```rust
pub struct PanakoConfig {
    pub fan_out: u16,            // default 5
    pub target_zone_t: u16,      // default 96
    pub target_zone_f: u16,      // default 96
    pub peaks_per_sec: u16,      // default 30
    pub min_anchor_mag_db: f32,  // default -50.0
}
```

#### Output: `PanakoFingerprint`

```rust
pub struct PanakoFingerprint {
    pub hashes: Vec<PanakoHash>,
    pub frames_per_sec: f32,    // 62.5
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable)]
pub struct PanakoHash {
    pub hash: u32,
    pub t_anchor: u32,
    pub t_b: u32,                // first target frame
    pub t_c: u32,                // second target frame
}
```

The extra `t_b`, `t_c` fields make tempo-aware time alignment possible during scoring.

### Haitsma–Kalker (band-power sign bits)

Philips robust hash: 33 logarithmically spaced bands from 300–2000 Hz, one bit per band per frame indicating whether the band-difference delta is positive between consecutive frames.

#### Hash layout

Per frame `n ≥ 1`:

```text
F[n][b] = ((E[n][b] − E[n][b+1]) − (E[n−1][b] − E[n−1][b+1])) > 0   for b ∈ {0..=31}
```

Packed `u32` with **band 0 in the most significant bit** (the "MSB-zero" convention) and band 31 in the LSB.

#### `HaitsmaConfig`

```rust
pub struct HaitsmaConfig {
    pub fmin: f32,    // default 300.0
    pub fmax: f32,    // default 2000.0
}
```

`Haitsma::new` panics if `fmin >= fmax` or `fmax >= sr / 2` (above Nyquist for the fixed 5 kHz operating rate).

#### Output: `HaitsmaFingerprint`

```rust
pub struct HaitsmaFingerprint {
    pub frames: Vec<u32>,         // one u32 per frame from n=1
    pub frames_per_sec: f32,      // 78.125
}
```

> Frame 0 has no hash (the algorithm needs frame n−1 for the delta). Frame indexing in `frames` is therefore offset by one relative to the spectrogram.

---

## Streaming Fingerprinters

Each classical fingerprinter has a streaming sibling:

| Streaming                | `Frame`        | `latency_ms()` |
| ------------------------ | -------------- | -------------- |
| `StreamingWang`          | `WangHash`     | 2 256          |
| `StreamingPanako`        | `PanakoHash`   | 2 784          |
| `StreamingHaitsma`       | `u32`          | 409            |

### Microphone-style usage

```rust
use audiofp::classical::StreamingWang;
use audiofp::StreamingFingerprinter;

let mut s = StreamingWang::default();
let mut all = Vec::new();

// Pretend incoming audio chunks (must be 8 kHz mono f32):
for chunk in audio_capture_iter() {
    for (t, hash) in s.push(&chunk) {
        all.push((t, hash));
    }
}

// Drain whatever's pending at end-of-stream.
all.extend(s.flush());

println!("{} hashes total, {:.1} ms latency", all.len(), s.latency_ms());
```

### Why the latency differs

- **Haitsma** depends only on the current and previous spectrogram frame → bounded by `n_fft / sr`.
- **Wang / Panako** must wait for the full target zone to elapse *and* one full second of peaks to settle the per-second adaptive thresholding. Without the +1 s, hashes near the buffer tail would briefly survive only to be culled by later peaks competing in the same bucket.

### Bit-exact equivalence

```rust
// Same audio fed offline vs streaming → same hash multiset.
let offline = Wang::default()
    .extract(AudioBuffer { samples: &whole_song, rate: SampleRate::HZ_8000 })?;

let mut streaming = StreamingWang::default();
let mut online = Vec::new();
for chunk in whole_song.chunks(1024) {
    online.extend(streaming.push(chunk).into_iter().map(|(_, h)| h));
}
online.extend(streaming.flush().into_iter().map(|(_, h)| h));

let mut a = offline.hashes;
let mut b = online;
a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
assert_eq!(a, b);   // ✅ guaranteed
```

---

## Audio File Decoding

Available with the default `std` feature, exposed as `audiofp::io`.

### `decode_to_mono`

```rust
pub fn decode_to_mono<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)>;
```

Returns `(samples, native_sample_rate_hz)`. Multi-channel files are downmixed to mono by averaging channels per frame.

```rust
use audiofp::io::decode_to_mono;

let (samples, sr) = decode_to_mono("song.flac")?;
println!("{} samples at {sr} Hz", samples.len());
```

### `decode_to_mono_at`

```rust
pub fn decode_to_mono_at<P: AsRef<Path>>(path: P, target_sr: u32) -> Result<Vec<f32>>;
```

Decode and resample to `target_sr` in one step. Internally uses `dsp::resample::SincResampler` at default quality (32-tap Kaiser, β=8.6). Pass-through when the file already matches `target_sr`.

```rust
// Get audio ready for Wang in one line:
let samples = decode_to_mono_at("song.mp3", 8_000)?;
```

### Supported formats

Whatever Symphonia provides with the features enabled in `Cargo.toml`:

| Format       | Extension(s)             |
| ------------ | ------------------------ |
| MP3          | `.mp3`                   |
| AAC          | `.aac`, `.m4a` (in MP4)  |
| FLAC         | `.flac`                  |
| OGG-Vorbis   | `.ogg`, `.oga`           |
| WAV / PCM    | `.wav`                   |

The decoder probes magic bytes too — extension-less files still work as long as they're a recognised format.

### Error handling

| Failure                                   | Error variant                    |
| ----------------------------------------- | -------------------------------- |
| File not found                            | `AfpError::Io`                   |
| Format unrecognised                       | `AfpError::Io`                   |
| Per-packet decode failure                 | (silently skipped — resilient)   |
| Stream-fatal decode failure               | `AfpError::Io`                   |

Recoverable per-packet failures are silently skipped to keep one corrupt block from killing a whole-file decode; only stream-fatal errors propagate.

---

## Watermark Detection

Available with the `watermark` feature. Wraps `tract-onnx` to run an AudioSeal-compatible model.

```toml
[dependencies]
audiofp = { version = "0.2", features = ["watermark"] }
```

### `WatermarkConfig`

```rust
pub struct WatermarkConfig {
    pub model_path: String,
    pub message_bits: u8,    // ≤ 32, default 16
    pub threshold: f32,      // [0, 1], default 0.5
    pub sample_rate: u32,    // default 16_000
}
```

Constructor with AudioSeal defaults:

```rust
use audiofp::watermark::WatermarkConfig;

let cfg = WatermarkConfig::new("audioseal_v0.2.onnx");
// message_bits: 16, threshold: 0.5, sample_rate: 16_000
```

### `WatermarkDetector`

```rust
let mut det = WatermarkDetector::new(cfg)?;

let buf = AudioBuffer { samples: &audio, rate: SampleRate::new(16_000).unwrap() };
let r = det.detect(buf)?;

println!("detected={} confidence={:.3} message={:#018b}",
         r.detected, r.confidence, r.message);
println!("localization length: {} samples", r.localization.len());
```

### `WatermarkResult`

| Field          | Type        | Meaning                                                                 |
| -------------- | ----------- | ----------------------------------------------------------------------- |
| `detected`     | `bool`      | `true` iff `confidence > threshold`                                     |
| `confidence`   | `f32`       | Mean of the per-output detection scores                                 |
| `message`      | `u32`       | Decoded message bits, LSB-first; bits at or above `message_bits` are 0  |
| `localization` | `Vec<f32>`  | Raw per-output detection scores (length depends on the model)           |

### Model contract

`audiofp::watermark` assumes the ONNX model has:

1. **One input** that accepts `[1, 1, T] f32` audio samples at `sample_rate`.
2. **At least two outputs**, in this order:
   - `[0]`: detection scores tensor (any shape; flattened for the localization vector and confidence mean).
   - `[1]`: message bit logits tensor (any shape; first `message_bits` values are read).

Bits are decoded as `logit ≥ 0`. If your AudioSeal export has a different layout, post-process accordingly before feeding it through this wrapper.

---

## DSP Primitives

For users wanting to build custom fingerprinters or analysis pipelines on top of `audiofp`'s building blocks. All available under `audiofp::dsp::*`.

### `dsp::stft`

```rust
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
use audiofp::dsp::windows::WindowKind;

let mut stft = ShortTimeFFT::new(StftConfig {
    n_fft: 2048,                  // power of two
    hop: 512,                     // 0 < hop ≤ n_fft
    window: WindowKind::Hann,
    center: true,                 // librosa-style reflect padding
});

let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin()).collect();
let spec = stft.magnitude(&samples);   // Vec<Vec<f32>>: (n_frames, n_bins)
println!("{} frames × {} bins", spec.len(), stft.n_bins());
```

Streaming `process_frame` lets you feed exactly `n_fft` samples and get one spectrum without allocating per call.

**0.2.0 fast-path methods:**

```rust
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};

let mut stft = ShortTimeFFT::new(StftConfig::new(2048));
let samples: Vec<f32> = vec![0.0; 16_000];

// Single contiguous Vec<f32> of shape (n_frames, n_bins).
let (mag, n_frames, n_bins) = stft.magnitude_flat(&samples);
assert_eq!(mag.len(), n_frames * n_bins);

// Power (|X|²) — skips the per-bin sqrt. Pair with 10·log10(p) instead
// of 20·log10(sqrt(p)) for an algebraically identical log result.
let (pow, _, _) = stft.power_flat(&samples);
assert_eq!(pow.len(), mag.len());

// Per-frame streaming variant of power_flat.
let frame = vec![0.0_f32; 2048];
let mut out = vec![0.0_f32; stft.n_bins()];
stft.process_frame_power(&frame, &mut out);
```

The classical fingerprinters all use `power_flat` / `process_frame_power`
internally — they avoid `O(N · M)` `sqrt` calls per spectrogram, a
notable win on the FFT-bound Haitsma path.

### `dsp::mel`

```rust
use audiofp::dsp::mel::{MelFilterBank, MelScale};

let fb = MelFilterBank::new(
    /* n_mels */ 128,
    /* n_fft  */ 2048,
    /* sr     */ 22_050,
    /* fmin   */ 0.0,
    /* fmax   */ 11_025.0,
    MelScale::Slaney,            // or MelScale::Htk
);

let mut log_mel = vec![0.0_f32; 128];
fb.log_mel(&magnitude_spectrum, &mut log_mel);
```

Slaney-normalised triangular filters; matches librosa's `feature.melspectrogram` defaults.

### `dsp::peaks`

```rust
use audiofp::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};

let picker = PeakPicker::new(PeakPickerConfig {
    neighborhood_t: 7,
    neighborhood_f: 7,
    min_magnitude: 1e-3,
    target_per_sec: 30,
});

let peaks: Vec<Peak> = picker.pick(&magnitude_spec, n_frames, n_bins, frames_per_sec);
```

2-D rolling max via Lemire's monotonic deque, amortised O(N · M) regardless of neighbourhood size.

> **0.2.0 breaking change.** `PeakPicker::pick` now takes `&mut self` so
> it can re-use its rolling-max scratch across calls. If you previously
> held a `PeakPicker` behind `&self`, store it as `Mutex<PeakPicker>` or
> use one picker per producing thread.

### `dsp::resample`

```rust
use audiofp::dsp::resample::{linear, SincQuality, SincResampler};

// Cheap and aliased on downsamples — only use for non-critical paths.
let y = linear(&x, 44_100, 8_000);

// Default quality (32-tap, β=8.6).
let r = SincResampler::new(44_100, 8_000);
let y = r.process(&x);

// Higher quality.
let r = SincResampler::with_quality(
    44_100,
    8_000,
    SincQuality { half_taps: 64, kaiser_beta: 12.0 },
);
let y = r.process(&x);
```

Cutoff is automatically `min(from, to) / 2` to suppress aliasing on downsamples.

### `dsp::windows`

```rust
use audiofp::dsp::windows::{make_window, WindowKind};

let w = make_window(WindowKind::Hann, 1024);
```

Periodic windows (period N, not N-1) — matches librosa / `scipy.signal.get_window(..., fftbins=True)`.

---

## Error Handling

All fallible APIs return `Result<T, AfpError>`:

```rust
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AfpError {
    #[error("audio too short: needed at least {needed} samples, got {got}")]
    AudioTooShort { needed: usize, got: usize },

    #[error("unsupported sample rate: {0} Hz (supported: 8000, 11025, 16000, 22050, 44100, 48000)")]
    UnsupportedSampleRate(u32),

    #[error("unsupported channel count: {0}")]
    UnsupportedChannels(u16),

    #[error("model not found at {0}")]
    ModelNotFound(String),

    #[error("model load failed: {0}")]
    ModelLoad(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("buffer overrun: dropped {dropped} samples")]
    BufferOverrun { dropped: usize },

    #[error("invalid configuration: {0}")]
    Config(String),

    #[error("io: {0}")]
    Io(String),
}
```

`#[non_exhaustive]` — match exhaustively only inside the crate. Add a `_` arm to keep your match safe across SDK upgrades.

### Typical error paths

```rust
use audiofp::{AfpError, AudioBuffer, Fingerprinter, SampleRate, classical::Wang};

let mut wang = Wang::default();
let buf = AudioBuffer { samples: &short_audio, rate: SampleRate::HZ_44100 };

match wang.extract(buf) {
    Ok(fp) => println!("{} hashes", fp.hashes.len()),

    Err(AfpError::UnsupportedSampleRate(hz)) => {
        eprintln!("Wang needs 8 kHz, got {hz}. Resample first.");
    }
    Err(AfpError::AudioTooShort { needed, got }) => {
        eprintln!("Need {needed} samples ({:.1} s), got {got}.", needed as f32 / 8_000.0);
    }

    Err(e) => eprintln!("Unexpected error: {e}"),
}
```

---

## Performance Tips

### 1. Reuse the `Fingerprinter` across calls

`Wang::new` allocates an FFT plan, window table, and scratch buffers. Don't recreate one per file:

```rust
// Slow: allocates per file
for path in paths {
    let mut wang = Wang::default();
    let samples = decode_to_mono_at(path, 8_000)?;
    let fp = wang.extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })?;
}

// Fast: one Wang, many extractions
let mut wang = Wang::default();
for path in paths {
    let samples = decode_to_mono_at(path, 8_000)?;
    let fp = wang.extract(AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 })?;
}
```

Same applies to `Panako`, `Haitsma`, and `WatermarkDetector`.

### 2. Pick the right algorithm for the workload

| Goal                                             | Algorithm   |
| ------------------------------------------------ | ----------- |
| Music identification (Shazam-style)              | Wang        |
| Music identification with tempo robustness       | Panako      |
| Frame-aligned dense IDs / streaming with low lag | Haitsma     |
| Smallest fingerprints                            | Haitsma     |

### 3. Tune `fan_out` and `peaks_per_sec` to match your index

A larger `fan_out` (more hashes per anchor) increases recall but balloons storage. For Wang, 5–10 is the typical range; 3 is acceptable for tight constraints, ≥ 15 wastes index space.

### 4. Avoid the `linear` resampler for production

It's there as a baseline. Use `SincResampler` for anything user-facing — the aliasing in `linear` will degrade fingerprint quality on rate conversions like 44.1k → 8k.

### 5. Opt in to `mimalloc` if your downstream binary doesn't pick an allocator

```toml
[dependencies]
audiofp = { version = "0.2", features = ["mimalloc"] }
```

This installs `mimalloc::MiMalloc` as the process-wide `#[global_allocator]`. Off by default because libraries shouldn't pick the allocator on behalf of their consumers — flip it on in your binary or in `default = ["std", "mimalloc"]` if you're vendoring `audiofp`.

### 6. Streaming hot path is allocation-free and truly incremental (0.2.0+)

After the first push warms up internal scratch buffers,
`StreamingFingerprinter::push` does no allocations on the hot path.
**The streaming impls are now genuinely incremental** — Wang and Panako
maintain a rolling spectrogram window of `2·neighborhood_t + 1` rows
and detect peaks frame-by-frame as each becomes ripe; Haitsma keeps
just one previous-frame band-energy array. Per-push CPU is proportional
to the number of new samples, **not** to total stream length. Safe to
call from realtime audio threads.

---

## Feature Flags

| Feature      | Default | Brings in                                                                       |
| ------------ | :-----: | ------------------------------------------------------------------------------- |
| `std`        |   ✅    | Symphonia file decoding helpers (`audiofp::io`)                                     |
| `watermark`  |         | `tract-onnx` + `ndarray`; enables `audiofp::watermark`                              |
| `neural`     |         | (Reserved for the upcoming Phase 5 ONNX neural fingerprinter)                   |
| `mimalloc`   |         | Installs `mimalloc` as the process-wide `#[global_allocator]`                   |

### Minimal build (no_std + alloc)

```toml
[dependencies]
audiofp = { version = "0.2", default-features = false }
```

This drops `symphonia` (so no `audiofp::io`), `tract-onnx` (so no `audiofp::watermark`), and `mimalloc`. The DSP primitives and classical fingerprinters all remain available.

### Watermark detection only

```toml
[dependencies]
audiofp = { version = "0.2", default-features = false, features = ["watermark"] }
```

`watermark` implies `std`; you get `audiofp::watermark` plus the rest of the SDK, without Symphonia.

---

## no_std / Embedded

The DSP primitives and classical fingerprinters compile under `no_std + alloc`:

```toml
[dependencies]
audiofp = { version = "0.2", default-features = false }
```

In your crate root:

```rust
#![no_std]
extern crate alloc;

use audiofp::{AudioBuffer, Fingerprinter, SampleRate, classical::Wang};
// ... use audiofp APIs as usual.
```

> ⚠️ **Bare-metal note.** `rustfft` (used by the STFT primitive) transitively pulls `num-traits` with the `std` feature, so the no_std build currently only runs on hosted targets where `std` is reachable for *dependencies* (even if your own crate is `no_std`). True Cortex-M support will require a `microfft`-backed swap — on the roadmap.

What works without `std` today:

| Module                | Status                                                  |
| --------------------- | ------------------------------------------------------- |
| `audiofp::dsp::*`         | ✅ host-only no_std (rustfft transitive issue)          |
| `audiofp::classical::*`   | ✅ same                                                 |
| `audiofp::io`             | ❌ requires `std`                                        |
| `audiofp::watermark`      | ❌ requires `std` + `watermark`                          |

---

## Determinism guarantees

- **Identical inputs → identical outputs.** Same audio, same fingerprinter, same config → bit-for-bit identical hashes on every call and on every supported target.
- **Stable algorithm IDs.** `Fingerprinter::name()` returns a versioned string (e.g. `"wang-v1"`); a future major bump that changes hash bytes will change the version suffix.
- **Stable hash layouts.** Bit positions in `WangHash::hash`, `PanakoHash::hash`, and Haitsma frames are stable across patch and minor versions inside `0.x`.

---

## License

MIT. See [LICENSE](LICENSE).

## Links

- [Crates.io](https://crates.io/crates/audiofp)
- [Documentation](https://docs.rs/audiofp)
- [Repository](https://github.com/themankindproject/audiofp)
- [Changelog](CHANGELOG.md)
