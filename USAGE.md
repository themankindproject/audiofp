# audiofp Usage Guide

> Complete reference for `audiofp` 0.4.0 — the pure-Rust audio fingerprinting SDK.
>
> This guide documents every public API and every algorithm at full
> implement-it-by-hand depth: exact constants, bit layouts, formulas, ordering
> rules, and failure modes. Every runnable snippet in this guide is compiled as
> part of the crate's CI (they live as a compile-checked fixture), so nothing
> here rots silently.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Core API](#core-api)
  - [Fingerprinter trait](#fingerprinter-trait)
  - [StreamingFingerprinter trait](#streamingfingerprinter-trait)
  - [Shared value types](#shared-value-types)
- [Algorithm Reference](#algorithm-reference)
  - [Shared front-end](#shared-front-end)
  - [Wang (landmark pairs)](#wang-landmark-pairs)
  - [Panako (triplet hashes)](#panako-triplet-hashes)
  - [Haitsma–Kalker (band-power sign bits)](#haitsmakalker-band-power-sign-bits)
- [Matching / Identification](#matching--identification)
  - [MatchResult semantics](#matchresult-semantics)
  - [WangMatcher](#wangmatcher)
  - [Prebuilt index for repeated 1:1 matches](#prebuilt-index-for-repeated-11-matches)
- [HaitsmaMatcher](#haitsmamatcher)
- [PanakoMatcher](#panakomatcher)
  - [Prebuilt index for repeated 1:1 Panako matches](#prebuilt-index-for-repeated-11-panako-matches)
  - [Tuning match thresholds](#tuning-match-thresholds)
- [NeuralMatcher](#neuralmatcher)
- [1:N helpers and in-memory indexes](#1n-helpers-and-in-memory-indexes)
- [Parallel 1:N matching (rayon)](#parallel-1n-matching-rayon)
- [Streaming Fingerprinters](#streaming-fingerprinters)
- [Fingerprint Serialization](#fingerprint-serialization)
  - [Cache files (.afp)](#cache-files-afp)
- [Audio File Decoding](#audio-file-decoding)
- [Watermark Detection](#watermark-detection)
- [Neural Embedder](#neural-embedder)
- [DSP Primitives](#dsp-primitives)
- [Async, batching, and models](#async-batching-and-models)
- [Error Handling](#error-handling)
- [Performance Tips](#performance-tips)
- [Feature Flags](#feature-flags)
- [no_std / Embedded](#no_std--embedded)
- [Determinism Guarantees](#determinism-guarantees)
- [Examples](#examples)

---

## Quick Start

Add the dependency. The default build is `no_std + alloc` with **no codecs**;
for file decoding (`audiofp::io`) enable the codecs you need — see
[Feature Flags](#feature-flags):

```toml
[dependencies]
audiofp = { version = "0.4", features = ["std-mp3", "std-wav"] }
```

### Example 1 — fingerprint raw PCM (no codec features needed)

```rust
use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

fn main() {
    let samples = vec![0.0_f32; 8_000 * 3]; // 3 s of silence @ 8 kHz
    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000).unwrap();
    println!("{} hashes", fp.hashes.len()); // silence → 0
    assert_eq!(fp.frames_per_sec, 62.5);
}
```

### Example 2 — fingerprint an MP3 and print the first hashes

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode any enabled format to mono f32 at Wang's required 8 kHz.
    let samples = decode_to_mono_at("song.mp3", 8_000)?;

    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000)?;

    println!("{} hashes at {:.1} frames/s", fp.hashes.len(), fp.frames_per_sec);
    for h in fp.hashes.iter().take(5) {
        // t_anchor is the anchor's STFT frame index (a plain u32).
        println!("  t_anchor={} hash={:08x}", h.t_anchor, h.hash);
    }
    Ok(())
}
```

### Example 3 — identify a clip against a catalog

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn fingerprint(
    path: &str,
    wang: &mut Wang,
) -> Result<audiofp::classical::WangFingerprint, Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at(path, 8_000)?;
    Ok(wang.extract(&samples, SampleRate::HZ_8000)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut wang = Wang::default();
    let query = fingerprint("clip.wav", &mut wang)?;
    let refs = vec![
        fingerprint("catalog_a.flac", &mut wang)?,
        fingerprint("catalog_b.flac", &mut wang)?,
    ];

    let matcher = WangMatcher::new(WangMatchConfig::default());
    let m = matcher.match_one(&query, &refs[0]);
    if m.is_match {
        // Positive offset ⇒ the query starts later in the reference.
        println!("match: score={:.3} votes={} offset={} ms",
                 m.score, m.votes, m.offset.ms);
    }
    Ok(())
}
```

### Example 4 — the prelude (fewer imports)

```rust
use audiofp::prelude::*; // Wang, configs, hash types, both traits, error types

fn main() {
    let samples = vec![0.0_f32; 16_000];
    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000).unwrap();
    let _ = fp.hashes.len();
}
```

The prelude carries the classical fingerprinters (offline + streaming), their
config/hash types, `AfpError`/`Result`, `SampleRate`/`TimestampMs`,
`FingerprintEnvelope`, and both traits. Feature-gated modules (`io`, `neural`,
`watermark`) are **not** in the prelude — import those from their own modules.

---

## Core Concepts

### What is an audio fingerprint?

A **perceptual hash** of an audio recording — small enough to store and search
at scale, yet stable across re-encoding, modest noise, and (for Panako) small
tempo changes. Two recordings of the same song share many hashes; two
unrelated recordings do not.

`audiofp` ships three classical fingerprinters, one neural embedder, and one
watermark detector:

| Algorithm  | Output          | Rate  | Frame rate | Storage / s        | When to use                            |
| ---------- | --------------- | ----- | ---------- | ------------------ | -------------------------------------- |
| `Wang`     | Landmark pairs  | 8 kHz | 62.5 fps   | ~2.4 KB (fan-out 10) | Music ID; "Shazam-style" matching    |
| `Panako`   | Triplet hashes  | 8 kHz | 62.5 fps   | ~2.0 KB (fan-out 5)  | Tempo-robust music ID (±5 % stretch) |
| `Haitsma`  | 32 bits / frame | 5 kHz | 78.125 fps | 312 B               | Compact dense IDs; lowest latency    |
| `NeuralEmbedder` | f32 vector / window | model-defined | 1/`hop_secs` | `4·dim` B / window | Semantic / cover detection (BYO ONNX model) |
| `WatermarkDetector` | message + confidence | 16 kHz | — | — | AudioSeal-style watermark detection (BYO model) |

All three classical fingerprinters:

- accept mono `f32` PCM, nominally in `[-1.0, 1.0]` (values outside are
  fingerprinted as-is; only NaN/Inf are rejected — see
  [Error Handling](#error-handling))
- **require** their native sample rate (wrong rate → `UnsupportedSampleRate`
  from `extract`; resample first — see
  [Audio File Decoding](#audio-file-decoding))
- need at least **2 seconds** of audio (`min_samples()`)
- produce hash structs that are `#[repr(C)]` + `bytemuck::Pod` — castable
  directly to/from bytes for storage, `mmap`, or IPC

### Matching vs persistence

**In-memory matching is in scope** — see
[Matching / Identification](#matching--identification). Use the matchers or
the 1:N indexes instead of naive hash-set overlap.

**Persistence is out of scope** beyond a simple self-describing binary blob
([Fingerprint Serialization](#fingerprint-serialization)) and its `.afp`
file-cache helper ([Cache files](#cache-files-afp)). There is no
on-disk index, RPC wire format, or database adapter. A typical production
pipeline is:

1. `audiofp` → fingerprints per track (enrolment)
2. Your store (RocksDB, SQLite, object store, …) → durable catalog
3. Load references into memory → `WangIndex` / `match_ranked` / your own
   inverted index over the raw Pod hashes

### The one-minute mental model

For Wang/Panako/Haitsma alike, extraction is a fixed pipeline:

```text
PCM f32 @ native rate
   │  Hann-windowed STFT (real FFT, pre-planned, non-centred framing)
   ▼
power spectrogram  (|X|² per bin — the sqrt is skipped algebraically)
   │  10·log10(max(p, floor)) — dB with a floor
   ▼
[ Wang / Panako ]  2-D peak picking → per-second top-K → pairing/triplets → hashes
[ Haitsma ]        33 log-spaced band energies → sign of band-delta change → 32 bits/frame
```

Matching then looks for **agreement**: Wang votes on a constant time offset,
Panako votes on a (time-scale, offset) line, Haitsma minimises a bit error
rate over sliding alignments. Every matcher outputs the same
[`MatchResult`](#matchresult-semantics) shape.

---

## Core API

### `Fingerprinter` trait

Offline (whole-buffer) extraction. Implementors are stateful only insofar as
they reuse scratch buffers — `extract(a)` never depends on a previous call.

```rust,ignore
// Reference definition (see src/fp.rs). Shown for orientation; use the
// real trait via `use audiofp::Fingerprinter;`.
pub trait Fingerprinter {
    type Output;
    type Config: Clone + Send + Sync;

    fn name(&self) -> &'static str;
    fn config(&self) -> &Self::Config;
    fn required_sample_rate(&self) -> SampleRate;
    fn min_samples(&self) -> usize;
    fn extract(&mut self, samples: &[f32], rate: SampleRate) -> Result<Self::Output>;
}
```

`required_sample_rate()` returns a `SampleRate` — compare it directly against
your audio's rate:

```rust
use audiofp::classical::{Haitsma, Panako, Wang};
use audiofp::Fingerprinter;

fn main() {
    assert_eq!(Wang::default().required_sample_rate().hz(), 8_000);
    assert_eq!(Panako::default().required_sample_rate().hz(), 8_000);
    assert_eq!(Haitsma::default().required_sample_rate().hz(), 5_000);

    // Minimum audio length (2 s at each algorithm's native rate):
    assert_eq!(Wang::default().min_samples(), 16_000);
    assert_eq!(Haitsma::default().min_samples(), 10_000);
}
```

Stable algorithm IDs (`name()`) — persist these alongside hashes if you ever
mix algorithm versions in one catalog:

| Type      | `name()`       |
| --------- | -------------- |
| `Wang`    | `"wang-v1"`    |
| `Panako`  | `"panako-v2"`  |
| `Haitsma` | `"haitsma-v1"` |
| `NeuralEmbedder` | `"neural-onnx-v0"` |

`extract` runs, in order: NaN/Inf rejection (`NonFiniteSample`), the
`max_input_samples` check (`InputTooLarge`), the sample-rate check
(`UnsupportedSampleRate`), the `min_samples` check (`AudioTooShort`), then the
algorithm itself. Cheap checks run before the O(n) finiteness scan.

### `StreamingFingerprinter` trait

Incremental, low-latency extraction:

```rust,ignore
// Reference definition (see src/fp.rs).
pub trait StreamingFingerprinter {
    type Frame;

    fn required_sample_rate(&self) -> u32;
    fn push(&mut self, samples: &[f32]) -> Result<Vec<(TimestampMs, Self::Frame)>>;
    fn flush(&mut self) -> Result<Vec<(TimestampMs, Self::Frame)>>;
    fn latency_ms(&self) -> u32;

    // Provided callback variants (see warning below):
    fn push_with<F>(&mut self, samples: &[f32], callback: F) -> Result<usize>
    where F: FnMut(TimestampMs, &Self::Frame);
    fn flush_with<F>(&mut self, callback: F) -> Result<usize>
    where F: FnMut(TimestampMs, &Self::Frame);
}
```

Notes that matter in practice:

- **All `push`/`flush` implementations are fallible** and return `Result`
  (since 0.4.0). The classical streams (Wang/Panako/Haitsma) never error on
  valid input; `StreamingNeuralEmbedder` propagates ONNX inference errors.
- `push()` is non-blocking and returns only frames whose full lookahead has
  elapsed. `flush()` drains everything still pending — call it at
  end-of-stream.
- **`flush` lifecycle contract** (all implementations):
  - *Idempotent* — a second `flush` after the stream is drained returns an
    empty `Vec`.
  - *Push after flush is valid* — the stream does not enter a "finished"
    state; continuation audio appends. Call `reset()` on the concrete type to
    start a fresh stream.
- There is **no runtime rate check on `push`** — feeding wrong-rate samples
  produces garbage hashes silently. Assert or resample up front.
- `latency_ms()` is a conservative upper bound from sample-in to hash-out.
- `push_with` / `flush_with` invoke `callback(timestamp, &frame)` per emitted
  frame and return the count — no intermediate `Vec` allocation.
  > **Warning — the defaults allocate.** The provided `push_with` /
  > `flush_with` bodies just call `push` / `flush` and iterate, so they
  > allocate exactly like the `Vec`-returning methods. Only impls marked
  > [`ZeroAllocStreaming`](#zeroallocstreaming-guarantee) override both with
  > true allocation-free paths. Generic code that must not allocate (audio
  > callbacks) should bound on that trait, not this one.

> **Bit-exact guarantee.** Feeding the same audio in any chunking pattern
> (including 1-sample-per-push) produces the identical hash *multiset* as a
> single `Fingerprinter::extract` over the full buffer. This is pinned by
> in-tree tests (`streaming_offline_equivalence`,
> `streaming_chunk_size_invariant`, `streaming_with_one_sample_chunks_…`).

### Shared value types

#### `SampleRate`

Newtype around `NonZeroU32`:

```rust
use audiofp::SampleRate;

fn main() {
    let r = SampleRate::HZ_44100;              // 44_100
    let r = SampleRate::new(32_000).unwrap();  // any non-zero rate
    assert!(SampleRate::new(0).is_none());
    println!("{r:?} runs at {} Hz", r.hz());
}
```

| Constant   | Hz      | Used by                        |
| ---------- | ------- | ------------------------------ |
| `HZ_5000`  | 5 000   | Haitsma                        |
| `HZ_8000`  | 8 000   | Wang, Panako                   |
| `HZ_11025` | 11 025  | quarter-rate audio             |
| `HZ_16000` | 16 000  | watermark + neural default     |
| `HZ_22050` | 22 050  | half-rate audio                |
| `HZ_44100` | 44 100  | CD audio                       |
| `HZ_48000` | 48 000  | pro/video audio                |

#### `TimestampMs`

```rust
pub struct TimestampMs(pub u64);
```

Milliseconds since stream start; `u64` gives ≈ 584 million years of headroom.

#### `VERSION`

`audiofp::VERSION` is the compile-time crate version string (e.g.
`"0.4.0"`). Useful for runtime sanity checks when the SDK is vendored.

---

## Algorithm Reference

This section documents each fingerprinter at implement-it-by-hand depth. The
constants below are the exact values used by the in-tree extractors; a
faithful reimplementation following these steps produces byte-identical
hashes.

### Shared front-end

Wang and Panako share an identical front-end up to (and including) peak
picking:

| Parameter        | Wang          | Panako        | Haitsma       |
| ---------------- | ------------- | ------------- | ------------- |
| Sample rate      | 8 000 Hz      | 8 000 Hz      | 5 000 Hz      |
| `n_fft`          | 1 024         | 1 024         | 2 048         |
| Hop              | 128 (16 ms)   | 128 (16 ms)   | 64 (12.8 ms)  |
| Frame rate       | 62.5 fps      | 62.5 fps      | 78.125 fps    |
| Window           | Hann, periodic (period = `n_fft`) | same | same |
| Framaming        | non-centred: frame `i` starts at sample `i·hop`; tail samples short of a full frame are dropped (streaming buffers them) | same | same |
| Bins             | 513 (`n_fft/2 + 1`) | 513     | 1 025         |

Per frame the STFT computes the **power** spectrum `p[k] = Re² + Im²` (the
per-bin `sqrt` is deliberately skipped). dB conversion is

```text
db[k] = 10 · log10( max(p[k], 1e-12) )      (Wang, Panako)
```

i.e. a magnitude floor of `1e-6` squared. (`10·log10(p) ≡ 20·log10(|X|)` —
identical to computing dB from magnitudes, one `sqrt` cheaper.)

Haitsma consumes band energies straight from the power spectrum (no dB step).

#### Peak picking (Wang & Panako)

Peaks are local maxima of the dB spectrogram under a
`(2·15+1) × (2·15+1)` = **31×31** box (15 frames × 15 bins half-width),
computed exactly as a 2-D rolling max (Lemire monotonic deque along each
axis). A cell `(t, f)` survives iff:

```text
db[t][f] >  -50.0                    (min_anchor_mag_db)
db[t][f] >= rolling_max_31x31(t, f)  (>= so flat plateaus emit every cell)
```

Then a **per-second adaptive cap** keeps only the top `30` peaks
(`peaks_per_sec`) in each 1-second bucket. Bucket index =
`floor(t_frame / 62.5)`. Ranking within a bucket is by magnitude descending,
ties by `(t_frame, f_bin)` ascending — a total order, so the kept set is
deterministic and identical offline vs streaming.

### Wang (landmark pairs)

Avery Wang's ISMIR-2003 "industrial-strength" algorithm (the Shazam paper;
the underlying patent family expired 2024-01-07). Anchor peaks are paired
with later target peaks; each pair packs into a 32-bit hash.

**Step-by-step:**

1. Resample to 8 kHz mono (caller's responsibility).
2. STFT per the [shared front-end](#shared-front-end); convert to dB.
3. Pick peaks as above; sort survivors by `(t_frame, f_bin)`.
4. For each anchor `a` (in time order), consider every later peak `b` with
   `1 ≤ Δt ≤ 63` (`target_zone_t`) and `|Δf| ≤ 64` bins (`target_zone_f`).
   Keep the top **10** (`fan_out`) targets ranked by magnitude descending,
   ties by `(t_frame, f_bin)` ascending.
5. Pack each surviving `(a, b)` pair:

```text
f_q(x)  = floor( f_bin · 512 / 513 )        — 513 bins → 9-bit bucket
hash    = (f_q(a) << 23) | (f_q(b) << 14) | clamp(Δt, 1, 16383)

[31..23]  f_q(a)   9 bits   anchor frequency bucket  (0..=511)
[22..14]  f_q(b)   9 bits   target frequency bucket  (0..=511)
[13.. 0]  Δt       14 bits  frames anchor → target   (1..=16383, clamped)
```

6. Output is sorted by `(t_anchor, hash)` — consumers may binary-search or
   merge-join on it.

**Decode** a hash with pure arithmetic:

```rust
fn main() {
    let hash: u32 = 0x1234_5678;
    let f_a = hash >> 23;              // 9 bits
    let f_b = (hash >> 14) & 0x1FF;    // 9 bits
    let dt = hash & 0x3FFF;            // 14 bits
    println!("anchor bucket {f_a}, target bucket {f_b}, Δt {dt} frames");
}
```

**Output sizes:** at the default config expect ~300 hashes/second of rich
music (~2.4 KB/s at 8 bytes per `WangHash`). Silence produces zero hashes.

### Panako (triplet hashes)

Joren Six's Panako algorithm (start-end fingerprinting). Each anchor is
paired with **two** targets; the geometry of the triplet survives ±5 %
time-stretch because the hash stores *ratios*, not absolute offsets.

**Step-by-step:**

1. Resample to 8 kHz mono; STFT + dB + peak picking exactly as Wang
   (same constants, same 31×31 neighbourhood, same 30/s cap).
2. For each anchor `a`, collect later peaks as candidate targets with the
   **strict** zone `1 ≤ Δt < 96` (`target_zone_t`) and `|Δf| < 96`
   (`target_zone_f`). Candidates beyond `2·fan_out = 10` evict the
   weakest-magnitude member (soft cap). This is provably lossless for the
   top-K pair selection that follows: the best `fan_out` pairs by
   `mag(b) + mag(c)` only ever use the `fan_out + 1` strongest candidates.
3. Form candidate triplets `(a, b, c)` with `t_a < t_b < t_c` from the
   surviving targets and keep the top `fan_out = 5` by
   `mag(b) + mag(c)` (ties by position ascending).
4. Pack:

```text
Δf_ab = clamp(f_b − f_a, −127, 127)      Δf_bc = clamp(f_c − f_b, −127, 127)
sign    = (f_b ≥ f_a) | ((f_c ≥ f_b) << 1)          — 2 bits
mag_ord = 0 if a largest | 1 if b largest | 2 if c largest   — 2 bits
β       = clamp( round( (t_c − t_b) / (t_c − t_a) · 31 ), 0, 31 )  — 5 bits

hash = (sign    << 30)      — bits 31..30
     | (mag_ord << 28)      — bits 29..28
     | (β       << 23)      — bits 27..23
     | ((Δf_ab as i8 as u8) << 15)   — bits 22..15 (two's complement byte)
     | ((Δf_bc as i8 as u8) <<  7)   — bits 14.. 7
                                    — bits  6.. 0 are zero (reserved)
```

5. The emitted struct keeps the raw frame indices `t_anchor`, `t_b`, `t_c`
   alongside the packed hash — matching needs them to re-derive the local
   time scale.

Why the divergences from Six's original: the original packs unsigned 6-bit
`|Δf|` quantisations and a 4-bit sign/mag slot; this crate uses signed 8-bit
`Δf` (clamped ±127) with the sign bits at the top. `target_zone_f` above 127
collides distinct triplets onto the same clamped code — keep
`target_zone_f ≤ 127` if you care (the default 96 is safe).

### Haitsma–Kalker (band-power sign bits)

The Philips robust hashing scheme (Haitsma & Kalker, 2002): extremely
compact (32 bits per frame) and extremely cheap to match (popcount).

**Step-by-step:**

1. Resample to **5 kHz** mono. STFT with `n_fft = 2048`, `hop = 64` →
   78.125 fps, 1 025 bins.
2. Build **33 band edges** logarithmically spaced between `fmin = 300 Hz`
   and `fmax = 2000 Hz`:

```text
edge(k) = 300 · (2000 / 300)^(k / 32),      k = 0..=32
```

   Each FFT bin is assigned to the band whose edge-interval contains its
   centre frequency; band `b`'s energy `E[n][b]` is the sum of the power
   bins assigned to it (empty bands degrade to energy 0 — no error).
3. For each frame `n ≥ 1` emit 32 sign bits — band 0 in the **most
   significant bit**:

```text
bit[b] = ( (E[n][b] − E[n][b+1]) − (E[n−1][b] − E[n−1][b+1]) ) > 0,   b = 0..=31
frame  = Σ bit[b] << (31 − b)        // MSB = band 0, LSB = band 31
```

4. Frame 0 has no hash (it needs `n−1`). `frames[i]` therefore corresponds
   to spectrogram frame `i + 1`.

At 78.125 fps and 4 bytes/frame this is 312 B/s — roughly 8× smaller than
Wang. The cost is a weaker notion of "hash identity": matching is per-frame
bit-error rate, so it wants the whole frame sequence, not a hash set.

---

## Matching / Identification

`audiofp::matching` scores a **query** fingerprint against one or more
**references** entirely in memory.

| Matcher          | Fingerprint          | Strategy                                       | Feature |
| ---------------- | -------------------- | ---------------------------------------------- | ------- |
| `WangMatcher`    | `WangFingerprint`    | Offset-histogram voter (Shazam-style)          | —       |
| `HaitsmaMatcher` | `HaitsmaFingerprint` | Sliding BER (+ optional sub-fingerprint LUT)   | —       |
| `PanakoMatcher`  | `PanakoFingerprint`  | 2-D Hough (scale × offset) + optional RANSAC   | —       |
| `NeuralMatcher`  | `NeuralFingerprint`  | Cosine similarity (Global / SlidingMax / DTW)  | `neural`|

All four implement the `Matcher` trait:

```rust,ignore
// Reference definition (src/matching/mod.rs).
pub trait Matcher {
    type Fingerprint;
    type Config;

    fn new(cfg: Self::Config) -> Self;
    fn config(&self) -> &Self::Config;
    fn match_one(&self, query: &Self::Fingerprint, reference: &Self::Fingerprint)
        -> MatchResult;
}
```

`match_one` never panics on any input (including empty fingerprints, NaN
frame rates, or length-mismatched neural sequences — those soft-fail to
`MatchResult::NONE`).

### MatchResult semantics

```rust,ignore
// Reference definition (src/matching/mod.rs).
pub struct MatchResult {
    pub is_match: bool,        // cleared every configured threshold?
    pub score: f32,            // normalised confidence in [0, 1]
    pub votes: u32,            // raw aligned-evidence count
    pub prominence: f32,       // peak ÷ background — the false-positive guard
    pub offset: TimeOffset,    // query position within the reference
    pub time_scale: f32,       // query_duration / reference_duration
}

pub struct TimeOffset {
    pub frames: i64,           // offset in reference STFT frames (exact)
    pub ms: i64,               // frames × 1000 / frames_per_sec (rounded)
}
```

Field semantics per matcher — **prominence is not comparable across matcher
types**, do not threshold it with one global constant:

| Matcher | `score` | `votes` | `prominence` | `time_scale` |
| --- | --- | --- | --- | --- |
| Wang | distinct contributing query hashes ÷ query hash count | consolidated peak (±tol window) | `peak / (mean_rest + 1)` on the consolidated offset histogram | `1.0` |
| Haitsma | `1 − BER` at the best alignment | aligned frame count (overlap) | `median_BER / (BER + ε)` (matcher) — index path uses `0.5 / BER` | `1.0` |
| Panako | inliers ÷ query hash count | RANSAC inliers (or Hough peak) | `peak / (mean_rest + 1)` on the consolidated 2-D Hough grid | `1 / fitted_scale`, clamped `[0.5, 2.0]` |
| Neural | cosine similarity (0..1) | depends on aggregation | relative cosine excess (SlidingMax) or `1.0` | `1.0` |

`offset.frames` is positive when the query starts **later** in the reference
(reference anchor at a larger frame index); `offset.ms` uses the *reference's*
frame rate.

### WangMatcher

The canonical Shazam alignment: matching landmark hashes must agree on one
**constant time offset**; random collisions scatter across offsets while a
true match spikes at one.

**Algorithm, step by step (defaults in parentheses):**

1. **Index the reference** — `hash → [t_anchor]`, dropping hashes whose
   posting list exceeds `max_postings_per_hash` (100). Internally a sorted
   flat array with binary search (`SortedPostings`) — one allocation, no
   per-hash `Vec`s.
2. **Vote** — for each query hash that hits the index, compute
   `δ = t_ref − t_query` and bump a dense histogram bin. The histogram spans
   `[−q_max, r_max]`, capped at 10 000 000 bins (≈4 min × 62.5 fps is ~18 K
   bins; the cap only engages on adversarial fingerprints).
3. **Consolidate** — box-sum each bin over `±offset_tolerance_frames` (1)
   so framing jitter coalesces into one peak. An O(1) sliding window — no
   transient prefix array.
4. **Peak** — take the max consolidated value; on a plateau of equal maxima,
   pick the **middle** bin (deterministic, jitter-unbiased). Require
   `≥ min_votes` (5).
5. **Prominence** — `peak / (mean of other consolidated bins + 1)`; require
   `≥ min_prominence` (5.0). This is the primary false-positive guard: a
   true match is a sharp spike (≫ 10), random collisions are flat (~1).
6. **Score** — count *distinct query hashes* with at least one vote within
   ±tol of the winning offset, divide by the query hash count; require
   `≥ min_score` (0.15).

```rust
use audiofp::classical::{Wang, WangFingerprint};
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn main() {
    // Build a query that is the reference shifted by 100 frames.
    let mut ref_hashes = Vec::new();
    let mut q_hashes = Vec::new();
    for i in 0..40_u32 {
        let t = 1000 + i * 10;
        ref_hashes.push(audiofp::classical::WangHash { hash: 500 + i, t_anchor: t });
        q_hashes.push(audiofp::classical::WangHash { hash: 500 + i, t_anchor: t - 100 });
    }
    let query = WangFingerprint { hashes: q_hashes, frames_per_sec: 62.5 };
    let reference = WangFingerprint { hashes: ref_hashes, frames_per_sec: 62.5 };

    let m = WangMatcher::new(WangMatchConfig::default()).match_one(&query, &reference);
    assert!(m.is_match);
    assert_eq!(m.offset.frames, 100);
    assert!((m.score - 1.0).abs() < 1e-6);
    let _ = Wang::default(); // trait import stays exercised
    let _ = SampleRate::HZ_8000;
}
```

#### Prebuilt index for repeated 1:1 matches

`matcher.match_one` rebuilds the reference's inverted index on **every
call** (`SortedPostings` is sorted per match — O(R log R)). When the same
reference is matched against many queries (batch 1:1, query loops against a
fixed catalog, streaming identification), build a [`WangRefIndex`] once and
reuse it via [`WangMatcher::match_one_prebuilt`]; the per-query cost drops
to the pure O(Q log U + range) voting pass. Both entry points agree by
construction — `match_one` is exactly build-then-`match_one_prebuilt`.

```rust
use audiofp::classical::{WangHash, WangFingerprint};
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher, WangRefIndex};

fn fp(offset: u32) -> WangFingerprint {
    WangFingerprint {
        hashes: (0..40)
            .map(|i| WangHash { hash: 500 + i, t_anchor: 1000 + i * 10 + offset })
            .collect(),
        frames_per_sec: 62.5,
    }
}

fn main() {
    let reference = fp(0);
    let query = fp(100); // the query shifted +100 frames later

    let cfg = WangMatchConfig::default();
    let matcher = WangMatcher::new(cfg.clone());
    let index = WangRefIndex::build(&reference, &cfg).expect("reference has hashes");

    let m = matcher.match_one_prebuilt(&query, &index);
    // offset = t_ref − t_query = 0 − 100 = −100 frames.
    assert_eq!(m.offset.frames, -100);
    // Same result as the plain 1:1 path, without the per-call rebuild.
    assert_eq!(m, matcher.match_one(&query, &reference));
}
```

#### `WangMatchConfig`

```rust,ignore
// Reference definition (src/matching/wang.rs).
pub struct WangMatchConfig {
    pub offset_tolerance_frames: u32, // default 1
    pub min_votes: u32,               // default 5
    pub min_score: f32,               // default 0.15
    pub min_prominence: f32,          // default 5.0
    pub max_postings_per_hash: u32,   // default 100
}
```

| Field | Raise it when … | Lower it when … |
| --- | --- | --- |
| `offset_tolerance_frames` | sources have unstable framing (analogue capture) | you need precise offsets |
| `min_votes` | large catalogs (false positives cost more) | short queries (< 10 s) |
| `min_score` | near-duplicate detection wants strictness | noisy phone-mic queries |
| `min_prominence` | — | repetitive music (same hook twice) — the second hook is *background* for the first |
| `max_postings_per_hash` | — | silence-heavy catalogs (stop-hash pruning) |

The shipped defaults are calibrated on a real CC0 corpus
(`tests/threshold_calibration.rs`): every same-track cross-codec pair is
separated from every cross-track pair with the margins documented in
`ROBUSTNESS.md`.

### HaitsmaMatcher

Haitsma is a dense per-frame 32-bit code; matching is **bit-error-rate (BER)
minimisation** over alignments:

```text
BER(δ) = hamming(query, reference aligned at δ) / (overlap(δ) · 32)
```

**Two tiers:**

- **Exact BER** — slide the query over the reference at every offset
  `δ ∈ [−(q_len−1), r_len−1]`, computing Hamming distance via hardware
  `POPCNT` with an early-abort. Runs when the reference has ≤ 512 frames or
  `use_lut = false`.
- **Sub-fingerprint LUT** — for larger references, build
  `frame_u32 → [positions]` over reference frames and probe each query
  frame's exact value (plus optional 1–2 bit-flip neighbours). Haitsma's
  key property: at BER < ~0.35 at least one query frame is bit-exact, so
  the probes discover the true offset(s), which are then verified with the
  exact BER path. `O(Q + candidates·overlap)` instead of `O(Q·R)`.

**Selection and pruning are BER-normalised.** The running best is compared
by *rate*, not absolute Hamming totals, and the early-abort bound passed to
each candidate is `best_BER × overlap × 32` — a short-overlap candidate with
a small raw total but worse rate can no longer suppress a longer
better-rate alignment. BER ties keep the first-found candidate; the exact
path scans δ ascending, so ties resolve to the smallest offset.

**Decision:** `is_match = BER ≤ max_ber (0.35) && overlap ≥
min_overlap_frames (256)`. `score = 1 − BER`. Prominence samples ~40 offsets
across the range to estimate a median background BER and reports
`median_BER / (BER + ε)`.

#### `HaitsmaMatchConfig`

```rust,ignore
// Reference definition (src/matching/haitsma.rs).
pub struct HaitsmaMatchConfig {
    pub max_ber: f32,             // default 0.35 (the paper's block threshold)
    pub min_overlap_frames: u32,  // default 256 (~3.3 s at 78.125 fps)
    pub use_lut: bool,            // default true; LUT for refs > 512 frames
    pub probe_bit_flips: u8,      // default 0; 1 = +32 probes, 2 = +528 probes
}
```

> **Recall caveat for the LUT path.** With `probe_bit_flips = 0` the LUT only
> discovers an alignment when at least one query frame is *bit-exactly*
> present in the reference. Under codec noise this can miss a match the
> exhaustive path would find. Raise `probe_bit_flips`, or set `use_lut =
> false`, when matching noisy queries.

```rust
use audiofp::classical::HaitsmaFingerprint;
use audiofp::matching::{HaitsmaMatchConfig, HaitsmaMatcher, Matcher};

fn main() {
    // Deterministic pseudo-random frames.
    let frames: Vec<u32> = (0..600)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) ^ 0x5555_5555)
        .collect();

    let fp = HaitsmaFingerprint { frames: frames.clone(), frames_per_sec: 78.125 };
    let query = HaitsmaFingerprint { frames, frames_per_sec: 78.125 };

    let m = HaitsmaMatcher::new(HaitsmaMatchConfig::default()).match_one(&query, &fp);
    assert!(m.is_match);           // self-match: BER = 0
    assert_eq!(m.offset.frames, 0);
    assert!((m.score - 1.0).abs() < 1e-6);

    // Exhaustive path on a short reference:
    let cfg = HaitsmaMatchConfig { use_lut: false, ..Default::default() };
    let m2 = HaitsmaMatcher::new(cfg).match_one(&query, &fp);
    assert!(m2.is_match);
}
```

### PanakoMatcher

The only matcher that produces a meaningful `time_scale`. Matching proceeds
in five stages:

1. **Index the reference** triplets by packed hash (posting cap
   `max_postings_per_hash = 100`).
2. **Vote into a 2-D Hough accumulator.** For each query triplet that hits
   a reference triplet, compute the local scale and offset:

```text
s = (t_c_ref − t_a_ref) / max(1, t_c_query − t_a_query)    — local time scale
b = t_a_ref − s · t_a_query                                — predicted offset
s_bin = clamp( floor((s − scale_min) / scale_per_bin), 0, scale_bins − 1 )
off_key = round(b)                                         — 1-frame granularity
vote (s_bin, off_key) += 1
```

   The default grid is `s ∈ [0.80, 1.25]` over 24 bins (~2 % resolution),
   i.e. the query may run up to 25 % slower / 20 % faster than the
   reference and still match. Offset keys are 1-frame precise; the
   consolidation window (±1 scale bin, ±`offset_tolerance_frames` offset
   bins) is therefore exactly ±tol *frames*.

3. **Consolidate & peak** — neighbourhood-sum the accumulator
   (±1 scale bin × ±tol offset), find the max (first in sorted
   `(s_bin, off_key)` order on ties), require `≥ min_votes` (5).
4. **Prominence** — same `peak / (mean_rest + 1)` formula on the
   consolidated grid; require `≥ min_prominence` (5.0).
5. **RANSAC refinement** (`ransac_refine = true`, default) — the
   `(t_query, t_ref)` anchor pairs collected during voting are re-fit with
   a deterministic RANSAC (seed derived from the data, so results are
   reproducible): sample 2 pairs, fit `t_ref = s·t_query + b`, count
   inliers within ±tol frames, keep the best fit. The final `votes` is the
   inlier count; `time_scale = 1/s` clamped to `[0.5, 2.0]`.

```rust
use audiofp::classical::{PanakoConfig, PanakoFingerprint, PanakoHash};
use audiofp::matching::{Matcher, PanakoMatchConfig, PanakoMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn main() {
    // Ten reference triplets (spans of exactly 10 frames), query shifted
    // by 7 frames — spans are preserved, so the local scale stays 1.0.
    let mut r = Vec::new();
    let mut q = Vec::new();
    for i in 0..10_u32 {
        let t = 100 + i * 10;
        r.push(PanakoHash { hash: 1_000 + i, t_anchor: t, t_b: t + 5, t_c: t + 10 });
        let tq = t - 7;
        q.push(PanakoHash { hash: 1_000 + i, t_anchor: tq, t_b: tq + 5, t_c: tq + 10 });
    }
    let reference = PanakoFingerprint { hashes: r, frames_per_sec: 62.5 };
    let query = PanakoFingerprint { hashes: q, frames_per_sec: 62.5 };

    let m = PanakoMatcher::new(PanakoMatchConfig::default())
        .match_one(&query, &reference);
    assert!(m.is_match);
    assert_eq!(m.offset.frames, 7); // frame-precise, RANSAC-refined
    assert!((m.time_scale - 1.0).abs() < 1e-3);
    let _ = (PanakoConfig::default(), SampleRate::HZ_8000); // imports exercised
}
```

#### `PanakoMatchConfig`

```rust,ignore
// Reference definition (src/matching/panako.rs). Degenerate scale grids
// (scale_bins = 0, inverted or non-finite bounds) are normalized to the
// defaults at construction — in every build mode, not just debug.
pub struct PanakoMatchConfig {
    pub scale_min: f32,                // default 0.80 (internal s = ref/query)
    pub scale_max: f32,                // default 1.25
    pub scale_bins: u32,               // default 24 (~2% resolution)
    pub offset_tolerance_frames: u32,  // default 1
    pub min_votes: u32,                // default 5
    pub min_score: f32,                // default 0.15
    pub min_prominence: f32,           // default 5.0
    pub max_postings_per_hash: u32,    // default 100
    pub ransac_refine: bool,           // default true
}
```

Widen `scale_min`/`scale_max` (and rebuild your catalog) to tolerate bigger
tempo changes; `time_scale` reported to callers is `1/s` clamped to
`[0.5, 2.0]` regardless of the search grid, so genuine large stretches are
visible even when saturated.

#### Prebuilt index for repeated 1:1 Panako matches

Like [`WangRefIndex`], `PanakoRefIndex` builds the reference's inverted
HashMap once so repeated matching against the same reference skips the
per-call O(R) construction:

```rust
use audiofp::classical::{PanakoHash, PanakoFingerprint};
use audiofp::matching::{Matcher, PanakoMatchConfig, PanakoMatcher, PanakoRefIndex};

fn fp(offset: u32) -> PanakoFingerprint {
    PanakoFingerprint {
        hashes: (0..20)
            .map(|i| PanakoHash {
                hash: 500 + i,
                t_anchor: 100 + i * 10 + offset,
                t_b: 105 + i * 10 + offset,
                t_c: 110 + i * 10 + offset,
            })
            .collect(),
        frames_per_sec: 62.5,
    }
}

fn main() {
    let reference = fp(0);
    let query = fp(0); // self-match

    let cfg = PanakoMatchConfig::default();
    let matcher = PanakoMatcher::new(cfg.clone());
    let index = PanakoRefIndex::build(&reference, &cfg).expect("reference has hashes");

    let m = matcher.match_one_prebuilt(&query, &index);
    assert!(m.is_match);
    assert!((m.time_scale - 1.0).abs() < 0.1);
    // Same result as the plain 1:1 path, without the per-call rebuild.
    assert_eq!(m, matcher.match_one(&query, &reference));
}
```

### Tuning match thresholds

The shipped defaults (`WangMatchConfig`, `HaitsmaMatchConfig`,
`PanakoMatchConfig`) are calibrated on a real CC0 corpus
(`tests/threshold_calibration.rs`, margins in
[ROBUSTNESS.md](ROBUSTNESS.md#threshold-calibration)): every same-track
cross-codec pair clears them and every cross-track pair fails them,
with score margins ≥ 0.35. Recalibrate on your own catalog when your
audio differs (short clips, phone-mic noise, genre skew) — the
procedure is the same one the test uses:

1. Collect **positive** pairs (same recording, different codec/crop/noise)
   and **negative** pairs (different recordings).
2. Match every pair with the decision thresholds zeroed (accept-all
   config) and record `(score, prominence)`.
3. Take `min_pos` (worst positive) and `max_neg` (best negative) per
   field. Any threshold between them separates your data with zero
   FP/FN; pick the midpoint so both sides keep headroom.
4. If the intervals overlap, the features don't separate your data —
   loosen the query side (longer clips, lower `min_votes`) before
   touching thresholds.

```rust
use audiofp::classical::{WangFingerprint, WangHash};
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};

fn fp(hashes: &[(u32, u32)]) -> WangFingerprint {
    WangFingerprint {
        hashes: hashes
            .iter()
            .map(|&(hash, t_anchor)| WangHash { hash, t_anchor })
            .collect(),
        frames_per_sec: 62.5,
    }
}

fn main() {
    // Accept-all config: thresholds zeroed so raw score/prominence is
    // observable instead of gated into is_match.
    let probe = WangMatcher::new(WangMatchConfig {
        min_votes: 1,
        min_score: 0.0,
        min_prominence: 0.0,
        ..Default::default()
    });

    // Two "same recording" variants (shared hashes) and one unrelated track.
    // 10 shared landmarks: enough consolidated votes to clear the default
    // prominence floor (each vote lands in 3 ±tol bins, diluting the peak).
    let a = fp(&[
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
        (5, 50),
        (6, 60),
        (7, 70),
        (8, 80),
        (9, 90),
        (10, 100),
        (11, 110),
        (12, 120),
    ]);
    let b = fp(&[
        (1, 10),
        (2, 20),
        (3, 30),
        (4, 40),
        (5, 50),
        (6, 60),
        (7, 70),
        (8, 80),
        (9, 90),
        (10, 100),
        (13, 130),
        (14, 140),
    ]);
    let other = fp(&[
        (101, 10),
        (102, 20),
        (103, 30),
        (104, 40),
        (105, 50),
        (106, 60),
        (107, 70),
        (108, 80),
        (109, 90),
        (110, 100),
        (111, 110),
        (112, 120),
    ]);

    let pos = probe.match_one(&a, &b); // same track
    let neg = probe.match_one(&a, &other); // different tracks
    println!("positive: score {:.3} prom {:.1}", pos.score, pos.prominence);
    println!("negative: score {:.3} prom {:.1}", neg.score, neg.prominence);

    // Threshold = midpoint of the margin. Here the positive scores 10/12
    // shared hashes and the negative scores 0, so ~0.42 separates them
    // with headroom on both sides.
    let min_score = (pos.score + neg.score) / 2.0;
    assert!((min_score - 0.417).abs() < 0.01);

    let tuned = WangMatcher::new(WangMatchConfig {
        min_score,
        ..Default::default()
    });
    assert!(tuned.match_one(&a, &b).is_match);
    assert!(!tuned.match_one(&a, &other).is_match);
}
```

Which knob to move follows the margin shape: positives scoring low
means the evidence is weak (raise `min_votes` cautiously, or lengthen
queries); negatives scoring high means collisions (raise
`min_prominence` for Wang/Panako, lower `max_ber` for Haitsma). Move
one knob at a time and re-sweep — the margins interact.

### NeuralMatcher

Cosine-similarity matching over `NeuralFingerprint` sequences (`neural`
feature). Three aggregation strategies:

| `Aggregation` | Strategy |
| ------------- | -------- |
| `Global`      | Mean-pool both sequences to one vector each; a single cosine. Fast, weakest. |
| `SlidingMax`  | Slide the query's embeddings over the reference's; report the max mean cosine. Default. |
| `Dtw`         | Dynamic time warping — tempo-flexible sequence alignment. |

```rust,ignore
// Reference definitions (src/matching/neural.rs).
pub struct NeuralMatchConfig {
    pub min_cosine: f32,        // default 0.80 — MODEL-DEPENDENT, tune per model
    pub aggregation: Aggregation, // default SlidingMax
    pub assume_normalized: bool,  // default true (embedder L2-normalises)
}

pub enum Aggregation { Global, SlidingMax, Dtw }
```

`min_cosine` depends entirely on the embedding model's cosine distribution —
calibrate on your own data (same-track vs cross-track) before trusting the
default. Length-mismatched or empty sequences soft-fail to `NONE`.

```rust
use audiofp::matching::{Aggregation, Matcher, NeuralMatchConfig, NeuralMatcher};
use audiofp::neural::{NeuralEmbedding, NeuralFingerprint};
use audiofp::TimestampMs;

fn main() {
    // Two sequences of unit-norm 4-dim embeddings.
    let mk = |v: [f32; 4]| NeuralEmbedding { vector: v.to_vec(), t_start: TimestampMs(0) };
    let query = NeuralFingerprint {
        embeddings: vec![mk([1.0, 0.0, 0.0, 0.0]), mk([0.0, 1.0, 0.0, 0.0])],
        embedding_dim: 4,
        frames_per_sec: 1.0,
    };
    let reference = NeuralFingerprint {
        embeddings: vec![mk([1.0, 0.0, 0.0, 0.0]), mk([0.0, 1.0, 0.0, 0.0])],
        embedding_dim: 4,
        frames_per_sec: 1.0,
    };

    let m = NeuralMatcher::new(NeuralMatchConfig::default()).match_one(&query, &reference);
    assert!(m.is_match); // identical unit vectors → cosine 1.0

    let cfg = NeuralMatchConfig {
        aggregation: Aggregation::Dtw,
        assume_normalized: true,
        ..Default::default()
    };
    let m2 = NeuralMatcher::new(cfg).match_one(&query, &reference);
    assert!(m2.is_match);
}
```

### 1:N helpers and in-memory indexes

**Sequential helpers** — each reference scored independently with
`matcher.match_one`:

- `match_best(matcher, query, refs) -> Option<(usize, MatchResult)>` —
  single best `is_match` result; early-exits on a perfect score. Iterates
  references in slice order (deterministic).
- `match_ranked(matcher, query, refs) -> Vec<(usize, MatchResult)>` — every
  reference scored, sorted by score descending (ties by prominence
  descending, then index order).

**In-memory indexes** — build once, query many times. Each combines a whole
catalog into one inverted index (hash → posting list) so per-query cost is
paid once instead of per reference:

```rust
use audiofp::classical::{HaitsmaFingerprint, PanakoFingerprint, WangFingerprint};
use audiofp::matching::{
    HaitsmaIndex, HaitsmaMatchConfig, PanakoIndex, PanakoMatchConfig, WangIndex,
    WangMatchConfig,
};

fn main() {
    // --- Wang: hash → [(ref_id, t_anchor)] -----------------------------
    let refs = vec![WangFingerprint {
        hashes: vec![audiofp::classical::WangHash { hash: 7, t_anchor: 100 }],
        frames_per_sec: 62.5,
    }];
    let index = WangIndex::build(&refs, /* max_postings_per_hash */ 100);
    let query = WangFingerprint {
        hashes: vec![audiofp::classical::WangHash { hash: 7, t_anchor: 50 }],
        frames_per_sec: 62.5,
    };
    // query.hash hits ref 0 at δ = +50 frames.
    let hit = index.query(&query, &WangMatchConfig::default());
    let _ = hit; // Some((0, result)) once thresholds are met — see below

    // --- Haitsma: frame u32 → [(ref_id, pos)] LUT -----------------------
    let h_refs = vec![HaitsmaFingerprint {
        frames: vec![42, 0xFFFF_FFFF, 7],
        frames_per_sec: 78.125,
    }];
    let _ = HaitsmaIndex::build(&h_refs, 1_000);
    let _ = HaitsmaMatchConfig::default();

    // --- Panako: hash → [(ref_id, t_a, t_b, t_c)] -----------------------
    let p_refs = vec![PanakoFingerprint {
        hashes: vec![audiofp::classical::PanakoHash {
            hash: 9, t_anchor: 10, t_b: 15, t_c: 20,
        }],
        frames_per_sec: 62.5,
    }];
    let _ = PanakoIndex::build(&p_refs, 100);
    let _ = PanakoMatchConfig::default();
}
```

Build/query cost and semantics:

| Index | Build | Query | Notes |
| ----- | ----- | ----- | ----- |
| `WangIndex` | `O(Σ hashes)` | `O(Q × postings + C)` | per-reference offset histogram + consolidation, same formulas as the matcher; prominence uses the matcher's dense-range semantics |
| `HaitsmaIndex` | `O(Σ frames)` | `O(Q + C × overlap)` | probes exact frames only (no bit-flips); verifies up to **8** most-hit offsets per reference with BER-normalized bounds |
| `PanakoIndex` | `O(Σ hashes)` | `O(Q × postings + C)` | per-reference 2-D Hough; RANSAC **not** applied (coarse peak gives `time_scale`) |

Guarantees worth knowing:

- **Determinism:** candidates are visited in ascending reference id, so ties
  and the perfect-score early-exit never depend on hash-map iteration order.
- **Index vs matcher parity:** scores/prominence can differ marginally from
  a direct `match_one` (sparse vs dense histograms); the *acceptance*
  thresholds are the same. For exact 1:1 scores use the matcher.
- **Memory:** postings are `u32`-packed (8–16 bytes per posting); a
  10 000-track catalog at ~300 hashes/s is on the order of a few hundred MB.
  Raise `max_postings_per_hash` or shard for larger catalogs.
- Indexes are transient accelerators: never serialised, no file handles,
  dropped with their owning scope.

### Parallel 1:N matching (rayon)

With the `rayon` feature, [`par_match_best`] and [`par_match_ranked`] score
every reference in parallel and return **result-identical** output to
`match_best` / `match_ranked` (parallel best breaks exact ties by lowest
reference id — same winner as the sequential scan; parallel ranking
preserves the sequential order). Unlike `match_best`, the parallel scan has
**no perfect-score early exit** — every reference is scored.

```rust
use audiofp::classical::{WangHash, WangFingerprint};
use audiofp::matching::{
    Matcher, WangMatchConfig, WangMatcher, match_ranked, par_match_best, par_match_ranked,
};

fn fp(hash_base: u32, offset: u32) -> WangFingerprint {
    WangFingerprint {
        hashes: (0..40)
            .map(|i| WangHash { hash: hash_base + i, t_anchor: 1000 + i * 10 + offset })
            .collect(),
        frames_per_sec: 62.5,
    }
}

fn main() {
    // Each reference has a unique hash base — only refs[0] shares hashes with the query.
    let refs: Vec<WangFingerprint> = (0..8).map(|i| fp(500 + i * 1000, 0)).collect();
    let query = fp(500, 0); // exact copy of refs[0]
    let matcher = WangMatcher::new(WangMatchConfig::default());

    let best = par_match_best(&matcher, &query, &refs).expect("ref 0 matches");
    assert_eq!(best.0, 0);

    // Ranking is element-for-element equal to the sequential helpers.
    assert_eq!(
        par_match_ranked(&matcher, &query, &refs),
        match_ranked(&matcher, &query, &refs),
    );
}
```

The `rayon` feature also parallelises batch *fingerprinting* via
`fingerprint_batch_parallel` (see
[Async, batching, and models](#async-batching-and-models)).

Benchmarks: `cargo bench --bench matching` (Criterion; Wang/Haitsma/Panako
1:1 and a 100-reference `WangIndex`).

---

## Streaming Fingerprinters

Each classical fingerprinter has a streaming sibling; the neural embedder
has one too:

| Streaming           | `Frame`      | `latency_ms()` | Carry bound                    | `ZeroAllocStreaming` |
| ------------------- | ------------ | -------------- | ------------------------------ | -------------------- |
| `StreamingWang`     | `WangHash`   | 2 256          | `< n_fft + max_push` samples   | yes |
| `StreamingPanako`   | `PanakoHash` | 2 784          | same                           | yes |
| `StreamingHaitsma`  | `u32`        | 409            | one frame of band energies     | yes |
| `StreamingNeuralEmbedder` | `Vec<f32>` | window length (ms) | `< window_samples + max_push` | no — see below |

Each streaming variant exposes `config() -> &XConfig`, `reset()` (clear all
state — start a fresh stream), and (for the neural streamer) the
fallible-semantic `try_push` / `try_push_with` described in the
[Neural Embedder](#neural-embedder) section.

### Microphone-style usage

```rust
use audiofp::classical::StreamingWang;
use audiofp::StreamingFingerprinter;

fn main() {
    let mut s = StreamingWang::default();
    let mut all = Vec::new();

    // Synthetic 8 kHz mono chunks — swap for cpal/rodio/decoder frames.
    let chunk = vec![0.0_f32; 128]; // ~16 ms at 8 kHz
    for _ in 0..200 {
        // push returns only hashes whose full lookahead has elapsed.
        for (t, hash) in s.push(&chunk).unwrap() {
            all.push((t, hash));
        }
    }

    // End-of-stream: drain pending material. Idempotent — calling flush
    // again returns an empty Vec.
    all.extend(s.flush().unwrap());
    assert!(s.flush().unwrap().is_empty());

    println!("{} hashes total, {} ms upper-bound latency", all.len(), s.latency_ms());
}
```

### Zero-allocation callback variant

`push_with` / `flush_with` invoke a callback per frame instead of building a
`Vec` — but only impls marked [`ZeroAllocStreaming`](#zeroallocstreaming-guarantee)
do so without allocating (the trait defaults call `push` / `flush` and
iterate). On the marked types this is the allocation-free hot path for
realtime threads:

```rust
use audiofp::classical::StreamingWang;
use audiofp::StreamingFingerprinter;

fn main() {
    let mut s = StreamingWang::default();
    let chunk = vec![0.0_f32; 8_000];

    let mut count = 0usize;
    let n = s
        .push_with(&chunk, |_t, _hash| count += 1)
        .unwrap();
    let m = s
        .flush_with(|_t, _hash| count += 1)
        .unwrap();
    assert_eq!(n + m, count);
    println!("{count} hashes");
}
```

### `ZeroAllocStreaming` guarantee

`audiofp::ZeroAllocStreaming` (also in the prelude) is the bound realtime
code should require. It marks the impls whose `push_with` / `flush_with`
drain pre-allocated buffers and allocate nothing after warmup:

```rust
use audiofp::{StreamingFingerprinter, ZeroAllocStreaming};

fn mic_loop<S: ZeroAllocStreaming>(s: &mut S, chunks: &[Vec<f32>]) {
    for c in chunks {
        // Guaranteed allocation-free after warmup — safe on the audio
        // thread. A plain `StreamingFingerprinter` bound cannot promise
        // this (the defaults allocate per call).
        s.push_with(c, |_, _| {}).unwrap();
    }
}
```

Per-algorithm notes:

- **Wang / Panako** share one `StreamCore` pipeline (rolling spectrogram +
  per-second buckets + per-anchor target lists). Warmup covers buffer growth
  plus pooled target/bucket buffers and Panako's pooled triplet heap.
- **Haitsma** is a separate, simpler pipeline (STFT frame → band energies →
  sign bits); warmup covers the carry and pending-frame buffers.
- **Neural is excluded by design**: `StreamingNeuralEmbedder::Frame` is
  `Vec<f32>`, so every emit allocates through the `StreamingFingerprinter`
  interface. Its zero-alloc path is the inherent `try_push_with`, whose
  callback borrows internal scratch (`&[f32]`) — use that directly on
  realtime neural paths.

Warmup rule (all three marked types): construction may allocate, and the
first pushes may grow buffers (`Vec` amortised growth). Feed representative
chunk sizes — including one full-backlog `flush_with` — before the realtime
section; steady state after that allocates zero. This is pinned by
`tests/zero_alloc.rs` (thread-local counting allocator: warmup, then 40
pushes + flush asserting zero allocations).

### Why the latency differs

- **Haitsma** needs only the current and previous spectrogram frame →
  bounded by `n_fft / sr` plus one hop.
- **Wang / Panako** must wait for the peak-picker's ±15-frame lookahead
  *and* one full second of peaks to settle the per-second adaptive
  threshold, *and* the target zone (63 / 96 frames) before an anchor's
  hashes are final. `latency_ms()` is the sum: `(zone + neighbourhood + 1
  second)` for Wang ≈ 2 256 ms.

### Push-after-flush: contract and bounds

A mid-stream `flush` is an early end-of-stream signal: it finalises the
current 1-second bucket with whatever peaks it has. Continuation audio
appends cleanly (no duplicate hashes — the stream tracks which rows have
been emitted), but a bucket split across a flush can emit slightly more
anchors than an uninterrupted run would. For offline equivalence, flush only
at the true end of stream.

### Bit-exact equivalence (offline vs streaming)

```rust
use audiofp::classical::{StreamingWang, Wang};
use audiofp::{Fingerprinter, SampleRate, StreamingFingerprinter};

fn main() {
    // Any 8 kHz mono buffer ≥ 2 s. Silence exercises the API path; real
    // audio exercises the hashes.
    let whole_song: Vec<f32> = vec![0.0; 16_000];

    let offline = Wang::default()
        .extract(&whole_song, SampleRate::HZ_8000)
        .unwrap();

    let mut streaming = StreamingWang::default();
    let mut online = Vec::new();
    for chunk in whole_song.chunks(1024) {
        online.extend(streaming.push(chunk).unwrap().into_iter().map(|(_, h)| h));
    }
    online.extend(streaming.flush().unwrap().into_iter().map(|(_, h)| h));

    let mut a = offline.hashes;
    let mut b = online;
    a.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
    b.sort_unstable_by_key(|h| (h.t_anchor, h.hash));
    assert_eq!(a, b); // guaranteed under arbitrary chunking
}
```

### Streaming error semantics

- Classical streams never error on finite input. Non-finite samples (NaN /
  ±Inf) are **sanitised to 0.0** on `push` (streaming must not crash an
  audio callback); offline `extract` rejects them with `NonFiniteSample`
  instead. Callers who want offline-grade validation on a stream use
  **`push_strict`**: same fail-fast contract as `extract` — pre-scans and
  returns `NonFiniteSample { index }` (identical index) instead of zeroing.
  Defaulted trait method, free for every impl; realtime callbacks keep
  using infallible `push`.
- A single `push` larger than `max_push_samples` is **truncated** — excess
  samples are dropped, no error. `Some(0)` is sanitised to `Some(1)` at
  construction.
- `max_pending_anchors` (default unbounded) evicts oldest-first under
  adversarially dense input. Evicted anchors and their hashes are **lost,
  not deferred** — output can shrink below offline extraction with no error
  signal. Recommended for untrusted input: `Some(10_000)`.

---

## Fingerprint Serialization

All three classical fingerprint types round-trip through a compact,
self-describing binary format. Useful for a fingerprint cache, IPC, or
shipping enrolment artifacts to the matcher process.

### Wire format (v1)

```text
offset  size  field
0       8     magic       b"AUDIOFP\0"
8       1     version     1
9       1     alg_id      0 = Wang, 1 = Panako, 2 = Haitsma
10      4     hash_count  u32, little-endian
14      4     fps         f32, little-endian
18      …     payload     bytemuck-cast Pod hash structs, packed
```

The payload is the raw `#[repr(C)]` little-endian representation of each
hash struct (`WangHash` = 8 bytes, `PanakoHash` = 16 bytes, Haitsma frame =
4 bytes). On little-endian hosts deserialisation is a single aligned copy.

**Validation on read:** wrong magic, unsupported version, algorithm-ID
mismatch, truncated payload (`checked_mul` sizing — no 32-bit wrap), and
non-finite / non-positive `fps` all return `AfpError::Deserialize`. Trailing
bytes after the payload are ignored (forward compatibility).

### Usage

```rust
use audiofp::classical::{Wang, WangFingerprint, WangHash};
use audiofp::{Fingerprinter, SampleRate};

fn main() {
    // A fingerprint with known content (extraction works too, of course).
    let fp = WangFingerprint {
        hashes: vec![
            WangHash { hash: 0xDEAD_BEEF, t_anchor: 42 },
            WangHash { hash: 0xCAFE_BABE, t_anchor: 100 },
        ],
        frames_per_sec: 62.5,
    };

    // Serialize → 18-byte header + 2 × 8-byte hashes.
    let bytes = fp.to_bytes();
    assert_eq!(bytes.len(), 18 + 16);

    // Deserialize (validates magic, version, algorithm, fps, length).
    let restored = WangFingerprint::from_bytes(&bytes).unwrap();
    assert_eq!(fp.hashes, restored.hashes);
    assert_eq!(fp.frames_per_sec, restored.frames_per_sec);

    // Metadata on a parsed fingerprint…
    let env = restored.envelope();
    assert_eq!(env.algorithm, "wang-v1");
    assert_eq!(env.sample_rate, 8_000);
    assert_eq!(env.hash_count, 2);

    // …or straight from raw bytes without touching the payload:
    let peeked = audiofp::FingerprintEnvelope::peek(&bytes).unwrap();
    assert_eq!(peeked.algorithm, "wang-v1");
    assert_eq!(peeked.hash_count, 2);
    assert_eq!(peeked.frames_per_sec, 62.5);
}
```

### `FingerprintEnvelope`

```rust,ignore
// Reference definition (src/serial.rs).
pub struct FingerprintEnvelope {
    pub algorithm: &'static str,     // "wang-v1" | "panako-v2" | "haitsma-v1"
    pub crate_version: &'static str, // the READER's audiofp::VERSION (v1 blobs
                                     // do not persist the producer version)
    pub sample_rate: u32,            // algorithm's native rate
    pub frames_per_sec: f32,
    pub hash_count: usize,
}
```

| API | On | Notes |
| --- | --- | --- |
| `to_bytes()` | all three fingerprint types | one allocation, exact size |
| `from_bytes(&[u8])` | all three | validated; `Deserialize` on any defect |
| `envelope()` | all three | metadata of a *parsed* fingerprint |
| `FingerprintEnvelope::peek(&[u8])` | raw bytes | header-only — never touches the payload |

### Cache files (.afp)

A `.afp` file is a v1 blob (above) written to disk — exactly
`fs::write(path, fp.to_bytes())`. `audiofp::cache` (available whenever any
`std`-implying feature is on) wraps the file I/O for the *parallel extract →
serial ingest* workflow: extraction is CPU-bound and runs on rayon; storage
and indexing are single-writer and serial.

```rust
use audiofp::cache::{cache_to_file, load_from_cache};
use audiofp::classical::{Wang, WangFingerprint};
use audiofp::{Fingerprinter, SampleRate};

fn main() -> audiofp::Result<()> {
    let samples = vec![0.0_f32; 8_000 * 3];
    let fp = Wang::default().extract(&samples, SampleRate::HZ_8000)?;
    let path = std::env::temp_dir().join("t.afp");
    cache_to_file(&fp, &path)?;

    let restored: WangFingerprint = load_from_cache(&path)?;
    assert_eq!(restored.hashes, fp.hashes);
    std::fs::remove_file(&path).ok();
    Ok(())
}
```

| API | Notes |
| --- | --- |
| `cache_to_file(&fp, &path)` | any of the 3 classical fingerprints; overwrites; parent dirs not created |
| `load_from_cache::<T>(&path)` | `Io` on read failure, `Deserialize` on corrupt blob |
| `load_all_cached(&dir)` | all `*.afp` (non-recursive, path-sorted); `CachedFingerprint` enum per file — mixed-algorithm dirs work; fails on the first bad file (error names it) |

End-to-end demo: `cargo run --example cache_workflow --features rayon -- <dir>`.

---

## Audio File Decoding

Available with any `std-*` codec feature; exposed as `audiofp::io`. The
decoder is symphonia-based: it probes magic bytes (extension-less files
work), picks the default audio track, decodes packet-by-packet, converts to
`f32` (symphonia handles i16/i24/i32/f32 scaling), and downmixes
multi-channel to mono by per-frame averaging.

### `decode_to_mono`

```rust,ignore
pub fn decode_to_mono<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, u32)>;
```

Returns `(samples, native_sample_rate_hz)`.

### `decode_to_mono_at`

Decode and resample to the target rate in one step (pass-through when the
file already matches). Internally `dsp::resample::SincResampler` at default
quality (32 half-taps, Kaiser β = 8.6, 256 polyphase steps; cutoff
`min(from, to)/2` to suppress aliasing).

```rust
use audiofp::io::decode_to_mono_at;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at("song.mp3", 8_000)?; // Wang-ready
    println!("{} samples at 8 kHz ({} s)", samples.len(), samples.len() / 8_000);
    Ok(())
}
```

### `decode_to_mono_limited` / `decode_to_mono_at_limited` — OOM & hang protection

```rust,ignore
// Reference definition (src/io/decoder.rs).
pub struct DecodeLimits {
    pub max_bytes: u64,              // 0 = unlimited; checked BEFORE opening
    pub max_samples: Option<usize>,  // None = unlimited; bounds returned PCM
    pub integrity_mode: bool,        // true = fail on any corrupt packet
    pub timeout: Option<Duration>,   // wall-clock cap, checked per packet
}

impl DecodeLimits {
    pub const fn bytes(max_bytes: u64) -> Self;
    pub const fn samples(max_samples: usize) -> Self;
    pub const fn both(max_bytes: u64, max_samples: usize) -> Self;
    pub const fn strict(self) -> Self;                    // integrity_mode = true
    pub const fn with_timeout(self, d: Duration) -> Self; // timeout = Some(d)
}
```

Semantics:

- `max_bytes` is checked via `fs::metadata()` **before** opening the stream
  — a malicious 4 GB upload is rejected in < 1 µs.
- `max_samples` bounds the **returned** buffer. In the `_at` variants it is
  enforced at the native rate during decode and re-checked after resampling
  (an upsample can grow the output by the resample ratio; over-limit output
  is an `InputTooLarge` error, not a silent overshoot).
- `timeout` is checked after every packet; exceeding it returns
  `AfpError::Timeout { elapsed_ms, limit_ms }`.
- Recoverable per-packet decode errors are **skipped** (one corrupt block
  doesn't kill the file). With `integrity_mode = true` any such error is
  fatal instead — for forensic/compliance pipelines.
- A mid-stream codec-parameter change (AAC config update, track switch)
  resets the decoder and retries the packet once; only a repeated failure
  is fatal.

```rust
use std::time::Duration;

use audiofp::io::{decode_to_mono_at_limited, decode_to_mono_limited, DecodeLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Byte cap only:
    let (samples, sr) =
        decode_to_mono_limited("user_upload.mp3", DecodeLimits::bytes(50 * 1024 * 1024))?;
    println!("{} samples at {sr} Hz", samples.len());

    // Production: byte + PCM caps, wall-clock timeout, strict integrity:
    let limits = DecodeLimits::both(50 * 1024 * 1024, 30 * 60 * 48_000)
        .with_timeout(Duration::from_secs(60))
        .strict();
    let samples = decode_to_mono_at_limited("user_upload.mp3", 8_000, limits)?;
    println!("{} samples at 8 kHz", samples.len());
    Ok(())
}
```

### Supported formats

Whatever symphonia provides for the enabled features:

| Feature      | Format / codec                          | Extensions          |
| ------------ | --------------------------------------- | ------------------- |
| `std-mp3`    | MP3                                     | `.mp3`              |
| `std-aac`    | raw AAC                                 | `.aac`              |
| `std-flac`   | FLAC                                    | `.flac`             |
| `std-ogg`    | Ogg Vorbis                              | `.ogg`, `.oga`      |
| `std-wav`    | WAV / PCM                               | `.wav`              |
| `std-mp4`    | MP4/M4A (isomp4 demuxer **and** AAC)   | `.m4a`, `.mp4`      |
| `std-aiff`   | AIFF (RIFF/AIFF demuxer + PCM)          | `.aiff`, `.aif`     |
| `std-mkv`    | Matroska                                | `.mkv`, `.webm`     |
| `std-adpcm`  | ADPCM                                   | —                   |
| `std-alac`   | ALAC (in MP4/M4A)                       | `.m4a`              |

Each `std-<codec>` feature pulls its companion decoders — e.g. `std-mp4`
without the AAC codec could demux M4A but not decode it, so both are
enabled together.

### Error handling

| Failure                                     | Error variant              |
| ------------------------------------------- | -------------------------- |
| File not found / unreadable                 | `AfpError::Io(IoError)`    |
| Format unrecognised (probe fails)           | `AfpError::Io(IoError)`    |
| Per-packet decode failure                   | skipped (or `Io` in strict mode) |
| Mid-stream spec change, reset fails         | `AfpError::Io(IoError)`    |
| File exceeds `max_bytes` / `max_samples`    | `AfpError::InputTooLarge`  |
| Wall-clock `timeout` exceeded               | `AfpError::Timeout`        |

### `decode_to_mono_report` — corruption counts

Lenient decode discards information: a half-corrupt MP3 that dropped 40% of
its frames fingerprints "successfully" into a partial hash set that then
fails to match — indistinguishable from a genuine non-match. The report
variant returns the same audio with the skip counts attached, so ingest can
policy-route instead of guessing:

```rust,ignore
// Reference definition (src/io/decoder.rs).
pub struct DecodeStats {
    pub packets_total: u64,    // audio-track packets inspected
    pub packets_skipped: u64,  // recoverable failures skipped + 0-channel packets
    pub resets: u64,           // container re-syncs (ResetRequired)
}

pub struct DecodeReport {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub stats: DecodeStats,
}

pub fn decode_to_mono_report<P: AsRef<Path>>(
    path: P, limits: DecodeLimits,
) -> Result<DecodeReport>;
```

```rust
use audiofp::io::{decode_to_mono_report, DecodeLimits};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = decode_to_mono_report("user_upload.mp3", DecodeLimits::default())?;
    let skip_ratio = report.stats.packets_skipped as f64
        / report.stats.packets_total.max(1) as f64;
    if skip_ratio == 0.0 {
        println!("clean: enroll");
    } else if skip_ratio < 0.05 {
        println!("minor damage ({} skipped): enroll + flag", report.stats.packets_skipped);
    } else {
        println!("quarantine for re-upload: {:?}", report.stats);
    }
    Ok(())
}
```

Notes:

- Accept/reject behaviour is **identical** to `decode_to_mono_limited` —
  same bytes out for the same file. The report only observes what the
  lenient path already did silently.
- `packets_total` counts audio-track packets only; other-track packets in
  multi-track files are excluded.
- `integrity_mode = true` (strict) is still the fail-fast option; the report
  is the middle policy between blind leniency and aborting the file.

---

## Watermark Detection

Available with the `watermark` feature. Wraps `tract-onnx` to run an
AudioSeal-compatible detector; the model is **not** bundled.

```toml
[dependencies]
audiofp = { version = "0.4", features = ["watermark"] }
```

### `WatermarkConfig`

```rust,ignore
// Reference definition (src/watermark/mod.rs).
pub struct WatermarkConfig {
    pub model_path: String,
    pub message_bits: u8,                   // ≤ 32, default 16
    pub threshold: f32,                     // [0, 1], default 0.5
    pub sample_rate: u32,                   // default 16_000
    pub max_input_samples: Option<usize>,   // default None (unlimited)
}

impl WatermarkConfig {
    pub fn new(model_path: impl Into<String>) -> Self;
}
```

### Detect

```rust
use audiofp::watermark::{WatermarkConfig, WatermarkDetector};
use audiofp::SampleRate;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = WatermarkConfig::new("/models/audioseal_v0.2.onnx");
    let mut det = WatermarkDetector::new(cfg)?;

    // 1 s of mono 16 kHz (a real detection wants ≥ a few seconds).
    let audio = vec![0.0_f32; 16_000];
    let r = det.detect(&audio, SampleRate::HZ_16000)?;

    println!("detected={} confidence={:.3}", r.detected, r.confidence);
    println!("message bits: {:#034b}", r.message); // LSB-first
    println!("localization samples: {}", r.localization.len());
    Ok(())
}
```

Input validation order: rate check, `max_input_samples` check, empty check,
then the NaN/Inf scan — cheap rejections before the O(n) pass.

### `WatermarkResult`

| Field          | Type       | Meaning                                                                 |
| -------------- | ---------- | ----------------------------------------------------------------------- |
| `detected`     | `bool`     | `true` iff `confidence > threshold`                                     |
| `confidence`   | `f32`      | mean of the per-output detection scores                                 |
| `message`      | `u32`      | decoded message bits, **LSB-first**; bits ≥ `message_bits` are 0        |
| `localization` | `Vec<f32>` | flattened detection-score tensor (see below)                            |

**`localization` contract** — it is the flattened first ONNX output copied
element-wise with **no resampling or time-axis alignment** applied. Its
length equals whatever the model emits (often one score per input sample
for AudioSeal exports, but this is a property of the *model*, not the API).
`confidence = mean(localization)` (or 0.0 when empty). For "where in the
clip is the watermark", threshold `localization` against your model card's
documented time base.

### Model contract

1. **One input** accepting `[1, 1, T] f32` at `cfg.sample_rate`.
2. **≥ 2 outputs**, in order: `[0]` detection scores (any shape),
   `[1]` message-bit logits (any shape; the first `message_bits` values
   are read, `logit ≥ 0` ⇒ bit set).

**Plan caching:** the first `detect` at a given input length builds a typed
*and optimised* tract plan, cached per length; same-length calls reuse it,
different lengths rebuild transparently. Batch at a fixed length for best
throughput.

Obtain a model from Meta's
[AudioSeal](https://github.com/facebookresearch/audioseal) repository
(ONNX export of `audioseal_detector_16khz`), then:

```bash
cargo run --example watermark_detect --features watermark,std-wav -- /path/to/audioseal.onnx clip.wav
```

---

## Neural Embedder

Available with the `neural` feature. A generic ONNX log-mel audio embedder:
you bring the model, `audiofp` runs the front-end + inference.

```toml
[dependencies]
audiofp = { version = "0.4", features = ["neural"] }
```

### Model contract

1. **Input 0** accepts `[1, n_mels, n_frames] f32` where `n_mels` is your
   configured mel count and
   `n_frames = (window_samples − n_fft) / hop + 1` (non-centred framing).
2. **Output 0** is any tensor whose flat length is the embedding dimension
   — discovered by a probe inference at construction.

The model is typed, optimised (`into_optimized`), and made runnable **once
at construction**; per-call work is only the front-end (windowed FFT +
log-mel) and inference. Public exports that fit: VGGish, YAMNet (channel
dim removed), OpenL3, audio-MAE distillations.

### `NeuralEmbedderConfig`

```rust,ignore
// Reference definition (src/neural/embedder.rs).
pub struct NeuralEmbedderConfig {
    pub model_path: String,
    pub sample_rate: u32,        // default 16_000
    pub n_fft: usize,            // default 1024 (power of two, ≤ 2^20)
    pub hop: usize,              // default 320 (20 ms @ 16 kHz; 0 < hop ≤ n_fft)
    pub n_mels: usize,           // default 128 (1..=8192)
    pub fmin: f32,               // default 0.0
    pub fmax: f32,               // default sample_rate / 2
    pub mel_scale: MelScale,     // default Slaney (librosa default)
    pub window_kind: WindowKind, // default Hann
    pub window_secs: f32,        // default 1.0; analysis-window length (≤ 3600)
    pub hop_secs: f32,           // default 1.0; between windows (≤ window_secs)
    pub l2_normalize: bool,      // default true
    pub max_input_samples: Option<usize>, // default None
    pub max_push_samples: Option<usize>,  // default None; streaming truncate
    pub batch_size: usize,       // default 1; >1 batches offline inference
}
```

Construction validates the whole config up front — non-finite/degenerate
values, upper bounds (`n_fft ≤ 2²⁰`, `n_mels ≤ 8192`, `window_secs ≤ 3600`,
`n_mels × n_frames ≤ 2²⁸` cells) all fail with `AfpError::Config` instead
of an OOM abort later. A degenerate-but-finite config can no longer take
down the process.

### `NeuralEmbedder` (offline)

```rust
use audiofp::neural::{NeuralEmbedder, NeuralEmbedderConfig};
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Construction performs the probe inference; embedding_dim is known
    // even before you feed real audio.
    let mut emb = NeuralEmbedder::new(NeuralEmbedderConfig::new("my_model.onnx"))?;
    println!("dim={} window={} samples hop={} samples",
             emb.embedding_dim(), emb.window_samples(), emb.hop_samples());

    // 16 kHz mono PCM (rate must match cfg.sample_rate).
    let samples: Vec<f32> = vec![0.0; 16_000 * 5];
    let fp = emb.extract(&samples, SampleRate::HZ_16000)?;

    println!("{} embeddings of dim {}", fp.embeddings.len(), fp.embedding_dim);
    for e in fp.embeddings.iter().take(3) {
        println!("  t_start={} ms, |v| = {}", e.t_start.0, e.vector.len());
    }
    Ok(())
}
```

```rust,ignore
// Reference definitions (src/neural/embedder.rs).
pub struct NeuralFingerprint {
    pub embeddings: Vec<NeuralEmbedding>,
    pub embedding_dim: usize,
    pub frames_per_sec: f32,    // 1.0 / hop_secs
}

pub struct NeuralEmbedding {
    pub vector: Vec<f32>,       // L2-normalised when l2_normalize = true
    pub t_start: TimestampMs,
}
```

`n_windows = (samples.len() − window_samples) / hop_samples + 1`; tail
samples short of a window are dropped. `batch_size > 1` builds a second
`[batch, n_mels, n_frames]` plan at construction and batches full groups in
one inference; partial tails fall back to single-window calls. Output is
bit-exact regardless of batch size.

### `StreamingNeuralEmbedder` (incremental)

```rust
use audiofp::neural::{NeuralEmbedderConfig, StreamingNeuralEmbedder};
use audiofp::StreamingFingerprinter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut s = StreamingNeuralEmbedder::new(NeuralEmbedderConfig::new("my_model.onnx"))?;

    // Feed 16 kHz PCM in arbitrary chunks; push propagates inference errors.
    let chunk = vec![0.0_f32; 320]; // 20 ms
    for _ in 0..200 {
        for (t, vector) in s.push(&chunk)? {
            println!("t={} ms dim={}", t.0, vector.len());
        }
    }

    // flush drops the sub-window tail (non-centred framing cannot emit a
    // partial window) and is idempotent.
    let drained = s.flush()?;
    assert!(drained.is_empty());
    Ok(())
}
```

| Method                                                | Per-emit allocation | Errors  |
| ----------------------------------------------------- | ------------------- | ------- |
| `push(&[f32]) -> Result<Vec<(TimestampMs, Vec<f32>)>>` | one `Vec<f32>`     | `Result` |
| `try_push(&[f32]) -> Result<Vec<…>>`                  | one `Vec<f32>`     | `Result` |
| `try_push_with(&[f32], |t, &[f32]|) -> Result<usize>` | **zero** (callback borrows a reused scratch) | `Result` |

Prefer `try_push_with` on realtime paths: the embedding scratch is sized
once at construction (`embedding_dim`) and reused across every emit of
every push. The sample carry compacts once per push call, so one large
push costs O(N) total — not O(N²/hop) of front-drains. `reset()` clears
the carry and the consumed-sample counter between independent streams.

**Bit-exactness:** streaming output equals offline `extract` for the same
total input under any chunking (pinned by in-tree tests across chunk sizes
`[1, 7, 17, 256, 1024, 8191]` and overlapping `hop_secs < window_secs`).

### Errors

| Failure                                          | Variant                       |
| ------------------------------------------------ | ----------------------------- |
| Empty `model_path`, file missing                 | `ModelNotFound(_)`            |
| File present but not parseable as ONNX           | `ModelLoad(_)`                |
| Invalid config (bounds, ranges, batch_size = 0)  | `Config(_)`                   |
| Rate mismatch                                    | `UnsupportedSampleRate(_)`    |
| Buffer shorter than `window_samples`             | `AudioTooShort { … }`         |
| Tract typing / optimise / run failure            | `Inference(_)`                |

---

## DSP Primitives

Everything the fingerprinters use internally is public under
`audiofp::dsp::*` — build your own pipelines on the same blocks.

### `dsp::stft` — pre-planned STFT

```rust
use audiofp::dsp::stft::{ShortTimeFFT, StftConfig};
use audiofp::dsp::windows::WindowKind;

fn main() {
    let mut stft = ShortTimeFFT::new(StftConfig {
        n_fft: 2048,             // non-zero power of two
        hop: 512,                // 0 < hop ≤ n_fft
        window: WindowKind::Hann,
        center: true,            // librosa-style reflect padding
    });

    let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin()).collect();

    // Flat power spectrogram — single allocation, (n_frames, n_bins) row-major.
    let (power, n_frames, n_bins) = stft.power_flat(&samples);
    assert_eq!(power.len(), n_frames * n_bins);
    assert_eq!(n_bins, 2048 / 2 + 1);

    // Flat magnitude (|X|) spectrogram.
    let (mag, n_frames2, _) = stft.magnitude_flat(&samples);
    assert_eq!(mag.len(), n_frames2 * n_bins);

    // Caller-owned buffer variant of power_flat (reuse across calls):
    let mut buf = Vec::new();
    let (nf, nb) = stft.power_flat_into(&samples, &mut buf);
    assert_eq!(buf.len(), nf * nb);

    // Streaming: one pre-windowed n_fft frame → one power spectrum.
    // Zero allocations per call; reuses internal scratch.
    let frame = vec![0.0_f32; 2048];
    let mut out = vec![0.0_f32; stft.n_bins()];
    stft.process_frame_power(&frame, &mut out).unwrap();
    assert!(out.iter().all(|&p| p == 0.0)); // silence → zero power

    let mut mag_out = vec![0.0_f32; stft.n_bins()];
    stft.process_frame(&frame, &mut mag_out).unwrap();

    let _ = stft.config(); // &StftConfig
}
```

- `StftConfig::new(n_fft)` builds `hop = n_fft/4`, Hann, centred.
- `n_frames(n_samples)`: centred framing gives `1 + n_samples/hop`;
  non-centred gives `1 + (n_samples − n_fft)/hop` (0 when shorter than
  one frame).
- `ShortTimeFFT::new` panics on invalid configs; `try_new` returns
  `Result<Self, AfpError::Config>`.
- Window application, power, and magnitude kernels are 8-lane SIMD
  (`wide::f32x8`) with scalar tails.
- The deprecated `magnitude()` (per-frame `Vec<Vec<f32>>`) remains for
  compatibility; prefer `magnitude_flat`.

### `dsp::mel` — sparse mel filterbank

```rust
use audiofp::dsp::mel::{MelFilterBank, MelScale};

fn main() {
    // 128 mel bands over 0–11 kHz at sr 22 050, n_fft 2048.
    let fb = MelFilterBank::new(128, 2048, 22_050, 0.0, 11_025.0, MelScale::Slaney);
    assert_eq!(fb.n_mels, 128);
    assert_eq!(fb.n_bins(), 1025);

    // One synthetic magnitude frame (n_bins long).
    let magnitude: Vec<f32> = (0..fb.n_bins()).map(|b| (b % 17) as f32 * 0.01).collect();
    let mut log_mel = vec![0.0_f32; 128];
    fb.log_mel(&magnitude, &mut log_mel);        // log10(M·|X|² + 1e-10)

    // Power-spectrum variant — skips the per-bin square:
    let power: Vec<f32> = magnitude.iter().map(|m| m * m).collect();
    let mut log_mel2 = vec![0.0_f32; 128];
    fb.log_mel_from_power(&power, &mut log_mel2);

    // Dense row-major weight matrix (n_mels × n_bins) for inspection.
    let weights: Vec<f32> = fb.matrix();
    assert_eq!(weights.len(), 128 * 1025);
}
```

- Slaney-normalised triangles (unit area in linear Hz), matching librosa's
  `melspectrogram` defaults; `MelScale::Htk` selects the HTK formula.
- Internally CSR: each band iterates only its ~20–40 non-zero bins.
- Silence floors at `log10(1e-10) = −10.0`.
- `try_new` is the fallible constructor; panicking `new` documents its
  conditions (`n_mels > 0`, even `n_fft ≥ 2`, `0 ≤ fmin < fmax`).

### `dsp::peaks` — 2-D peak picking

```rust
use audiofp::dsp::peaks::{Peak, PeakPicker, PeakPickerConfig};

fn main() {
    // 8 frames × 8 bins with a single peak at (3, 4).
    let mut spec = vec![0.0_f32; 64];
    spec[3 * 8 + 4] = 1.0;

    let mut picker = PeakPicker::new(PeakPickerConfig {
        neighborhood_t: 1,                       // half-width, frames
        neighborhood_f: 1,                       // half-width, bins
        min_magnitude_db: f32::NEG_INFINITY,     // dB floor (input is dB here)
        min_magnitude_linear: Some(0.1),         // optional linear floor
        target_per_sec: 0,                       // 0 disables the per-second cap
    });

    let peaks: Vec<Peak> = picker.pick(&spec, 8, 8, /* frames_per_sec */ 100.0);
    assert_eq!(peaks.len(), 1);
    assert_eq!((peaks[0].t_frame, peaks[0].f_bin), (3, 4));
}
```

- A cell survives iff it clears both magnitude floors **and**
  `v ≥ rolling_max` over the `(2t+1)×(2f+1)` box — `>=` so flat plateaus
  emit every cell (matches streaming semantics).
- `target_per_sec > 0` keeps the top-K peaks per 1-second bucket
  (bucket = `floor(t_frame / frames_per_sec)`), ranked by magnitude
  descending then `(t, f)` ascending — deterministic.
- Output sorted by `(t_frame, f_bin)`. The picker is `&mut self` and pools
  all scratch — reuse one instance per producing thread or put it behind a
  `Mutex`.
- `IncrementalPeakDetector` (same module) is the streaming equivalent:
  `push_row` per spectrogram row, returns each row's 2-D max as it
  ripens; `flush` drains the tail **idempotently**.

### `dsp::resample` — linear & windowed-sinc

```rust
use audiofp::dsp::resample::{SincQuality, SincResampler, linear};

fn main() {
    let x: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.05).sin()).collect();

    // Linear: cheap, aliases on downsamples — baseline only.
    let y = linear(&x, 44_100, 8_000);

    // Sinc (default: 32 half-taps, Kaiser β 8.6, 256 polyphase steps).
    let r = SincResampler::new(44_100, 8_000);
    let y2 = r.process(&x);
    let _ = r.quality(); // &SincQuality

    // Higher quality (≈ -120 dB stopband).
    let hq = SincResampler::with_quality(
        44_100,
        8_000,
        SincQuality { half_taps: 64, kaiser_beta: 12.0, polyphase_steps: 256 },
    );
    let y3 = hq.process(&x);

    // Hot-loop, allocation-free variant: reuse the output Vec.
    let mut out = Vec::new();
    for chunk in [x.as_slice(), y2.as_slice()] {
        hq.process_into(chunk, &mut out);
        let _ = out.len(); // capacity preserved across chunks
    }
    let _ = y;
    let _ = y3;
}
```

- Output length is `ceil(n_in · to / from)`; out-of-range taps are
  zero-padded; DC gain normalised to 1.
- Cutoff is `min(from, to) / 2` in the input's frame — suppresses aliasing
  on downsamples and images on upsamples.
- `try_new` / `try_with_quality` are the fallible constructors.
- **Do not use `linear` for production downsamples** — the aliasing
  measurably degrades fingerprints.

### `dsp::windows` — periodic windows

```rust
use audiofp::dsp::windows::{make_window, WindowKind};

fn main() {
    let hann = make_window(WindowKind::Hann, 1024);
    assert_eq!(hann.len(), 1024);
    assert!((hann[0] - 0.0).abs() < 1e-6);  // periodic: w[0] = 0, not w[1] = 0
    let _ = make_window(WindowKind::Hamming, 1024);
    let _ = make_window(WindowKind::Blackman, 1024);
}
```

Periodic (period `N`, not `N−1`) — matches librosa /
`scipy.signal.get_window(..., fftbins=True)`.

---

## Async, batching, and models

### Async usage

`audiofp` is synchronous. From `tokio` (or any runtime), offload the
CPU-heavy work onto the blocking pool:

```rust
use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

async fn fingerprint_blocking(
    samples: Vec<f32>,
) -> Result<audiofp::classical::WangFingerprint, audiofp::AfpError> {
    tokio::task::spawn_blocking(move || {
        let mut wang = Wang::default();
        wang.extract(&samples, SampleRate::HZ_8000)
    })
    .await
    .expect("blocking task join")
}
```

Keep extraction off the async executor threads — STFT + peak picking are
CPU-bound and would stall the runtime.

#### Sharing an extractor across tasks

`extract` takes `&mut self`, so a shared fingerprinter needs a lock —
or skip sharing entirely with one extractor per task. Both are safe:
every public type is asserted `Send + Sync` at compile time
(`send_sync_assertions` in `src/lib.rs`), so they can cross task and
thread boundaries:

```rust
use std::sync::{Arc, Mutex};

use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

// One extractor behind a lock: FFT plan and scratch stay warm, tasks
// serialise on the mutex. Fine when extraction is not the bottleneck.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wang = Arc::new(Mutex::new(Wang::default()));
    let w2 = Arc::clone(&wang);
    let samples = vec![0.0_f32; 8_000 * 3];
    let fp = w2.lock().unwrap().extract(&samples, SampleRate::HZ_8000)?;
    assert!(fp.hashes.is_empty()); // silence → no hashes
    Ok(())
}
```

Prefer one extractor per `spawn_blocking` task when throughput matters
— no lock contention, and each task keeps its own warm scratch:

```rust
use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

async fn fingerprint_blocking_each(
    samples: Vec<f32>,
) -> Result<audiofp::classical::WangFingerprint, audiofp::AfpError> {
    tokio::task::spawn_blocking(move || {
        let mut wang = Wang::default(); // fresh plan per task
        wang.extract(&samples, SampleRate::HZ_8000)
    })
    .await
    .expect("blocking task join")
}
```

Rules of thumb: decode (`symphonia` packets + resample) and extract go
on the blocking pool; 1:1 matching is single-digit milliseconds and can
run inline unless you batch it. Streaming `push` follows the same
`&mut self` rule — drive one streamer per source from a single task.

### Batching files

Reuse one fingerprinter across paths — the FFT plan, window, and scratch
buffers stay warm:

```rust
use std::path::PathBuf;

use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

fn enroll_batch(paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    let mut wang = Wang::default();
    for path in paths {
        let samples = decode_to_mono_at(path, 8_000)?;
        let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
        println!("{} → {} hashes", path.display(), fp.hashes.len());
        // your_store.insert(track_id, &fp.hashes);
    }
    Ok(())
}
```

For process-level parallelism enable the `rayon` feature and use
`audiofp::fingerprint_batch_parallel`, which fingerprints many buffers
across cores (it parallelises extraction only — matching stays sequential
by design; use the indexes for large 1:N). To decouple parallel extraction
from single-writer storage/indexing, cache each result to a `.afp` file
([Cache files](#cache-files-afp)) and ingest the directory serially.

#### Enrolling a directory: skip corrupt, abort on resource pressure

The batch loop above bails on the first error (`?` inside the loop) —
wrong for a 10k-file ingest where one corrupt upload must not kill the
run. Classify per-file failures instead: **skip** corrupt content,
**abort** on resource signals (the rest of the directory will hit the
same wall):

```rust
use std::path::{Path, PathBuf};

use audiofp::classical::Wang;
use audiofp::io::{DecodeLimits, decode_to_mono_at_limited};
use audiofp::{AfpError, Fingerprinter, SampleRate};

fn enroll_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let limits = DecodeLimits::both(50_000_000, 8_000 * 600).strict();
    let mut wang = Wang::default();
    let (mut ok, mut skipped, mut failed) = (0usize, 0usize, 0usize);

    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    paths.sort();
    for path in &paths {
        let r: Result<(), AfpError> = (|| {
            let samples = decode_to_mono_at_limited(path, 8_000, limits)?;
            let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
            println!("{} → {} hashes", path.display(), fp.hashes.len());
            // your_store.insert(track_id, &fp.hashes);
            Ok(())
        })();
        match r {
            Ok(()) => ok += 1,
            // Corrupt content: log and continue with the next file.
            Err(AfpError::Io(_)) | Err(AfpError::Deserialize(_)) => {
                eprintln!("{}: corrupt, skipping", path.display());
                skipped += 1;
            }
            // Too short / wrong shape for this fingerprinter: data issue,
            // not a crash — skip unless your catalog guarantees otherwise.
            Err(AfpError::AudioTooShort { .. }) | Err(AfpError::NonFiniteSample { .. }) => {
                eprintln!("{}: unusable audio, skipping", path.display());
                skipped += 1;
            }
            // Resource pressure or config error: abort, the rest of the
            // directory will hit the same wall.
            Err(e @ (AfpError::InputTooLarge { .. } | AfpError::Timeout { .. }
                | AfpError::Config(_))) => {
                eprintln!("{}: aborting ingest: {e}", path.display());
                failed += 1;
                break;
            }
            Err(e) => {
                eprintln!("{}: skipping: {e}", path.display());
                skipped += 1;
            }
        }
    }
    println!("enrolled={ok} skipped={skipped} failed={failed}");
    Ok(())
}
```

Notes:

- `.strict()` makes per-packet decode errors fatal *for that file* so a
  half-decoded buffer never enters your index silently; the loop above
  still continues with the next file.
- `InputTooLarge` carries `{ limit, provided }` — log both when
  aborting so the operator knows which cap to raise.
- `AfpError` is `#[non_exhaustive]`: keep the trailing catch-all arm so
  future variants don't break your enroll loop on upgrade.

### Model sourcing

Neither `watermark` nor `neural` ships ONNX weights:

| Feature    | Model source                                                                             |
| ---------- | ---------------------------------------------------------------------------------------- |
| `watermark`| Meta [AudioSeal](https://github.com/facebookresearch/audioseal) ONNX detector export      |
| `neural`   | Any log-mel embedder matching the [model contract](#neural-embedder) (VGGish, YAMNet, …) |

---

## Error Handling

All fallible APIs return `Result<T, AfpError>`; `AfpError` is
`#[non_exhaustive]` — always keep a catch-all arm:

```rust,ignore
// Reference definition (src/error.rs) — abridged.
pub enum AfpError {
    AudioTooShort { needed: usize, got: usize },
    UnsupportedSampleRate(u32),
    UnsupportedChannels(u16),
    ModelNotFound(String),
    ModelLoad(String),
    Inference(String),
    BufferOverrun { dropped: usize },
    NonFiniteSample { index: usize },   // first offending sample's index
    InputTooLarge { limit: usize, provided: usize },
    Config(String),
    Deserialize(String),
    Timeout { elapsed_ms: u64, limit_ms: u64 },  // std only
    Io(IoError),                                  // std only
}
```

`IoError` carries `path: Option<PathBuf>`, `kind: std::io::ErrorKind`, and
the underlying `std::io::Error` as `source`.

**PCM policy:**

- Offline `extract` and watermark `detect` **reject** NaN/Inf with
  `NonFiniteSample { index }` (first offending index).
- Streaming `push` **sanitises** non-finite samples to `0.0` — an audio
  callback must not die mid-stream.

```rust
use audiofp::classical::Wang;
use audiofp::{AfpError, Fingerprinter, SampleRate};

fn main() {
    let samples = vec![0.0_f32; 8_000];
    let mut wang = Wang::default();

    match wang.extract(&samples, SampleRate::HZ_8000) {
        Ok(fp) => println!("{} hashes", fp.hashes.len()),
        Err(AfpError::AudioTooShort { needed, got }) => {
            eprintln!("need {needed} samples ({:.1} s), got {got}",
                      needed as f32 / 8_000.0);
        }
        Err(AfpError::UnsupportedSampleRate(hz)) => {
            eprintln!("Wang needs 8 kHz, got {hz} — resample first");
        }
        Err(e) => eprintln!("unexpected: {e}"),
    }
}
```

---

## Performance Tips

1. **Reuse the fingerprinter across calls.** `Wang::new` allocates an FFT
   plan, window table, and scratch; recreating per file wastes all of it.
   Same for `Panako`, `Haitsma`, `WatermarkDetector`, and the peak picker.

2. **Pick the algorithm for the workload.** Wang for music ID; Panako when
   tempo robustness matters; Haitsma for smallest fingerprints and lowest
   streaming latency; neural for semantic/cover similarity.

3. **Tune `fan_out` / `peaks_per_sec` to your index.** Wang 5–10 is the
   useful range (3 for tight storage, ≥ 15 wastes index space). Halving
   `peaks_per_sec` roughly halves both hashes and recall.

4. **Never use the `linear` resampler in production** — aliasing on
   downsamples degrades fingerprint quality. `SincResampler` default
   quality is the right default.

5. **`mimalloc`** installs `mimalloc::MiMalloc` process-wide when your own
   binary doesn't pick an allocator:

   ```toml
   [dependencies]
   audiofp = { version = "0.4", features = ["mimalloc"] }
   ```

6. **The streaming hot path is incremental and allocation-free** after
   warmup: per-push CPU is proportional to the new samples only. The
   neural streamer additionally reuses one embedding scratch and compacts
   its carry once per push.

7. **Build with LTO.** The crate ships `lto = "fat"`, `codegen-units = 1`
   in its release profile; if you consume it as a library, set the same in
   your binary's `[profile.release]` — cross-crate inlining of the DSP
   kernels is worth ~10–15 %.

8. **Batch neural inference.** `batch_size > 1` amortises per-run ONNX
   overhead; fixed-length watermark inputs reuse the cached plan.

9. **Consider `target-cpu=native` for a single-machine deployment.** The
   SIMD kernels (`wide::f32x8`) and the FFT dispatch at runtime on a
   portable baseline; compiling for the host CPU additionally unlocks
   hardware POPCNT, AVX2, and FMA. Measure before adopting — the win is
   workload-dependent (Haitsma BER and RANSAC inlier counting benefit
   most).

   ```bash
   RUSTFLAGS="-C target-cpu=native" cargo build --release
   ```

   The crate deliberately does **not** ship this in a `.cargo/config.toml`:
   a binary built with `target-cpu=native` can fault with an illegal
   instruction on a different CPU, so it is unsafe for portable release
   artifacts and container images that may run on heterogeneous hosts.
   Enable it only when the build host and the run host are the same
   microarchitecture.

---

## Feature Flags

Default = `[]` (no_std + alloc, no codecs):

| Feature      | Brings in                                                                    |
| ------------ | ---------------------------------------------------------------------------- |
| `std`        | Symphonia itself, no codecs; also enables `audiofp::cache` (`.afp` fingerprint files). Bare `std` without any codec/`neural`/… feature is a `compile_error!` when `audiofp::io` is touched. |
| `std-mp3`    | MP3 (`symphonia/mp3`)                                                        |
| `std-aac`    | raw AAC (`symphonia/aac`)                                                    |
| `std-flac`   | FLAC (`symphonia/flac`)                                                      |
| `std-ogg`    | Ogg + Vorbis (`symphonia/ogg`, `symphonia/vorbis`)                           |
| `std-wav`    | WAV + PCM (`symphonia/wav`, `symphonia/pcm`)                                 |
| `std-mp4`    | AAC-in-MP4 / ISO-BMFF (`symphonia/isomp4`, `symphonia/aac`)                  |
| `std-aiff`   | AIFF + PCM payloads (`symphonia/aiff`, `symphonia/pcm`)                      |
| `std-mkv`    | Matroska (`symphonia/mkv`)                                                   |
| `std-adpcm`  | ADPCM (`symphonia/adpcm`)                                                    |
| `std-alac`   | ALAC in MP4/M4A (`symphonia/alac`, `symphonia/isomp4`)                       |
| `all-codecs` | every codec feature above — the pre-0.4.0 monolithic `std`                    |
| `rayon`      | parallel batch fingerprinting via `fingerprint_batch_parallel` + parallel 1:N matching via `par_match_best` / `par_match_ranked` (implies std) |
| `watermark`  | `tract-onnx`; enables `audiofp::watermark` (implies `std`)                   |
| `neural`     | `tract-onnx`; enables `audiofp::neural` (implies `std`)                      |
| `mimalloc`   | `mimalloc` as process-wide `#[global_allocator]` (implies `std`)             |

`all-codecs` deliberately excludes the heavyweight subsystems (`neural`,
`watermark`, `rayon`, `mimalloc`) — enable those explicitly.

```toml
# WAV-only decoding (typical embedded-service case):
audiofp = { version = "0.4", features = ["std-wav"] }

# Everything the pre-0.4.0 `std` feature decoded:
audiofp = { version = "0.4", features = ["all-codecs"] }

# Minimal no_std + alloc build (no io at all):
audiofp = { version = "0.4", default-features = false }

# Watermark only:
audiofp = { version = "0.4", default-features = false, features = ["watermark"] }
```

---

## no_std / Embedded

The DSP primitives, classical fingerprinters, matching, and serialisation
compile under `no_std + alloc`:

```rust
#![no_std]
extern crate alloc;

use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

fn fingerprint_here(samples: &[f32]) -> audiofp::Result<audiofp::classical::WangFingerprint> {
    let mut wang = Wang::default();
    wang.extract(samples, SampleRate::HZ_8000)
}
```

| Module                | no_std status                                                       |
| --------------------- | ------------------------------------------------------------------- |
| `audiofp::dsp` / `classical` / `matching` / `serial` | host-only no_std (rustfft transitively reaches std; your crate itself is `#![no_std]`) |
| `audiofp::io`         | requires a `std-*` codec feature or `all-codecs`                    |
| `audiofp::neural` / `watermark` | require `std`                                              |

> **Bare-metal caveat.** `rustfft` (the STFT backend) transitively pulls
> `num-traits` with `std`, so the no_std build runs on hosted targets where
> dependencies may reach std even though *your* crate is `no_std`. True
> Cortex-M support needs a `microfft`-backed FFT swap (on the roadmap).

---

## Determinism Guarantees

- **Identical inputs → identical outputs.** Same audio, fingerprinter, and
  config produce bit-identical hashes on every call, every run, every
  supported target. No RNG, no time, no hash-order leakage anywhere in the
  extract path — including streaming under arbitrary chunking.
- **Deterministic matching.** All matchers and indexes break ties by total
  orders (offset, `(t, f)` position, reference id) — never by hash-map
  iteration order. Repeated 1:N queries return the same winner every run.
- **Stable algorithm IDs.** `name()` returns a versioned string
  (`"wang-v1"`); a future change to hash bytes bumps the suffix.
- **Stable hash layouts.** Bit positions in `WangHash::hash`,
  `PanakoHash::hash`, Haitsma frames, and the v1 serialisation format are
  stable across patch and minor versions inside `0.x`.
- **Serialisation is self-describing** and validated on read; blobs written
  by any 0.4.x reader remain readable (trailing bytes are ignored for
  forward compatibility).

---

## License

MIT. See [LICENSE](LICENSE).

## Examples

Runnable starters under `examples/`:

| Example            | Features | Command |
| ------------------ | -------- | ------- |
| `enroll_file`      | `std-mp3,std-flac,std-ogg,std-wav,std-mp4` | `cargo run --example enroll_file --features std-mp3,std-flac,std-ogg,std-wav,std-mp4 -- song.flac` |
| `match_two_files`  | `std-mp3,std-flac,std-ogg,std-wav,std-mp4` | `cargo run --example match_two_files --features std-mp3,std-flac,std-ogg,std-wav,std-mp4 -- a.flac b.mp3` |
| `compare_algorithms` | `std-mp3,std-flac,std-ogg,std-wav,std-mp4` | `cargo run --example compare_algorithms --features std-mp3,std-flac,std-ogg,std-wav,std-mp4 -- song.flac` |
| `stream_buffer`    | none | `cargo run --example stream_buffer` |
| `dsp_starter`      | none | `cargo run --example dsp_starter` |
| `neural_embed`     | `neural` | `cargo run --example neural_embed --features neural -- model.onnx` |
| `watermark_detect` | `watermark,std-wav` | `cargo run --example watermark_detect --features watermark,std-wav -- model.onnx [audio.wav]` |

## Links

- [Crates.io](https://crates.io/crates/audiofp)
- [Documentation](https://docs.rs/audiofp)
- [Repository](https://github.com/themankindproject/audiofp)
- [Changelog](CHANGELOG.md)
- [ROBUSTNESS.md](ROBUSTNESS.md) — measured cross-codec/noise margins
