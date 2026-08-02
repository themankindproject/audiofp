# audiofp — Codebase Context

> Auto-generated deep context for AI assistants. Avoids re-reading the entire codebase each session.

## Project Overview

**audiofp** is a pure-Rust audio fingerprinting SDK (v0.3.8) providing three classical fingerprinting algorithms (Wang, Panako, Haitsma–Kalker), a generic ONNX neural embedder, and an AudioSeal-compatible watermark detector. Every algorithm has both offline (whole-buffer) and streaming (incremental) variants with **bit-exact parity** between them.

- **Crate**: `audiofp` on crates.io
- **Edition**: 2024
- **MSRV**: 1.93.0
- **License**: MIT
- **Author**: bravo1goingdark
- **Repository**: github.com/themankindproject/audiofp

## Project Structure

```
├── Cargo.toml              # Workspace root, single lib crate + fuzz member
├── Cargo.lock
├── src/
│   ├── lib.rs              # Crate root, feature gates, re-exports
│   ├── error.rs            # AfpError enum (#[non_exhaustive])
│   ├── fp.rs               # Fingerprinter + StreamingFingerprinter traits
│   ├── types.rs            # SampleRate, AudioBuffer, TimestampMs
│   ├── pcm.rs              # PCM validation helpers (reject_non_finite, sanitize)
│   ├── prelude.rs          # Convenience re-exports
│   ├── classical/
│   │   ├── mod.rs          # Re-exports Wang, Panako, Haitsma + streaming
│   │   ├── wang.rs         # Wang landmark fingerprinter (~67 KB)
│   │   ├── panako.rs       # Panako triplet fingerprinter (~63 KB)
│   │   └── haitsma.rs      # Haitsma–Kalker band-power (~34 KB)
│   ├── dsp/
│   │   ├── mod.rs          # DB_LOG2_FACTOR constant, sub-module exports
│   │   ├── stft.rs         # ShortTimeFFT (realfft, SIMD windowing)
│   │   ├── mel.rs          # MelFilterBank (HTK/Slaney, sparse CSR)
│   │   ├── peaks.rs        # PeakPicker + IncrementalPeakDetector (Lemire deque)
│   │   ├── resample.rs     # SincResampler (Kaiser windowed-sinc, polyphase)
│   │   └── windows.rs      # Hann/Hamming/Blackman window generation
│   ├── io/
│   │   ├── mod.rs          # Re-exports decode functions
│   │   └── decoder.rs      # Symphonia decode + downmix + resample
│   ├── neural/
│   │   ├── mod.rs          # Re-exports NeuralEmbedder, StreamingNeuralEmbedder
│   │   ├── embedder.rs     # Offline ONNX embedder (tract-onnx)
│   │   ├── streaming.rs    # Streaming ONNX embedder with carry buffer
│   │   ├── frontend.rs     # LogMelFrontend (STFT→mel pipeline, pub(crate))
│   │   └── test_support.rs # Passthrough model builder for tests
│   └── watermark/
│       ├── mod.rs          # Re-exports WatermarkDetector
│       └── detector.rs     # AudioSeal ONNX detector (lazy plan caching)
├── tests/
│   ├── assets/             # Real audio: galway.*, freak.*, piano.ogg, speech.ogg
│   ├── goldens/            # Binary golden files for regression
│   ├── codec_roundtrip.rs  # Same-song hash survival across codecs
│   ├── codec_extended.rs   # AIFF, two-track discrimination, SR ladder
│   ├── real_audio.rs       # Noise/lowpass robustness, streaming==offline
│   ├── real_audio_e2e.rs   # Segment matching, gain invariance, determinism
│   ├── real_audio_golden.rs# Bit-exact regression on real audio
│   ├── generate_real_goldens.rs # Golden file generator (#[ignore])
│   ├── property.rs         # Proptest: streaming≡offline, NaN safety, tempo
│   ├── regression.rs       # Bit-exact synthetic golden regression
│   ├── robustness.rs       # Synthetic noise/lowpass survival
│   ├── watermark.rs        # Config validation (no model shipped)
│   └── neural.rs           # Config validation (no model shipped)
├── benches/
│   ├── extract.rs          # Criterion: offline extract (2s/5s/30s)
│   ├── streaming.rs        # Criterion: streaming push (small/large chunks)
│   └── neural_frontend.rs  # Criterion: STFT→mel, tensor fill, L2 norm
├── examples/
│   ├── enroll_file.rs      # Fingerprint one file
│   ├── match_two_files.rs  # Compare two files by hash overlap
│   ├── compare_algorithms.rs # All 3 algos side-by-side
│   ├── stream_buffer.rs    # StreamingWang demo
│   ├── dsp_starter.rs      # STFT→mel→peaks (no file, no features)
│   ├── neural_embed.rs     # BYO ONNX model embedding
│   └── watermark_detect.rs # AudioSeal detection
├── fuzz/
│   ├── Cargo.toml          # libfuzzer-sys + arbitrary
│   └── fuzz_targets/       # 7 targets (hash roundtrip, streaming equiv, resampler)
├── .github/workflows/ci.yml # 11-job CI (fmt, clippy, test, golden, audit, proptest, msrv, no-std, deny, fuzz)
├── deny.toml               # License/ban/source policy
├── rustfmt.toml            # edition=2024, max_width=100
├── clippy.toml             # Bans Instant::now in lib code
└── rust-toolchain.toml     # Pinned 1.93.0 stable
```

## Cargo Features

| Feature    | Default | Dependencies          | Enables                                      |
|------------|:-------:|-----------------------|----------------------------------------------|
| `std`      | ✅      | `symphonia`           | `io` module (file decoding)                  |
| `neural`   |         | `tract-onnx` + `std`  | `neural` module (ONNX embedder)              |
| `watermark`|         | `tract-onnx` + `std`  | `watermark` module (AudioSeal detector)      |
| `rayon`    |         | `rayon` + `std`        | `fingerprint_batch_parallel()`               |
| `mimalloc` |         | `mimalloc` + `std`     | Global allocator override                    |

## Dependencies

### Runtime
- `thiserror 2.0` — error derive (no_std compatible)
- `libm 0.2` — math functions for no_std
- `num-traits 0.2` + `num-complex 0.4` — numeric abstractions
- `realfft 3.5` — real-input FFT (wraps rustfft)
- `bytemuck 1.25` — zero-copy Pod/Zeroable derives
- `symphonia 0.6` (optional) — audio codec decoding
- `tract-onnx 0.23` (optional) — ONNX model inference
- `rayon 1.10` (optional) — parallel iteration
- `mimalloc 0.1` (optional) — global allocator

### Dev
- `approx 0.5` — float comparison
- `hound 3.5` — WAV writing for tests
- `criterion 0.5` — benchmarking
- `proptest 1.5` — property-based testing

## Core Public API

### Types (`src/types.rs`)

```rust
pub struct SampleRate(pub NonZeroU32);  // Constants: HZ_5000, HZ_8000, HZ_11025, HZ_16000, HZ_22050, HZ_44100, HZ_48000
pub struct AudioBuffer<'a> { pub samples: &'a [f32], pub rate: SampleRate }
pub struct TimestampMs(pub u64);  // Milliseconds since stream start
```

### Traits (`src/fp.rs`)

```rust
pub trait Fingerprinter {
    type Output;
    type Config: Clone + Send + Sync;
    fn name(&self) -> &'static str;           // e.g. "wang-v1", "panako-v2", "haitsma-v1"
    fn config(&self) -> &Self::Config;
    fn required_sample_rate(&self) -> u32;
    fn min_samples(&self) -> usize;
    fn extract(&mut self, audio: AudioBuffer<'_>) -> Result<Self::Output>;
}

pub trait StreamingFingerprinter {
    type Frame;
    fn required_sample_rate(&self) -> u32;
    fn push(&mut self, samples: &[f32]) -> Vec<(TimestampMs, Self::Frame)>;
    fn flush(&mut self) -> Vec<(TimestampMs, Self::Frame)>;
    fn latency_ms(&self) -> u32;
    fn push_with<F>(&mut self, samples: &[f32], callback: F) -> usize;   // zero-alloc variant
    fn flush_with<F>(&mut self, callback: F) -> usize;                   // zero-alloc variant
}

// rayon feature:
pub fn fingerprint_batch_parallel<F, T>(items, make_fingerprinter) -> Vec<(T, Result<F::Output>)>;
```

### Error (`src/error.rs`)

```rust
#[non_exhaustive]
pub enum AfpError {
    AudioTooShort { needed, got },
    UnsupportedSampleRate(u32),
    UnsupportedChannels(u16),
    ModelNotFound(String),
    ModelLoad(String),
    Inference(String),
    InputTooLarge { limit, provided },
    BufferOverrun { dropped },
    NonFiniteSample { index },
    Config(String),
    Io(IoError),           // std feature
}
pub type Result<T> = core::result::Result<T, AfpError>;
```

### PCM Helpers (`src/pcm.rs`, pub(crate))
- `reject_non_finite(samples)` — offline paths reject NaN/Inf
- `extend_sanitized(carry, samples)` — streaming paths replace NaN/Inf with 0.0
- `truncate_push(samples, max)` — streaming push size limiter

## Classical Fingerprinters (`src/classical/`)

### Wang (`wang.rs`) — Shazam-style Landmark Pairs

**Constants**: N_FFT=1024, HOP=128, SR=8000 Hz, FPS=62.5, FREQ_BUCKETS=512, PEAK_NEIGHBOURHOOD=15

**Hash type**:
```rust
#[repr(C)] pub struct WangHash { pub hash: u32, pub t_anchor: u32 }  // 8 bytes, Pod
// hash bits: f_a_q(9) | f_b_q(9) | Δt(14)
```

**Config** (`WangConfig`): fan_out=10, target_zone_t=63, target_zone_f=64, peaks_per_sec=30, min_anchor_mag_db=-50.0, max_input_samples=14.4M, max_hashes=500K

**Offline pipeline**:
1. Reject non-finite → check limits/rate/length
2. Power STFT (skip sqrt) → dB via `DB_LOG2_FACTOR * log2(max(v, 1e-12))`
3. PeakPicker: 31×31 Lemire rolling-max, 30/s adaptive cap
4. `build_hashes()`: linear-insert top-K targets per anchor (faster than BinaryHeap for K≤16)
5. Sort by `(t_anchor, hash)`

**Streaming** (`StreamingWang`):
- Rolling spectrogram (31-row ring buffer)
- `IncrementalPeakDetector` (per-column Lemire deques)
- Per-second bucket thresholding (sorted Vec, ≤3 entries)
- `VecDeque<PendingAnchor>` with BinaryHeap for top-K targets
- Latency: ~2256 ms
- **Bit-exact** with offline `extract`

---

### Panako (`panako.rs`) — Tempo-Invariant Triplet Hashes

**Constants**: Same STFT front-end as Wang. Target zones wider: t=96, f=96.

**Hash type**:
```rust
#[repr(C)] pub struct PanakoHash { pub hash: u32, pub t_anchor: u32, pub t_b: u32, pub t_c: u32 }  // 16 bytes, Pod
// hash bits: sign(2) | mag_order(2) | β(5) | Δf_ab(8) | Δf_bc(8) | reserved(7)
// β = round((t_c - t_b) / (t_c - t_a) * 31) — tempo-invariant ratio
```

**Config** (`PanakoConfig`): fan_out=5, target_zone_t=96, target_zone_f=96, peaks_per_sec=30

**Offline pipeline**: Same front-end → triplet enumeration with suffix-max early-exit → pack_triplet → sort by `(t_anchor, t_b, t_c, hash)`

**Streaming** (`StreamingPanako`):
- Same incremental peak detection
- Stores ALL targets per anchor (capped at 2×fan_out), triplet enumeration at emit time
- Uses strict inequality for zone bounds
- Latency: ~2784 ms

---

### Haitsma (`haitsma.rs`) — Band-Power Sign Bits

**Constants**: N_FFT=2048, HOP=64, SR=5000 Hz, FPS=78.125, N_BANDS=33

**Output**: `HaitsmaFingerprint { frames: Vec<u32>, frames_per_sec: 78.125 }` — one u32 per frame

**Config** (`HaitsmaConfig`): fmin=300, fmax=2000, max_input_samples=9M

**Algorithm**:
1. Power STFT
2. Sum power in 33 log-spaced bands [300–2000 Hz] → E[n][b]
3. Frame n≥1: `bit[b] = ((E[n][b]−E[n][b+1]) − (E[n-1][b]−E[n-1][b+1])) > 0`
4. Pack 32 bits: band 0 → bit 31 (MSB-zero)

**Streaming** (`StreamingHaitsma`):
- Trivially incremental (only needs previous frame's 33 energies)
- No peak detection, no deques
- Latency: ~409 ms

---

### Performance Summary

| Algorithm | 30s audio | Realtime× | Streaming latency |
|-----------|-----------|-----------|-------------------|
| Wang      | 99 ms     | 303×      | 2256 ms           |
| Panako    | 104 ms    | 288×      | 2784 ms           |
| Haitsma   | 47 ms     | 638×      | 409 ms            |

## DSP Primitives (`src/dsp/`)

### STFT (`stft.rs`)

```rust
pub struct StftConfig { pub n_fft, pub hop, pub window: WindowKind, pub center: bool }
pub struct ShortTimeFFT { /* realfft plan (Arc), precomputed window, reusable scratch */ }
```

**Key methods**: `magnitude()`, `magnitude_flat()`, `power_flat()`, `power_flat_into()`, `process_frame()`, `process_frame_power()`

**Implementation details**:
- Real-input FFT via `realfft` crate (only n_fft/2+1 bins)
- SIMD-accelerated windowing: AVX2 (x86_64), NEON (aarch64), scalar fallback
- Center framing: numpy-style reflect padding
- All scratch buffers pre-allocated at construction

### Mel Filterbank (`mel.rs`)

```rust
pub enum MelScale { Htk, Slaney }
pub struct MelFilterBank { pub n_mels, pub n_fft, pub sr, pub fmin, pub fmax, pub scale, /* dense + CSR sparse */ }
```

**Key methods**: `log_mel(magnitude, out)`, `log_mel_from_power(power, out)`

**Implementation details**:
- Sparse CSR representation per band (~20-40 non-zero bins instead of 513+)
- Slaney normalisation (unit-area triangles)
- `log10` via `LOG10_2 * log2f()` for speed
- Floor: `1e-10` prevents log(0)

### Peak Picker (`peaks.rs`)

```rust
pub struct Peak { pub t_frame: u32, pub f_bin: u16, pub _pad: u16, pub mag: f32 }  // 12 bytes, Pod
pub struct PeakPickerConfig { pub neighborhood_t, pub neighborhood_f, pub min_magnitude, pub target_per_sec }
pub struct PeakPicker { /* pooled scratch buffers */ }
pub struct IncrementalPeakDetector { /* per-column Lemire deques, ring buffer */ }
```

**Algorithm**: Separable 2-D rolling max (Lemire monotonic deque, O(N×M) total regardless of neighbourhood) → local-max check → per-second adaptive thresholding (top-K by magnitude)

**Streaming variant** (`IncrementalPeakDetector`): maintains per-column vertical deques + horizontal ring → zero-alloc after construction, bit-exact with offline

### Resampler (`resample.rs`)

```rust
pub struct SincQuality { pub half_taps: usize, pub kaiser_beta: f32, pub polyphase_steps: u16 }
pub struct SincResampler { /* polyphase kernel table, precomputed at construction */ }
pub fn linear(input, from_sr, to_sr) -> Vec<f32>  // Simple linear interpolation
```

**Windowed-sinc algorithm**:
- Kaiser window (β=8.6 → -80dB stopband), 32 taps default
- Polyphase table: 256 steps × 65 coefficients, precomputed
- Three-region processing: left-boundary / safe-middle (no bounds check) / right-boundary
- Anti-alias cutoff: `min(from_sr, to_sr) / (2 * from_sr)`
- Modified Bessel I₀ via series expansion (≤30 terms)

### Windows (`windows.rs`)

```rust
pub enum WindowKind { Hann, Hamming, Blackman }
pub fn make_window(kind, n) -> Vec<f32>  // Periodic form (N, not N-1)
```

## I/O Module (`src/io/`, feature = "std")

```rust
pub struct DecodeLimits { pub max_bytes: u64, pub max_samples: Option<usize> }
pub fn decode_to_mono(path) -> Result<(Vec<f32>, u32)>
pub fn decode_to_mono_limited(path, limits) -> Result<(Vec<f32>, u32)>
pub fn decode_to_mono_at(path, target_sr) -> Result<Vec<f32>>          // + resample
pub fn decode_to_mono_at_limited(path, target_sr, limits) -> Result<Vec<f32>>
```

**Supported codecs**: MP3, FLAC, WAV, OGG-Vorbis, AAC-in-MP4, PCM, ADPCM, ALAC, MKV, AIFF
**Pipeline**: Symphonia probe → format reader → decoder → multi-channel downmix (average) → optional SincResampler
**Error handling**: Recoverable per-packet IO/Decode errors silently skipped

## Neural Module (`src/neural/`, feature = "neural")

### NeuralEmbedder (offline)

```rust
pub struct NeuralEmbedderConfig {
    pub model_path: String,     // ONNX file path
    pub sample_rate: u32,       // default 16000
    pub n_fft: usize,           // default 1024
    pub hop: usize,             // default 320 (20ms at 16kHz)
    pub n_mels: usize,          // default 128
    pub fmin/fmax: f32,         // default 0.0 / sr/2
    pub mel_scale: MelScale,    // default Slaney
    pub window_secs: f32,       // default 1.0
    pub hop_secs: f32,          // default 1.0
    pub l2_normalize: bool,     // default true
}
pub struct NeuralEmbedding { pub vector: Vec<f32>, pub t_start: TimestampMs }
pub struct NeuralFingerprint { pub embeddings: Vec<NeuralEmbedding>, pub embedding_dim, pub frames_per_sec }
```

**Model contract**: Input `[1, n_mels, n_frames] f32` → Output `[1, embedding_dim]` (or any shape flattening to non-empty f32)

**Construction**: Load ONNX → type+optimize+make-runnable (once) → probe inference for embedding_dim
**Extract pipeline**: Sliding window → LogMelFrontend → unsafe uninit tensor → strided write → `runnable.run()` → optional L2 norm

### StreamingNeuralEmbedder

```rust
pub struct StreamingNeuralEmbedder { /* EmbedderCore + carry buffer + embedding_scratch */ }
```

- `try_push(samples)` / `try_push_with(samples, callback)` — fallible, zero-alloc hot path
- `push(samples)` — panics on inference error (StreamingFingerprinter trait)
- `flush()` → empty (partial windows dropped)
- Carry buffer < window_samples always
- Bit-exact with offline extract

### LogMelFrontend (`frontend.rs`, pub(crate))
- Combines `ShortTimeFFT` + `MelFilterBank`
- `for_each_frame(window, callback)` — strided STFT→power→log_mel per frame
- Zero per-call allocation

## Watermark Module (`src/watermark/`, feature = "watermark")

```rust
pub struct WatermarkConfig {
    pub model_path: String,
    pub message_bits: u8,       // ≤32, default 16
    pub threshold: f32,         // [0,1], default 0.5
    pub sample_rate: u32,       // default 16000
}
pub struct WatermarkResult {
    pub detected: bool,         // confidence > threshold
    pub confidence: f32,        // mean detection score
    pub message: u32,           // decoded bits LSB-first
    pub localization: Vec<f32>, // raw per-sample scores
}
pub struct WatermarkDetector { /* InferenceModel + lazy cached (length, Runnable) */ }
```

**Model contract**: Input `[1, 1, T] f32` waveform → Output[0]: detection scores, Output[1]: message logits

**Key pattern**: Typed plan lazily built and **cached per input length** (model.clone() + retype on length change)

## Testing Strategy

### Test Files (356+ tests)

| File | Strategy | What it verifies |
|------|----------|-----------------|
| `codec_roundtrip.rs` | Integration | Same-song hash survival: Wang≥0.25, Panako≥0.20, Haitsma≥0.80 Jaccard |
| `codec_extended.rs` | Integration | AIFF, two-track discrimination (<0.05), mono/stereo equiv, SR ladder |
| `real_audio_e2e.rs` | End-to-end | Segment matching, gain invariance, determinism (10×), decoder edge cases |
| `real_audio_golden.rs` | Golden-file | Bit-exact regression on real audio (.bin files) |
| `regression.rs` | Golden-file | Bit-exact synthetic regression with magic headers (AFPWANG\0 etc.) |
| `property.rs` | Proptest | Streaming≡offline (random chunks), NaN safety, tempo robustness |
| `robustness.rs` | Integration | Synthetic noise/lowpass survival |
| `real_audio.rs` | Integration | Real audio noise/lowpass, streaming==offline, resampler cross-check |
| `watermark.rs` | Config validation | Error variants (no model shipped in repo) |
| `neural.rs` | Config validation | Error variants, trait object safety |

### Key Test Audio
- **piano.ogg, speech.ogg**: CC0, generated for this project
- **galway.*** (6 codecs + stereo): Kevin MacLeod CC-BY 3.0, 16s
- **freak.*** (5 codecs + SR ladder): Kevin MacLeod CC-BY 3.0, 16s

### Fuzz Targets (7)
- `wang_hash_roundtrip`, `panako_hash_roundtrip`, `haitsma_hash_roundtrip` — encode/decode invariant
- `streaming_wang_equiv`, `streaming_panako_equiv`, `streaming_haitsma_equiv` — streaming≡offline
- `sinc_resampler` — resampler robustness

## CI Pipeline (11 jobs)

1. `fmt` — rustfmt check
2. `clippy` — all-features + no-default-features
3. `test` — unit/integration (skip golden), doc tests, doc build `-D warnings`
4. `golden` — regression golden tests only
5. `audit` — cargo-audit for CVEs
6. `property-tests` — proptest with 256 cases
7. `msrv` — check + test at Rust 1.93.0
8. `no-std` — cargo check --no-default-features
9. `no-std-test` — cargo test --no-default-features
10. `deny` — license/ban/source compliance
11. `fuzz` — smoke 7 targets (30s each, nightly, sanitizer=none)

## Key Design Patterns & Invariants

### Streaming/Offline Bit-Exact Parity
The strongest invariant: `StreamingWang::push()` produces the exact same hash multiset as `Wang::extract()` for any chunk partitioning (down to 1-sample-per-push). Verified by proptest, fuzz, and integration tests.

### Zero-Allocation Hot Path
All streaming `push()` calls reuse pre-allocated scratch after warmup. No `Vec::push`, no `Box::new`, no allocation in the inner loop. Buffers sized at construction.

### Power-Domain STFT
All classical algos compute `|X|²` directly (skip sqrt). Downstream `10·log10(p)` = `DB_LOG2_FACTOR·log2(p)` absorbs the missing root. Single `log2` instruction on x86.

### Sanitized Input
- Offline: `reject_non_finite()` → error
- Streaming: `extend_sanitized()` → replace NaN/Inf with 0.0 (infallible API)

### Pod Hash Types
`WangHash` and `PanakoHash` are `#[repr(C)]` + `bytemuck::Pod` — zero-copy serializable to mmap'd files or C ABI.

### Dual Constructors
Every configurable struct has `new()` (panics) + `try_new()` (returns Result).

### Sorted Vec over BTreeMap
Bucket tracking uses binary_search on sorted Vec (≤3 entries) — lower overhead than tree structures for tiny collections.

### Linear-Insert Top-K (Wang)
`partition_point + insert` beats BinaryHeap for K≤16 due to lower constant factor and cache locality.

### Suffix-Max Early-Exit (Panako)
Pre-computed suffix array of maximum magnitudes allows pruning (b,c) pair enumeration when no remaining c can beat the current K-th best score.

### Lazy Plan Caching (Watermark)
ONNX model `clone()` + retype on input-length change; same-length calls reuse the compiled plan.

## Build Commands

```bash
# Full test suite
cargo test --all-features

# No-std build
cargo build --no-default-features

# Benchmarks
cargo bench --bench extract
cargo bench --bench streaming
cargo bench --bench neural_frontend --features neural

# Clippy
cargo clippy --all-targets --all-features -- -D warnings

# Fuzz (requires nightly)
cargo +nightly fuzz run streaming_wang_equiv -- -max_total_time=30

# Generate docs
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# Update goldens (after intentional algorithm changes)
UPDATE_GOLDENS=1 cargo test --test regression --all-features
cargo test --test generate_real_goldens --all-features -- --ignored
```

## File Size Reference

| Module | Size | Notes |
|--------|------|-------|
| `wang.rs` | 67 KB | Largest — offline + streaming + helpers |
| `panako.rs` | 63 KB | Similar structure to Wang |
| `haitsma.rs` | 34 KB | Simpler (no peak detection) |
| `peaks.rs` | 34 KB | Lemire deque + incremental detector |
| `stft.rs` | 28 KB | SIMD paths inflate size |
| `embedder.rs` | 27 KB | ONNX interaction + config validation |
| `resample.rs` | 21 KB | Polyphase table construction |
| `streaming.rs` | 21 KB | Neural streaming carry logic |
| `decoder.rs` | 20 KB | Symphonia wrapper |
| `mel.rs` | 17 KB | HTK/Slaney + sparse construction |
| `detector.rs` | 15 KB | Watermark lazy plan caching |
| `fp.rs` | 14 KB | Trait definitions + batch parallel |
| `error.rs` | 10 KB | AfpError + IoError |
