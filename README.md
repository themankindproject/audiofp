# audiofp

[![Crates.io](https://img.shields.io/crates/v/audiofp)](https://crates.io/crates/audiofp)
[![Documentation](https://docs.rs/audiofp/badge.svg)](https://docs.rs/audiofp)
[![License](https://img.shields.io/crates/l/audiofp)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/audiofp/ci.yml?branch=main&label=CI)](https://github.com/themankindproject/audiofp/actions/workflows/ci.yml)
![Crates.io Downloads](https://img.shields.io/crates/d/audiofp)
![Rust Version](https://img.shields.io/badge/rust-1.93%2B-blue)

Audio fingerprinting library for Rust with **classical landmark and band-power algorithms**, **streaming extraction**, **file decoding**, and **AudioSeal-compatible watermark detection**.

## Overview

`audiofp` provides three complementary classical fingerprinters for music identification, each with offline and streaming variants:

| Method | Use Case | Sample Rate | Frame Rate | Output Size |
|--------|----------|-------------|------------|-------------|
| **Wang** | Music ID, Shazam-style matching | 8 kHz | 62.5 fps | ~2.4 KB/s (fan-out 10) |
| **Panako** | Music ID with ±5 % tempo robustness | 8 kHz | 62.5 fps | ~2.0 KB/s (fan-out 5) |
| **Haitsma** | Compact dense IDs, fastest extraction | 5 kHz | 78.125 fps | 312 B/s |
| **Streaming** | Real-time hash emission | (per algorithm) | (per algorithm) | Bit-exact offline parity |
| **Watermark** | AudioSeal detection (BYO ONNX) | 16 kHz | (per model) | Detection + 16-bit message |

Perfect for:
- Music identification ("what is this song?")
- Audio deduplication at scale
- Royalty / rights enforcement against re-encoded content
- Embedding-based similarity search and cover/remix detection (BYO ONNX model via the `neural` feature)
- Watermark verification on generative-AI audio

## Features

- **Three Classical Algorithms** - Wang (landmark pairs) + Panako (triplet hashes with tempo β) + Haitsma–Kalker (32-bit/frame band sign)
- **Truly Incremental Streaming** - Per-push CPU proportional to new samples, not total stream length. Rolling spectrogram + per-bucket finalisation + per-anchor target accumulator. Bit-exact parity with offline `extract` (verified by the test suite at every chunk size).
- **Bit-Exact Determinism** - Same input always produces the same hashes; verified down to 1-sample-per-push streaming chunks
- **`bytemuck::Pod` Hash Types** - Persist hashes directly to mmap'd files or ship over a C ABI without serialization
- **Audio File Decoding** - MP3, FLAC, WAV, OGG-Vorbis, AAC-in-MP4, raw PCM via Symphonia
- **High-Quality Resampling** - Built-in windowed-sinc Kaiser resampler with auto anti-aliasing cutoff
- **Watermark Detection** - AudioSeal-compatible ONNX wrapper (Tract backend); typed model is cached per input length and rebuilt automatically when the length changes
- **Neural Embedder** - Generic ONNX log-mel embedder with offline + streaming modes; build-once-runnable, zero-alloc `try_push_with` callback (scratch is allocated at construction, reused on every push)
- **DSP Primitives Reusable** - Public `dsp::stft`, `dsp::mel`, `dsp::peaks`, `dsp::resample`, `dsp::windows`
- **Allocation-Free Hot Path** - Streaming `push` reuses pre-allocated scratch after warmup
- **`no_std + alloc` Capable** - DSP and classical fingerprinters compile without std (host-only today; bare-metal in roadmap)
- **Feature-Gated Heavy Deps** - Symphonia and Tract both opt-in via Cargo features
- **Optional `mimalloc`** - Single-flag opt-in to install `mimalloc` as the global allocator

## Installation

```toml
[dependencies]
audiofp = "0.3"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Enables `audiofp::io` (Symphonia file decoder) |
| `watermark` | No | Enables `audiofp::watermark` via Tract ONNX runtime |
| `neural` | No | Enables `audiofp::neural`: generic ONNX log-mel embedder via Tract (BYO model) |
| `mimalloc` | No | Installs `mimalloc::MiMalloc` as the process-wide `#[global_allocator]` |

Minimal build (no_std + alloc, DSP and classical only):
```toml
[dependencies]
audiofp = { version = "0.3", default-features = false }
```

With watermark detection (pulls in Tract):
```toml
[dependencies]
audiofp = { version = "0.3", features = ["watermark"] }
```

With the neural embedder (pulls in Tract):
```toml
[dependencies]
audiofp = { version = "0.3", features = ["neural"] }
```

With mimalloc for a faster global allocator:
```toml
[dependencies]
audiofp = { version = "0.3", features = ["mimalloc"] }
```

## Quick Start

### Zero-deps (no audio file)

Paste this into a fresh `cargo new` binary after `cargo add audiofp` — no MP3 required:

```rust
use audiofp::classical::Wang;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn main() {
    // 3 s of silence at Wang's 8 kHz. Silence yields 0 hashes, but the SDK path runs.
    let samples = vec![0.0_f32; 8_000 * 3];
    let mut wang = Wang::default();
    let fp = wang
        .extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
        .unwrap();
    println!("{} hashes (silence → usually 0)", fp.hashes.len());
}
```

### Fingerprint a file

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{AudioBuffer, Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode any supported file format and resample to Wang's 8 kHz.
    // Needs ≥ ~2 s of audio or extract returns AudioTooShort.
    let samples = decode_to_mono_at("song.mp3", 8_000)?;

    let mut wang = Wang::default();
    let buf = AudioBuffer::new(&samples, SampleRate::HZ_8000);
    let fp = wang.extract(buf)?;

    println!("{} hashes at {:.1} fps", fp.hashes.len(), fp.frames_per_sec);
    for h in fp.hashes.iter().take(5) {
        println!("  t_anchor={} hash={:08x}", h.t_anchor, h.hash);
    }

    Ok(())
}
```

### Streaming Mode

```rust
use audiofp::classical::StreamingWang;
use audiofp::StreamingFingerprinter;

fn main() {
    let mut s = StreamingWang::default();

    // Synthetic 8 kHz mono chunks (16 ms ≈ 128 samples). Swap for mic/file chunks.
    let chunk = vec![0.0_f32; 128];
    for _ in 0..100 {
        for (timestamp, hash) in s.push(&chunk) {
            println!("{:?} {:08x}", timestamp, hash.hash);
        }
    }

    // Drain whatever's pending at end-of-stream.
    for (timestamp, hash) in s.flush() {
        println!("{:?} {:08x}", timestamp, hash.hash);
    }

    println!("latency: {} ms", s.latency_ms());
}
```

## Documentation

For complete API reference and usage examples, see [USAGE.md](USAGE.md).

## Architecture

### Fingerprint Types

Each algorithm emits a strongly-typed, `bytemuck::Pod`-castable result:

```
Wang offline                         Panako offline
┌──────────────────────────┐         ┌──────────────────────────┐
│ WangFingerprint          │         │ PanakoFingerprint        │
│   hashes: Vec<WangHash>  │         │   hashes: Vec<PanakoHash>│
│   frames_per_sec: f32    │         │   frames_per_sec: f32    │
└──────────────────────────┘         └──────────────────────────┘

WangHash (8 bytes, repr(C))          PanakoHash (16 bytes, repr(C))
├── hash: u32                        ├── hash: u32
└── t_anchor: u32                    ├── t_anchor: u32
                                     ├── t_b: u32
                                     └── t_c: u32

Haitsma offline
┌──────────────────────────┐
│ HaitsmaFingerprint       │
│   frames: Vec<u32>       │   one u32 per spectrogram frame ≥ 1
│   frames_per_sec: f32    │
└──────────────────────────┘
```

### Algorithm Pipeline

1. **Decode** — Parse any supported format (MP3, FLAC, WAV, OGG-Vorbis, AAC-in-MP4, PCM) via Symphonia and downmix to mono `f32`
2. **Resample** — Built-in windowed-sinc Kaiser resampler (default 32 taps, β=8.6) brings the audio to the algorithm's required rate
3. **STFT** — `realfft`-backed real-input transform with reusable scratch; Hann window, configurable hop and `n_fft`
4. **Algorithm-specific extraction**:
   - **Wang**: dB log-mag → 31×31 peak picker (capped at 30/s) → anchor-target landmark pairs in `Δt ∈ [1, 63], |Δf| ≤ 64`
   - **Panako**: same front-end → triplet enumeration in cone `Δt < 96, |Δf| < 96` → tempo-invariant β packing
   - **Haitsma**: 33 log-spaced bands (300–2000 Hz) → 32 sign bits per frame from band-difference deltas
5. **Streaming variants** maintain a rolling `2·neighborhood_t + 1`-row spectrogram window and detect peaks frame-by-frame as each ripens; finalise per-second adaptive thresholding bucket-by-bucket; grow per-anchor target heaps incrementally; emit hashes when each anchor's target zone is fully observed. Bit-exact equivalence with offline `extract` is guaranteed under arbitrary chunking — including the 1-sample-per-push pathological case.

### Hash Layouts

```text
WangHash::hash (32 bits)
[31..23]  f_a_q  9 bits, anchor frequency (quantised to 512 buckets)
[22..14]  f_b_q  9 bits, target frequency (same quantisation)
[13.. 0]  Δt    14 bits, frames between anchor and target

PanakoHash::hash (32 bits)
[31..30]  sign       2 bits, signs of Δf_ab and Δf_bc
[29..28]  mag_order  2 bits, which of {a, b, c} has the largest magnitude
[27..23]  β          5 bits, round((t_c - t_b) / (t_c - t_a) · 31)
[22..15]  Δf_ab      8 bits signed, clamped to ±127
[14.. 7]  Δf_bc      8 bits signed, clamped to ±127
[ 6.. 0]  reserved   7 bits, zero

Haitsma frame (32 bits, "MSB-zero" packing)
bit 31 → band 0,  bit 0 → band 31
F[n][b] = ((E[n][b] − E[n][b+1]) − (E[n−1][b] − E[n−1][b+1])) > 0
```

## Performance

> Performance numbers from **v0.2.0** (offline extract) and **v0.3.4**
> (streaming), measured on an Intel i5-1135G7 (4 cores, 8 threads,
> 2.40 GHz base). Your numbers will vary by CPU, features, and input.
> Reproduce locally with `cargo bench --bench extract` and
> `cargo bench --bench streaming`.
>
> Since **v0.3.4** the classical streaming `push` path is ~16× faster
> (~94 % less wall time on Wang/Panako small-chunk benches) — see
> [CHANGELOG](CHANGELOG.md#034---2026-06-14).

Offline extract (`cargo bench --bench extract`, 30 s of synthetic audio):

| Algorithm  | 30 s of audio | Realtime factor |
| ---------- | ------------- | --------------- |
| `Wang`     |  99 ms        | 303×            |
| `Panako`   | 104 ms        | 288×            |
| `Haitsma`  |  47 ms        | 638×            |

Hot-path design notes:

- All three classical fingerprinters share the same Hann-windowed STFT and Lemire monotonic-deque peak picker (amortised O(N · M)), so cost is dominated by the FFT.
- Streaming `push` reuses pre-allocated scratch; no allocation per frame after the initial ring is sized.
- `SincResampler` with the default 32-tap Kaiser kernel is O(N · 2 · half_taps) per output sample with a precomputed Bessel I₀(β).

| Streaming type      | `latency_ms()` | Notes                                                  |
| ------------------- | -------------- | ------------------------------------------------------ |
| `StreamingWang`     | 2 256 ms       | Includes 1 s for per-second adaptive peak thresholding |
| `StreamingPanako`   | 2 784 ms       | Wider target zone (96 frames vs Wang's 63)             |
| `StreamingHaitsma`  | 409 ms         | No peak picker → bounded by `n_fft / sr`               |

Run benchmarks for your own host:
```bash
cargo bench --bench extract
cargo bench --bench streaming
cargo bench --bench extract -- --save-baseline main   # save for diffing later
```

### Memory Safety

- Sample-rate-strict APIs reject mismatched inputs with `AfpError::UnsupportedSampleRate`
- Audio length checks reject buffers shorter than each algorithm's minimum (≥ 2 s)
- Allocation-free streaming hot path after warmup (no `Vec::push` in the inner loop)
- `bytemuck::Pod` derive on hash types is sound: every field is `repr(C)` with explicit padding

### Determinism

- **Identical inputs → identical outputs** — same audio, same fingerprinter, same config produces bit-for-bit identical hashes on every call and every supported target
- **Stable algorithm IDs** — `Fingerprinter::name()` returns versioned strings (`"wang-v1"`, `"panako-v2"`, `"haitsma-v1"`); a future major bump that changes hash bytes will change the version suffix
- **Stable hash layouts** — bit positions in `WangHash::hash`, `PanakoHash::hash`, and Haitsma frames are stable across patch and minor versions inside `0.x`
- **Verified streaming/offline parity** — the test suite feeds randomised chunk sequences (down to 1 sample per push) through the streaming impl and asserts the output hash multiset matches `extract`

## Robustness

- **Codec-tolerant by design** — Wang and Panako are spectral-peak based; Haitsma is band-power-difference based. All three survive lossy re-encoding, verified by the test suite on real music:

  | Codec | Wang (Jaccard) | Panako (Jaccard) | Haitsma (bit-sim) |
  |-------|---------------|-----------------|-------------------|
  | WAV/FLAC (lossless) | 1.000 | — | 1.000 |
  | MP3 128 kbps | 0.40 | 0.45 | 0.93 |
  | OGG-Vorbis | 0.36 | 0.42 | 0.91 |
  | AAC (M4A) | 0.50 | 0.54 | 0.77 |
  | AIFF (lossless) | 1.000 | — | — |
  | **Cross-track (different song)** | **0.001** | — | — |

  > Test audio: "Galway" and "Furious Freak" by Kevin MacLeod, 16 s each, 6 codec variants.
  > Thresholds: Wang ≥ 0.25, Panako ≥ 0.20, Haitsma ≥ 0.75.
  > In practice, 5–10 matching hashes suffice for confident identification.

- **Two-track discrimination verified** — different songs produce <0.1% hash overlap (random collision floor), while the same song across codecs produces 25–80% overlap. The "is this the same recording?" question is decidable.
- **Sample-rate agnostic** — the test suite verifies fingerprinting from source rates of 8 kHz to 44.1 kHz (resampled to the algorithm's native rate). Even telephone-quality (8 kHz) audio produces 1700+ usable hashes.
- **Mono only** — multi-channel inputs must be downmixed by the caller (the file decoder does this for you). Stereo → mono downmix preserves 99.5% of hashes (verified).
- **Sample-rate-strict** — each fingerprinter requires its native rate (8 kHz / 5 kHz). Resample with `dsp::resample::SincResampler` or `decode_to_mono_at` if your source differs.
- **Resilient decoder** — recoverable per-packet failures inside Symphonia are silently skipped so a single corrupt block doesn't kill a whole-file decode.
- **356 tests** including 44 real-audio E2E tests across 6 codecs, 2 tracks, stereo/mono, and a sample-rate ladder from 8 kHz to 44.1 kHz.

## Comparison with Alternatives

| Feature | audiofp | chromaprint-rust | dejavu (Python) |
|---------|-----|------------------|-----------------|
| Pure Rust | Yes | No (FFI to C lib) | No |
| Wang landmarks | Yes | No | Yes |
| Panako triplets (tempo-robust) | Yes | No | No |
| Haitsma–Kalker | Yes | No | No |
| Streaming variants | Yes | Limited | No |
| Bit-exact streaming/offline parity | Yes | No | N/A |
| File decoding included | Yes (Symphonia) | Yes (limited) | Yes (FFmpeg) |
| Watermark detection | Yes (AudioSeal) | No | No |
| `no_std + alloc` capable | Yes (host) | No | N/A |
| `bytemuck::Pod` hash types | Yes | No | N/A |
| Built-in resampler | Yes | No | No |

## Examples

The `examples/` directory contains complete working programs that can be run with `cargo run --example <name>`:

- `enroll_file` — fingerprint a single audio file and print the unique Wang landmark count (`--features` default / `std`).
- `match_two_files` — print the number of Wang hash collisions between two files (the canonical "is this the same recording?" check).
- `compare_algorithms` — run Wang, Panako, and Haitsma–Kalker over the same file and report per-algorithm timing and hash counts.
- `stream_buffer` — feed Wang's streaming fingerprinter from an `io::Read` chunk-by-chunk.
- `dsp_starter` — STFT → mel → peaks pipeline on synthetic audio (no file, no optional features).
- `neural_embed` — load a BYO ONNX embedder and print embedding dim (`--features neural`).
- `watermark_detect` — load an AudioSeal-compatible ONNX model and print confidence (`--features watermark`).

```bash
cargo run --example dsp_starter
cargo run --example neural_embed --features neural -- path/to/model.onnx
cargo run --example watermark_detect --features watermark -- path/to/audioseal.onnx [audio.wav]
```

The doctests across the public API and [USAGE.md](USAGE.md) cover the full surface for users wiring `audiofp` into their own binary.

## Security

See [SECURITY.md](SECURITY.md) for the threat model (audio / PCM / ONNX / hash outputs) and how to report vulnerabilities privately. Fingerprints are **perceptual**, not cryptographic MACs — use `DecodeLimits` with `decode_to_mono_limited` for untrusted uploads.

## Contributing

Please read the [Code of Conduct](CODE_OF_CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md).
Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run tests: `cargo test --all-features`
4. Run clippy: `cargo clippy --all-targets --all-features -- -D warnings`
5. Run formatter: `cargo fmt --all -- --check`
6. Commit your changes
7. Push the branch and open a Pull Request

### Development Setup

```bash
# Clone
git clone https://github.com/themankindproject/audiofp
cd audiofp

# Run all tests
cargo test --all-features

# Run no_std build path
cargo build --no-default-features

# Generate documentation
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps --open
```

CI (`.github/workflows/ci.yml`) runs `fmt`, `clippy`, and `test` jobs in parallel on every push and PR.

## License

MIT License — see [LICENSE](LICENSE) for details.

## References

- Avery Wang, *An Industrial-Strength Audio Search Algorithm* (ISMIR 2003) — Wang landmarks
- Joren Six & Marc Leman, *Panako: A Scalable Acoustic Fingerprinting System* (ISMIR 2014); 2021 update — triplet β hash
- Jaap Haitsma & Ton Kalker, *A Highly Robust Audio Fingerprinting System* (ISMIR 2002) — band-power sign bits
- San Roman, R., Fernandez, P., Elsahar, H., Défossez, A., Furon, T. & Tran, T. *Proactive Detection of Voice Cloning with Localized Watermarking.* arXiv:2401.17264, 2024 (AudioSeal) — watermark model. <https://arxiv.org/abs/2401.17264>
