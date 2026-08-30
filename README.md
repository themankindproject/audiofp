# audiofp

[![Crates.io](https://img.shields.io/crates/v/audiofp)](https://crates.io/crates/audiofp)
[![Documentation](https://docs.rs/audiofp/badge.svg)](https://docs.rs/audiofp)
[![License](https://img.shields.io/crates/l/audiofp)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/audiofp/ci.yml?branch=main&label=CI)](https://github.com/themankindproject/audiofp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/themankindproject/audiofp/branch/main/graph/badge.svg)](https://codecov.io/gh/themankindproject/audiofp)
![Crates.io Downloads](https://img.shields.io/crates/d/audiofp)
![Rust Version](https://img.shields.io/badge/rust-1.93%2B-blue)

Audio fingerprinting library for Rust with **classical landmark and band-power algorithms**, **in-memory matching**, **streaming extraction**, **file decoding**, and **AudioSeal-compatible watermark detection**.

## Overview

`audiofp` provides three complementary classical fingerprinters for music identification, each with offline and streaming variants, plus an in-memory matching layer for identification:

| Method | Use Case | Sample Rate | Frame Rate | Output Size |
|--------|----------|-------------|------------|-------------|
| **Wang** | Music ID, Shazam-style matching | 8 kHz | 62.5 fps | ~2.4 KB/s (fan-out 10) |
| **Panako** | Music ID with ±5 % tempo robustness | 8 kHz | 62.5 fps | ~2.0 KB/s (fan-out 5) |
| **Haitsma** | Compact dense IDs, fastest extraction | 5 kHz | 78.125 fps | 312 B/s |
| **Matching** | In-memory ID (`WangMatcher`, `HaitsmaMatcher`, …) | — | — | — |
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
- **In-Memory Matching** - `WangMatcher` / `HaitsmaMatcher` / `PanakoMatcher` (tempo-invariant 2-D Hough + RANSAC) / `NeuralMatcher` plus `match_best` / `match_ranked` and transient `WangIndex` / `HaitsmaIndex` / `PanakoIndex` accelerators for 1:N identification. No persistence or DB adapters.
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
# WAV + MP3 decoding for the quick-start below (pick the codecs you need):
audiofp = { version = "0.4", features = ["std-wav", "std-mp3"] }
```

The default build is `no_std + alloc` with **no codecs**. Decoding helpers
(`audiofp::io`) are opt-in per codec: `std-wav`, `std-mp3`, `std-flac`,
`std-ogg`, `std-aac`, `std-mp4`, plus `std-aiff` / `std-mkv` / `std-adpcm` /
`std-alac` for the extended formats — or `all-codecs` for every codec at
once (the pre-0.4.0 `std` behavior).

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std-wav` | No | WAV + raw PCM decoding via Symphonia (`audiofp::io`) |
| `std-mp3` | No | MP3 decoding via Symphonia |
| `std-flac` | No | FLAC decoding via Symphonia |
| `std-ogg` | No | Ogg-Vorbis decoding via Symphonia |
| `std-aac` | No | AAC decoding via Symphonia |
| `std-mp4` | No | AAC-in-MP4 / ISO-BMFF decoding via Symphonia |
| `std-aiff` / `std-mkv` / `std-adpcm` / `std-alac` | No | Extended codecs |
| `all-codecs` | No | Every codec at once — the pre-0.4.0 `std` behavior |
| `rayon` | No | Parallel batch fingerprinting via `fingerprint_batch_parallel` (implies `std`) |
| `watermark` | No | Enables `audiofp::watermark` via Tract ONNX runtime (implies `std`) |
| `neural` | No | Enables `audiofp::neural`: generic ONNX log-mel embedder via Tract (BYO model; implies `std`) |
| `mimalloc` | No | Installs `mimalloc::MiMalloc` as the process-wide `#[global_allocator]` (implies `std`) |

Minimal build (no_std + alloc, DSP and classical only):
```toml
[dependencies]
audiofp = { version = "0.4", default-features = false }
```

## Quick Start

### Fingerprint a file

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Decode any supported file format and resample to Wang's 8 kHz.
    // Needs ≥ ~2 s of audio or extract returns AudioTooShort.
    let samples = decode_to_mono_at("song.mp3", 8_000)?;

    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000)?;

    println!("{} hashes at {:.1} fps", fp.hashes.len(), fp.frames_per_sec);
    for h in fp.hashes.iter().take(5) {
        println!("  t_anchor={.0} hash={:08x}", h.t_anchor, h.hash);
    }

    Ok(())
}
```

### Match two fingerprints (Wang)

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = decode_to_mono_at("clip.wav", 8_000)?;
    let query = Wang::default().extract(&samples, SampleRate::HZ_8000)?;
    let reference = query.clone(); // same recording

    let m = WangMatcher::new(WangMatchConfig::default()).match_one(&query, &reference);
    println!("is_match={} score={:.3} offset={} ms", m.is_match, m.score, m.offset.ms);
    Ok(())
}
```

### Streaming Mode

```rust
use audiofp::StreamingFingerprinter;
use audiofp::classical::StreamingWang;
use std::f32::consts::PI;

fn main() {
    let mut s = StreamingWang::default();

    // 8 kHz mono, fed in 200 ms chunks — swap for mic/file chunks.
    let sr = s.required_sample_rate(); // 8_000
    let chunk_len = (sr / 5) as usize;
    let mut total = 0usize;
    for i in 0..25 {
        // 5 s of a two-tone signal (silence emits nothing).
        let chunk: Vec<f32> = (0..chunk_len)
            .map(|j| {
                let t = (i * chunk_len + j) as f32 / sr as f32;
                0.5 * (2.0 * PI * 880.0 * t).sin() + 0.3 * (2.0 * PI * 1320.0 * t).sin()
            })
            .collect();

        // push returns hashes that finalised during this chunk.
        for (ts, hash) in s.push(&chunk).unwrap() {
            println!("t={} ms hash={:08x}", ts.0, hash.hash);
            total += 1;
        }
    }

    // Drain whatever is pending at end-of-stream.
    for (ts, hash) in s.flush().unwrap() {
        println!("t={} ms hash={:08x}", ts.0, hash.hash);
        total += 1;
    }

    println!("{total} hashes; latency {} ms", s.latency_ms());
}
```

## Documentation

For complete API reference and usage examples, see [USAGE.md](USAGE.md).

## Performance

Offline extract (`cargo bench --bench extract`, 30 s of synthetic audio):

| Algorithm  | 30 s of audio | Realtime factor |
| ---------- | ------------- | --------------- |
| `Wang`     |  79 ms        | 380×            |
| `Panako`   |  81 ms        | 370×            |
| `Haitsma`  |  42 ms        | 714×            |

Streaming push (`cargo bench --bench streaming`, 10 s of synthetic audio):

| Streaming type      | Small chunks (256 samples) | Large chunks (1 s) | `latency_ms()` |
| ------------------- | -------------------------: | ------------------:| --------------- |
| `StreamingWang`     | 10.5 ms                    | 10.6 ms            | 2 256 ms        |
| `StreamingPanako`   | 11.6 ms                    | 11.4 ms            | 2 784 ms        |
| `StreamingHaitsma`  |  6.3 ms                    |  6.7 ms            | 409 ms          |

Neural front-end (`cargo bench --features neural --bench neural_frontend`):

| Path                          | Time       |
|-------------------------------|:----------:|
| `log_mel_pipeline_1s_window`  | 297 µs     |
| `strided_tensor_write`        | 7.6 µs     |
| `l2_normalize_1024d`          | 2.5 µs     |

Matching (`cargo bench --bench matching`, 5 s synthetic fingerprints):

| Path                           | Time       | Notes                                      |
|--------------------------------|:----------:|--------------------------------------------|
| `WangMatcher` 1:1 self-match  | ~111 µs    | Offset-histogram voting + prominence       |
| `HaitsmaMatcher` 1:1 exact    | ~18 µs     | Exhaustive BER at best alignment           |
| `PanakoMatcher` 1:1           | ~264 µs    | 2-D Hough + RANSAC line-fitting            |
| `WangIndex` N=100 query       | ~102 µs    | Inverted index + sliding-window peak       |

Latency budget (per query, default configs, Intel i5-1135G7):

| Catalog size | `WangIndex` query | Throughput  |
|:------------:|:-----------------:|:-----------:|
| 100 tracks   | ~102 µs           | ~9 800 q/s  |
| 1 000 tracks | ~1 ms (est.)      | ~1 000 q/s  |
| 10 000 tracks| ~10 ms (est.)     | ~100 q/s    |

> Index query scales approximately linearly with catalog size (one
> candidate-scoring pass per reference with hash hits). For catalogs
> above ~10 000 tracks, use `min_votes` / `min_score` pre-filters or
> shard the index.

Run benchmarks for your own host:
```bash
cargo bench --bench extract
cargo bench --bench streaming
cargo bench --bench extract -- --save-baseline main   # save for diffing later
```

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

- **Two-track discrimination verified** — different songs produce <0.1% hash overlap (random collision floor), while the same song across codecs produces 25–80% overlap.
- **606 tests** including adversarial stress tests, real-audio E2E across 6 codecs, and property-based streaming/offline parity checks. See [ROBUSTNESS.md](ROBUSTNESS.md) for full methodology.

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
| In-memory matcher (Wang/Haitsma) | Yes | No | Yes (Dejavu) |

## Security

See [SECURITY.md](SECURITY.md) for the threat model (audio / PCM / ONNX / hash outputs) and how to report vulnerabilities privately. Fingerprints are **perceptual**, not cryptographic MACs — use `DecodeLimits` with `decode_to_mono_limited` for untrusted uploads.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Quick start:

```bash
git clone https://github.com/themankindproject/audiofp && cd audiofp
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

CI runs `fmt`, `clippy`, and `test` on ubuntu/macOS/Windows on every push and PR.

## License

MIT License — see [LICENSE](LICENSE) for details.

## References

- Avery Wang, *An Industrial-Strength Audio Search Algorithm* (ISMIR 2003) — Wang landmarks
- Joren Six & Marc Leman, *Panako: A Scalable Acoustic Fingerprinting System* (ISMIR 2014); 2021 update — triplet β hash
- Jaap Haitsma & Ton Kalker, *A Highly Robust Audio Fingerprinting System* (ISMIR 2002) — band-power sign bits
- San Roman, R., Fernandez, P., Elsahar, H., Défossez, A., Furon, T. & Tran, T. *Proactive Detection of Voice Cloning with Localized Watermarking.* arXiv:2401.17264, 2024 (AudioSeal) — watermark model. <https://arxiv.org/abs/2401.17264>
