# audiofp

[![Crates.io](https://img.shields.io/crates/v/audiofp)](https://crates.io/crates/audiofp)
[![Documentation](https://docs.rs/audiofp/badge.svg)](https://docs.rs/audiofp)
[![License](https://img.shields.io/crates/l/audiofp)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/themankindproject/audiofp/ci.yml?branch=main&label=CI)](https://github.com/themankindproject/audiofp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/themankindproject/audiofp/branch/main/graph/badge.svg)](https://codecov.io/gh/themankindproject/audiofp)
![Crates.io Downloads](https://img.shields.io/crates/d/audiofp)
![Rust Version](https://img.shields.io/badge/rust-1.93%2B-blue)

Pure-Rust audio fingerprinting — **Wang (Shazam) landmarks, Panako tempo-robust triplets, and Haitsma–Kalker band hashes** with bit-exact streaming, in-memory matching, and BYO ONNX for neural/watermark.

> `no_std + alloc` by default. File decoding, neural, and watermark are opt-in.

---

## Contents

- [Why audiofp](#why-audiofp)
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Performance](#performance)
- [Robustness](#robustness)
- [Comparison](#comparison-with-alternatives)
- [Examples](#examples)
- [Security](#security)
- [Contributing](#contributing)

Full API reference and bit-layout specs: **[USAGE.md](USAGE.md)** · Codec methodology: **[ROBUSTNESS.md](ROBUSTNESS.md)** · Threat model: **[SECURITY.md](SECURITY.md)**

---

## Why audiofp

| You need | Use |
|---|---|
| "What is this song?" (Shazam-style) | **Wang** — ~300 hashes/s, 62.5 fps |
| Same but survives ±5% time-stretch | **Panako** — 2-D Hough + RANSAC, same rate |
| Ultra-compact ID / lowest latency | **Haitsma** — 312 B/s, 409 ms latency |
| Cover / remix similarity | **Neural** BYO ONNX log-mel embedder |
| Generative-AI provenance | **Watermark** AudioSeal-compatible detector |

Also: deduplication at scale, royalty enforcement, `bytemuck::Pod` persistence, real-time mic identification.

**Out of scope:** on-disk index / DB adapter / wire format beyond a tiny `to_bytes` blob (`src/serial.rs:10`). Persist with your store, match in-memory via `WangMatcher` / `WangIndex`.

---

## Features

- **3 classical algorithms** + streaming twins with **bit-exact offline parity** down to 1-sample-per-push
- **In-memory matching** — `WangMatcher` (offset histogram) / `HaitsmaMatcher` (BER + LUT) / `PanakoMatcher` (2-D Hough + RANSAC) + `match_best` / `match_ranked` and transient `WangIndex` / `HaitsmaIndex` / `PanakoIndex` for 1:N
- **Deterministic** — same PCM → same hashes, every time; `name()` version contract (`wang-v1`, `panako-v2`, `haitsma-v1`)
- **`bytemuck::Pod` hashes** — `WangHash` 8 B / `PanakoHash` 16 B `src/classical/wang.rs:65` — mmap/FFI without serialization; `to_bytes` blob `AUDIOFP\0` v1 also available
- **File decoding** via Symphonia (MP3/FLAC/WAV/OGG/AAC-in-MP4/…) + **Kaiser windowed-sinc resampler** with auto anti-alias cutoff
- **Watermark / Neural** — Tract ONNX; model cached per input length, streaming `try_push_with` zero-alloc after warmup
- **DSP primitives** — `dsp::stft` / `mel` / `peaks` / `resample` / `windows` all public and `no_std + alloc`
- **Hardened** — `DecodeLimits`, `max_input_samples` / `max_hashes` / `max_pending_anchors`, `InputTooLarge`, `NonFiniteSample`, cooperative `Timeout`

---

## Installation

```toml
[dependencies]
# Minimal — no codecs, no_std + alloc (DSP + classical only)
audiofp = { version = "0.4.1", default-features = false }

# File decoding — pick only what you decode:
audiofp = { version = "0.4.1", features = ["std-wav", "std-mp3"] }
audiofp = { version = "0.4.1", features = ["std-flac", "std-ogg", "std-mp4"] }
audiofp = { version = "0.4.1", features = ["all-codecs"] }  # every codec (pre-0.4 std)

# Heavy optional subsystems (each implies std, no codecs on their own):
audiofp = { version = "0.4.1", features = ["neural"] }      # ONNX embedder
audiofp = { version = "0.4.1", features = ["watermark", "std-wav"] }
audiofp = { version = "0.4.1", features = ["rayon"] }       # par_match_* + fingerprint_batch_parallel
```

Default build is `no_std + alloc` with no codecs. `audiofp::io` exists only with at least one `std-*` / `all-codecs` — bare `std` alone is a `compile_error!` `src/lib.rs:139`.

### Feature flags

| Feature | Description |
|---|---|
| `std-wav` | WAV + raw PCM → `audiofp::io` |
| `std-mp3` | MP3 |
| `std-flac` | FLAC |
| `std-ogg` | Ogg Vorbis |
| `std-aac` | AAC |
| `std-mp4` | AAC-in-MP4 / ISOBMFF (pulls AAC demuxer + decoder) |
| `std-aiff` / `std-mkv` / `std-adpcm` / `std-alac` | Extended codecs |
| `all-codecs` | All of the above at once |
| `rayon` | `fingerprint_batch_parallel` + `par_match_best` / `par_match_ranked` |
| `watermark` | `audiofp::watermark` (Tract) |
| `neural` | `audiofp::neural` (Tract, BYO model) |
| `mimalloc` | `mimalloc` as `#[global_allocator]` |

---

## Quick start

### 1) Fingerprint a file

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Resampled to Wang's native 8 kHz; needs ≥2 s or returns AudioTooShort.
    let samples = decode_to_mono_at("song.mp3", 8_000)?;

    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000)?;

    println!("{} hashes at {:.1} fps", fp.hashes.len(), fp.frames_per_sec);
    for h in fp.hashes.iter().take(5) {
        println!("  t_anchor={} hash={:08x}", h.t_anchor, h.hash);
    }
    Ok(())
}
```

Untrusted uploads — bound both phases:

```rust,no_run
use audiofp::io::{DecodeLimits, decode_to_mono_at_limited};
use std::time::Duration;
let limits = DecodeLimits::both(50_000_000, 14_400_000) // 50 MB on-disk, 30 min @8 kHz
    .with_timeout(Duration::from_secs(30));
let samples = decode_to_mono_at_limited("upload.mp3", 8_000, limits)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

### 2) Raw PCM (no `io` feature needed)

```rust
use audiofp::classical::Wang;
use audiofp::{Fingerprinter, SampleRate};

fn main() -> audiofp::Result<()> {
    let samples = vec![0.0_f32; 8_000 * 3]; // 3 s silence @8 kHz
    let mut wang = Wang::default();
    let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
    assert_eq!(fp.frames_per_sec, 62.5);
    assert!(fp.hashes.is_empty()); // silence → 0 hashes
    Ok(())
}
```

Decode `SampleRate` mismatch is `Err(UnsupportedSampleRate)`; NaN/Inf is `Err(NonFiniteSample)` offline, `0.0`-sanitized on streaming `push` `src/pcm.rs:28`.

### 3) Match two recordings

```rust
use audiofp::classical::Wang;
use audiofp::io::decode_to_mono_at;
use audiofp::matching::{Matcher, WangMatchConfig, WangMatcher};
use audiofp::{Fingerprinter, SampleRate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q = {
        let s = decode_to_mono_at("query.wav", 8_000)?;
        Wang::default().extract(&s, SampleRate::HZ_8000)?
    };
    let r = {
        let s = decode_to_mono_at("reference.flac", 8_000)?;
        Wang::default().extract(&s, SampleRate::HZ_8000)?
    };

    let m = WangMatcher::new(WangMatchConfig::default()).match_one(&q, &r);
    println!("is_match={} score={:.3} offset={}ms prominence={:.1}", m.is_match, m.score, m.offset.ms, m.prominence);
    Ok(())
}
```

Repeated 1:1 against a fixed catalog — build once, reuse:

```rust
use audiofp::classical::{WangFingerprint, WangHash};
use audiofp::matching::{WangMatchConfig, WangMatcher, WangRefIndex, Matcher};

let reference = WangFingerprint { hashes: vec![WangHash{hash:1,t_anchor:0}], frames_per_sec:62.5 };
let cfg = WangMatchConfig::default();
let matcher = WangMatcher::new(cfg.clone());
let index = WangRefIndex::build(&reference, &cfg).unwrap();
let query = reference.clone();
assert_eq!(matcher.match_one(&query, &reference), matcher.match_one_prebuilt(&query, &index));
```

### 4) Streaming (mic / chunked file)

```rust
use audiofp::StreamingFingerprinter;
use audiofp::classical::StreamingWang;
use std::f32::consts::PI;

fn main() -> audiofp::Result<()> {
    let mut s = StreamingWang::default();
    let sr = s.required_sample_rate(); // 8_000 — push has no runtime rate check
    let chunk_len = (sr / 5) as usize; // 200 ms
    let mut total = 0usize;

    for i in 0..25 {
        let chunk: Vec<f32> = (0..chunk_len).map(|j| {
            let t = (i * chunk_len + j) as f32 / sr as f32;
            0.5*(2.0*PI*880.0*t).sin() + 0.3*(2.0*PI*1320.0*t).sin()
        }).collect();

        // Zero-alloc variant also available: s.push_with(&chunk, |ts, hash| …)?
        for (ts, hash) in s.push(&chunk)? {
            let _ = (ts.0, hash.hash);
            total += 1;
        }
    }
    total += s.flush()?.len();          // idempotent; push after flush is valid — use reset() for a fresh stream
    println!("{total} hashes; latency {} ms", s.latency_ms()); // 2256 ms Wang, 2784 Panako, 409 Haitsma
    Ok(())
}
```

Bit-exact: any chunking (even 1 sample per `push`) yields the identical hash multiset as offline `extract`.

Compact alternative for audio callbacks: `push_with` / `flush_with` — no `Vec` allocation.

---

## Architecture

### Fingerprint types (`bytemuck::Pod`, `repr(C)`)

```
Wang          8 B  { hash: u32, t_anchor: u32 }           frames_per_sec 62.5 (8000/128)
Panako       16 B  { hash: u32, t_anchor, t_b, t_c }      frames_per_sec 62.5
Haitsma       4 B  frames: Vec<u32>  (32 bits / frame)    frames_per_sec 78.125 (5000/64)
Neural        variable  Vec<NeuralEmbedding { vector, t_start }>  frames_per_sec = 1/hop_secs
```

Wang hashes sorted `(t_anchor, hash)` `src/classical/wang.rs:312`; Panako `(t_anchor,t_b,t_c,hash)` `src/classical/panako.rs:323`; Haitsma temporal order. Hash layouts, STFT params, and peak-picker contract are pinned in **[USAGE.md](USAGE.md)** — reimplementing from there is byte-identical.

### Pipeline (all three classical share the front-end to dB)

```
PCM @ native rate → Hann STFT (non-centered) → power |X|² → 10·log10 → 31×31 peak picker → adaptive top-30/s → pairing
```

Wang pairs `Δt≤63, |Δf|≤64` top-10; Panako triplets `β=(t_c-t_b)/(t_c-t_a)·31`; Haitsma 33 log-bands 300–2000 Hz → 32 sign bits `src/classical/haitsma.rs:33`.

---

## Performance

Measured on 30 s synthetic audio (`cargo bench --bench extract`), Intel i5-1135G7 with `RUSTFLAGS="-C target-cpu=native"` (enables POPCNT/AVX2/FMA for `wide`+FFT). Reproduce: `cargo bench --bench extract -- --save-baseline main`.

| Algorithm | 30 s audio | Realtime | Streaming push (10 s) | Latency |
|---|---|---|---|---|
| **Wang** | 79 ms | 380× | 10.5 ms | 2256 ms |
| **Panako** | 81 ms | 370× | 11.6 ms | 2784 ms |
| **Haitsma** | 42 ms | 714× | 6.3 ms | 409 ms |

Matching (5 s fingerprints, `cargo bench --bench matching`):

| Path | Time | Notes |
|---|---|---|
| `WangMatcher` 1:1 | ~111 µs | histogram + prominence |
| `HaitsmaMatcher` 1:1 | ~18 µs | BER + POPCNT |
| `PanakoMatcher` 1:1 | ~264 µs | 2-D Hough + RANSAC |
| `WangIndex` N=100 | ~102 µs | ~9.8k q/s, scales ~linearly |

> For >10k tracks shard the index or raise `min_votes`/`min_score`. Full tables in [USAGE.md](USAGE.md).

Neural front-end (`--features neural --bench neural_frontend`): `log_mel 1s window` 297 µs, `strided write` 7.6 µs.

---

## Robustness

Spectral-peak (Wang/Panako) and band-delta (Haitsma) designs survive lossy transcoding. Verified on **"Galway" / "Furious Freak" (CC-BY, 16 s, 6 codecs)** — same-track Jaccard 0.36–0.54 (Wang/Panako) and bit-sim 0.77–0.93 (Haitsma) at 128 kbps; cross-track <0.001. Thresholds (`Wang ≥0.25`, `Panako ≥0.20`, `Haitsma ≥0.75`) calibrated so 600+ tests separate every same-track vs cross-track pair with margin ≥0.35.

Reproduce: `cargo test --test codec_roundtrip --all-features -- --nocapture` · Full methodology and numbers: **[ROBUSTNESS.md](ROBUSTNESS.md)** · Threshold sweep: `tests/threshold_calibration.rs`.

---

## Comparison with alternatives

| Feature | audiofp | chromaprint-rust | dejavu (Python) |
|---|---|---|---|
| Pure Rust | Yes | No (FFI) | No |
| Wang landmarks | Yes | No | Yes |
| Panako triplets (tempo-robust) | Yes | No | No |
| Haitsma–Kalker | Yes | No | No |
| Streaming + bit-exact parity | Yes | Limited | No |
| File decoding | Yes (Symphonia) | Limited | FFmpeg |
| Watermark (AudioSeal) | Yes | No | No |
| `no_std + alloc` | Yes (host) | No | — |
| `Pod` hash types | Yes | No | — |
| Built-in resampler | Yes | No | No |
| In-memory matcher | Yes (3 algos) | No | Yes |

---

## Examples

`examples/` are runnable with `cargo run --example <name>`:

| Example | What it does | Features |
|---|---|---|
| `dsp_starter` | STFT→mel→peaks on synthetic audio | none |
| `stream_buffer` | `StreamingWang` from `io::Read` chunks | none |
| `enroll_file` | Wang hash count for one file | `all-codecs` |
| `match_two_files` | Hash collisions between two files | `all-codecs` |
| `compare_algorithms` | Wang vs Panako vs Haitsma on one file | `all-codecs` |
| `neural_embed` | BYO ONNX embedding dim | `neural` |
| `watermark_detect` | AudioSeal confidence + message | `watermark,std-wav` |

```bash
cargo run --example dsp_starter
cargo run --example compare_algorithms --features all-codecs -- song.flac
cargo run --example neural_embed --features neural -- model.onnx
cargo run --example watermark_detect --features watermark,std-wav -- audioseal.onnx audio.wav
```

Doctests in `src/` + [USAGE.md](USAGE.md) cover the full surface.

---

## Security

Fingerprints are **perceptual**, not cryptographic. See [SECURITY.md](SECURITY.md) for trust boundaries and disclosure.

- **Untrusted audio:** always use `DecodeLimits` + `decode_to_mono_limited` / `decode_to_mono_at_limited` (`max_bytes` + `max_samples` + optional `Timeout`). Without it a tiny compressed file can expand to GiB of PCM and OOM.
- **Untrusted PCM / ONNX:** offline `extract`/`detect` reject NaN/Inf (`NonFiniteSample`); streaming `push` sanitizes to `0.0`. ONNX path is `unsafe` tensor init only where noted `src/neural/embedder.rs:3`.
- **Hash outputs:** not MACs — don't use for authenticity.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Quick start:

```bash
git clone https://github.com/themankindproject/audiofp && cd audiofp
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

CI runs `fmt` + `clippy` + `test` on Ubuntu/macOS/Windows per push/PR. MSRV 1.93.

---

## License

MIT — see [LICENSE](LICENSE).

---

## References

- Wang, A. *An Industrial-Strength Audio Search Algorithm* (ISMIR 2003) — exp. 2024-01-07, free to use
- Six, J. & Leman, M. *Panako — A Scalable Acoustic Fingerprinting System* (ISMIR 2014) + 2021 update — triplet β
- Haitsma, J. & Kalker, T. *A Highly Robust Audio Fingerprinting System* (ISMIR 2002)
- San Roman et al. *Proactive Detection of Voice Cloning with Localized Watermarking* (AudioSeal, arXiv:2401.17264)
