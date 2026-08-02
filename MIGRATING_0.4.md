# Migrating to audiofp 0.4.0

This guide covers every breaking change shipped in **0.4.0**. All changes
were batched into this release per the API-reshape epic
([#85](https://github.com/themankindproject/audiofp/issues/85)); there is
no migration between intermediate 0.3.x releases.

Quick reference:

| Change | Old API | New API |
|---|---|---|
| [#65] Drop `AudioBuffer` | `fp.extract(buf)` | `fp.extract(&samples, rate)` |
| [#66] Hash timestamps → `TimestampMs` | `h.t_anchor: u32` (frames) | `h.t_anchor: TimestampMs` (ms) |
| [#63] `push`/`flush` return `Result` | `s.push(&x)` → `Vec` | `s.push(&x)?` → `Result<Vec>` |
| [#62] `min_magnitude` → `min_magnitude_db` | `min_magnitude: f32` | `min_magnitude_db: f32` + `min_magnitude_linear` |
| [#64] Flat crate-root re-exports | `use audiofp::classical::Wang` | `use audiofp::Wang` (alias) |
| [#66] Serialization format v2 | `FORMAT_VERSION = 1` | `FORMAT_VERSION = 2` (v1 rejected) |

[#62]: https://github.com/themankindproject/audiofp/issues/62
[#63]: https://github.com/themankindproject/audiofp/issues/63
[#64]: https://github.com/themankindproject/audiofp/issues/64
[#65]: https://github.com/themankindproject/audiofp/issues/65
[#66]: https://github.com/themankindproject/audiofp/issues/66

---

## 1. `AudioBuffer` removed — `extract` takes `&[f32]` + `SampleRate` (#65)

The `AudioBuffer<'a>` wrapper is gone. All `Fingerprinter::extract` (and
`extract_with_progress`, watermark `detect`, neural `extract`) now take
the sample slice and rate as separate arguments:

```rust
// 0.3.x
let buf = AudioBuffer { samples: &samples, rate: SampleRate::HZ_8000 };
let fp = wang.extract(buf)?;

// 0.4.0
let fp = wang.extract(&samples, SampleRate::HZ_8000)?;
```

Mechanical migration:

- `AudioBuffer::new(&x, R)` / `AudioBuffer { samples: &x, rate: R }` →
  pass `(&x, R)` directly to the method.
- `AudioBuffer` imports and the `prelude` re-export are removed — delete
  `use audiofp::AudioBuffer;`.

This removes the lifetime parameter from the public surface, simplifying
generic code.

## 2. Hash timestamps are now `TimestampMs` (#66)

`WangHash::t_anchor` and `PanakoHash::t_anchor` / `t_b` / `t_c` changed
from raw `u32` STFT-frame indices to `TimestampMs` (milliseconds since
stream start):

```rust
// 0.3.x — frame units, 62.5 fps → 16 ms/frame
println!("anchor frame {}", h.t_anchor);

// 0.4.0 — milliseconds
println!("anchor at {} ms", h.t_anchor.0);
```

Notes:

- `TimestampMs` is `#[repr(transparent)]` over `u64`, is `bytemuck::Pod`,
  and implements `Ord` — sorting by `(t_anchor, hash)` still works.
- The hash **byte layout** changed (Wang 8 → 12 bytes, Panako 16 → 28
  bytes; structs are `#[repr(C, packed)]` with no padding). Persisted
  fingerprints and golden files are invalidated and must be regenerated
  (see §6).
- `Fingerprint::name()` bumped: `wang-v1` → `wang-v2` (Panako was
  already `panako-v2`; Haitsma unchanged at `haitsma-v1` — its frames
  are dense per-frame codes, not timestamps).
- Matching internals are unchanged: matchers convert `TimestampMs` →
  frames at the boundary using `frames_per_sec`, so match behaviour is
  identical.

## 3. `StreamingFingerprinter::push` / `flush` return `Result` (#63)

The streaming trait methods now return `Result`, so neural inference
errors are recoverable instead of panicking:

```rust
// 0.3.x — infallible (neural panicked on inference error)
let frames = s.push(&chunk);

// 0.4.0 — fallible
let frames = s.push(&chunk)?;
let tail = s.flush()?;
```

`push_with` / `flush_with` also return `Result<usize>`.

For classical fingerprinters the `Result` is always `Ok`; for
`StreamingNeuralEmbedder` it is `Err(AfpError::…)` when ONNX inference
fails. The old `try_push` / `try_push_with` methods remain available for
explicit error handling.

## 4. `PeakPickerConfig::min_magnitude` renamed (#62)

The field was misnamed: Wang/Panako pass a **dB** value
(`min_anchor_mag_db`) into what was documented as a linear floor.

```rust
// 0.3.x
PeakPickerConfig { min_magnitude: cfg.min_anchor_mag_db, .. }

// 0.4.0 — honest dB contract
PeakPickerConfig { min_magnitude_db: cfg.min_anchor_mag_db, .. }
```

A new optional `min_magnitude_linear: Option<f32>` provides a genuine
linear floor for callers feeding `pick` raw (pre-log) spectrograms. When
set, cells must exceed **both** floors.

## 5. Flat crate-root re-exports (#64)

The major classical types are re-exported at the crate root:

```rust
// Both work; the canonical location is audiofp::classical
use audiofp::Wang;
use audiofp::classical::Wang;
```

Re-exported: `Wang`, `WangConfig`, `WangFingerprint`, `WangHash`,
`StreamingWang`, `Panako`, `PanakoConfig`, `PanakoFingerprint`,
`PanakoHash`, `StreamingPanako`, `Haitsma`, `HaitsmaConfig`,
`HaitsmaFingerprint`, `StreamingHaitsma`.

## 6. Serialization format v2 (#66)

`FingerprintEnvelope::to_bytes` now writes `FORMAT_VERSION = 2`. Version
1 blobs (0.3.x, 8/16-byte hashes) are **rejected** with
`AfpError::UnsupportedVersion` — there is no read-and-migrate path.
Re-extract and re-serialize any persisted fingerprints.

Golden regression files were regenerated with `UPDATE_GOLDENS=1`.

---

## How to verify your migration

```bash
cargo test --all-features        # unit + integration + doctests
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo build --no-default-features
```
