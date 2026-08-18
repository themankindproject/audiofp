# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- **`AudioBuffer` removed** (#65) — `Fingerprinter::extract`,
  `extract_with_progress`, watermark `detect`, and neural `extract` now
  take `(&[f32], SampleRate)` directly. The `prelude` no longer
  re-exports `AudioBuffer`.
- **Hash timestamps keep frame units** (#66, reverted) — the
  originally-proposed conversion of `WangHash::t_anchor` and
  `PanakoHash::t_anchor`/`t_b`/`t_c` from raw `u32` STFT frames to
  `TimestampMs` was evaluated and **reverted before release**: it would
  have grown the hash byte layout (Wang 8→12, Panako 16→28 bytes),
  forced a serialization `FORMAT_VERSION` bump to 2 with v1-blob
  rejection, and provided no algorithmic benefit (matching is
  frame-native and would convert back at the boundary). Frame-index
  timestamps and the 8/16-byte v1 layouts are preserved; `TimestampMs`
  remains the unit of the streaming emit tuples, which were always ms.
- **`StreamingFingerprinter::push` / `flush` return `Result`** (#63) —
  `push`/`flush`/`push_with`/`flush_with` are now fallible
  (`Result<Vec<…>>` / `Result<usize>`); `StreamingNeuralEmbedder` no
  longer panics on inference errors.
- **`PeakPickerConfig::min_magnitude` renamed to `min_magnitude_db`**
  (#62) — plus a new optional `min_magnitude_linear: Option<f32>` floor.
- **Flat crate-root re-exports** (#64) — `audiofp::Wang`,
  `audiofp::Panako`, `audiofp::Haitsma`, configs, fingerprints, and
  streaming variants are re-exported at the crate root (the
  `classical` module remains canonical).
- **`std` feature split into per-codec sub-features** (#60) — the
  monolithic `std` feature is replaced by `std-mp3`, `std-aac`,
  `std-flac`, `std-ogg`, `std-wav`, `std-mp4`, plus extended `std-aiff` /
  `std-mkv` / `std-adpcm` / `std-alac`, and the `all-codecs` feature
  restores every codec at once. `default` is now `[]`, so the default
  build is `no_std + alloc` with **no codecs**. `audiofp::io` is only
  available with at least one `std-*` feature (or `all-codecs`);
  enabling bare `std` without a codec is a `compile_error!` pointing at
  the feature list. `neural`, `watermark`, `rayon`, and `mimalloc` are
  unaffected (they imply `std` but not any codec).
- **`all` feature renamed to `all-codecs`** — the old `all` name didn't
  convey that the feature covers only the *codec* features and not the
  heavyweight optional subsystems (`neural`, `watermark`, `rayon`,
  `mimalloc`). The rename ships in the same unreleased 0.4.0 cycle that
  introduced the feature, so there is no back-compat alias: update
  `features = ["all"]` to `features = ["all-codecs"]`.
- **`Fingerprinter::required_sample_rate()` returns `SampleRate`** (#61) —
  the offline trait now returns the `SampleRate` newtype
  (`SampleRate::HZ_8000` for Wang/Panako, `SampleRate::HZ_5000` for
  Haitsma) instead of a bare `u32`, so callers compare it directly
  against an audio buffer's rate without an unwrap. `StreamingFingerprinter::required_sample_rate()`
  still returns `u32` (streams feed raw `&[f32]` with no rate tag).
- **`ShortTimeFFT::process_frame` / `process_frame_power` return
  `Result`** (#8) — both methods now return `Result<(), AfpError>` and
  reject mismatched `frame`/`out` lengths with `AfpError::Config`
  instead of panicking via `assert_eq!`.

### Migration

All changes were batched into 0.4.0 per the API-reshape epic
([#85](https://github.com/themankindproject/audiofp/issues/85)); there is
no migration between intermediate 0.3.x releases.

Quick reference:

| Change | Old API | New API |
|---|---|---|
| [#65] Drop `AudioBuffer` | `fp.extract(buf)` | `fp.extract(&samples, rate)` |
| [#66] Hash timestamps unchanged | `h.t_anchor: u32` (frames) | `h.t_anchor: u32` (frames) — **no change** |
| [#63] `push`/`flush` return `Result` | `s.push(&x)` → `Vec` | `s.push(&x)?` → `Result<Vec>` |
| [#62] `min_magnitude` → `min_magnitude_db` | `min_magnitude: f32` | `min_magnitude_db: f32` + `min_magnitude_linear` |
| [#64] Flat crate-root re-exports | `use audiofp::classical::Wang` | `use audiofp::Wang` (alias) |
| [#60] Per-codec features | `features = ["std"]` | `features = ["std-wav"]` (pick your codecs) or `["all-codecs"]` |
| [#61] `required_sample_rate` → `SampleRate` | `let sr = fp.required_sample_rate();` → `u32` | `let sr = fp.required_sample_rate();` → `SampleRate` |
| [#8] `process_frame*` returns `Result` | `stft.process_frame_power(&f, &mut o);` | `stft.process_frame_power(&f, &mut o)?;` |

[#62]: https://github.com/themankindproject/audiofp/issues/62
[#63]: https://github.com/themankindproject/audiofp/issues/63
[#64]: https://github.com/themankindproject/audiofp/issues/64
[#65]: https://github.com/themankindproject/audiofp/issues/65
[#66]: https://github.com/themankindproject/audiofp/issues/66
[#60]: https://github.com/themankindproject/audiofp/issues/60
[#61]: https://github.com/themankindproject/audiofp/issues/61
[#8]: https://github.com/themankindproject/audiofp/issues/8

#### 1. `AudioBuffer` removed — `extract` takes `&[f32]` + `SampleRate` (#65)

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

#### 2. Hash timestamps: no change — kept as frame indices (#66)

The proposed conversion of `WangHash::t_anchor` and
`PanakoHash::t_anchor` / `t_b` / `t_c` from raw `u32` STFT-frame indices
to `TimestampMs` was **evaluated and reverted before release**. Hash
timestamp fields remain `u32` frame indices:

```rust
// 0.3.x and 0.4.0 — identical
println!("anchor frame {}", h.t_anchor);
```

Why it was reverted:

- The hash **byte layout** would have grown (Wang 8 → 12 bytes, Panako
  16 → 28 bytes), invalidating persisted fingerprints and requiring a
  serialization `FORMAT_VERSION` bump with v1-blob rejection.
- Matching is frame-native (δ histograms, scale ratios, tolerances,
  RANSAC) and would convert ms → frames at the boundary anyway — the
  change was pure representation churn with no algorithmic benefit.
- `TimestampMs` remains the unit of the **streaming emit tuples**,
  which were already ms — no migration there either.

`Fingerprint::name()` is unchanged (`wang-v1`, `panako-v2`,
`haitsma-v1`), `FORMAT_VERSION` stays 1, and persisted v1 blobs and
golden files remain valid.

#### 3. `StreamingFingerprinter::push` / `flush` return `Result` (#63)

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

#### 4. `PeakPickerConfig::min_magnitude` renamed (#62)

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

#### 5. Flat crate-root re-exports (#64)

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

#### 6. Serialization format: unchanged (v1) (#66)

`FingerprintEnvelope::to_bytes` still writes `FORMAT_VERSION = 1`, and
the 8-byte (Wang) / 16-byte (Panako) / 4-byte-per-frame (Haitsma) hash
layouts are preserved. Persisted 0.3.x fingerprints and golden files
remain valid — no re-extraction needed.

#### 7. Per-codec features — `std` split into `std-*` (#60)

The biggest *build-time* change. `audiofp = "0.3"` used to default to the
full Symphonia decoder stack; now `default = []` and codecs are opt-in:

```toml
# 0.3.x — all 8 codecs, always
audiofp = "0.3"

# 0.4.0 — pick the codecs you decode
audiofp = { version = "0.4", features = ["std-wav"] }
audiofp = { version = "0.4", features = ["std-mp3", "std-flac", "std-ogg"] }

# 0.4.0 — every codec, exactly like the old `std`
audiofp = { version = "0.4", features = ["all-codecs"] }
```

- Each `std-*` feature pulls only its own Symphonia codec (plus `pcm` for
  `std-wav`, `vorbis` for `std-ogg`); `all-codecs` pulls every codec at
  once.
- `audiofp::io` exists only when at least one `std-*` feature (or
  `all-codecs`) is on.
- Bare `std` without a codec feature is a `compile_error!` with a helpful
  message — you can't silently lose decoding.
- `neural`, `watermark`, `rayon`, `mimalloc` imply `std` but **not** any
  codec; they keep working unchanged.
- Migration for 0.3.x users: replace `features = ["std"]` with
  `features = ["all-codecs"]` for identical behavior, or `features = ["std-wav"]`
  (or whichever codecs you need) to trim the build. No source-code
  changes.

#### 8. `required_sample_rate()` returns `SampleRate` (#61)

```rust
// 0.3.x — u32, callers unwrap
let want = fp.required_sample_rate();
if rate.hz() != want { /* … */ }

// 0.4.0 — SampleRate, direct comparison
let want = fp.required_sample_rate();
if rate != want { /* … */ }
```

Mechanical for all four implementors (`Wang`, `Panako`, `Haitsma`,
`NeuralEmbedder`); the streaming trait is unchanged (`u32`).

#### 9. `process_frame` / `process_frame_power` return `Result` (#8)

```rust
// 0.3.x — panics on size mismatch
stft.process_frame_power(&frame, &mut out);

// 0.4.0 — returns Result<(), AfpError>
stft.process_frame_power(&frame, &mut out)?;
```

Mismatched `frame.len() != n_fft` or `out.len() != n_bins` now returns
`AfpError::Config` instead of panicking. All built-in streaming
fingerprinters call these with exact-length buffers, so their public
behaviour is unchanged.

#### How to verify your migration

```bash
cargo test --all-features        # unit + integration + doctests
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo build --no-default-features
cargo build --features std-wav   # codec picker works
```

### Added
- **`FingerprintEnvelope::peek`** — reads a blob's metadata (algorithm,
  sample rate, frame rate, hash count) from the fixed 18-byte header
  without touching the hash payload, for triaging mixed-format blobs
  before a full decode.
- **`DecodeLimits::timeout` — wall-clock decode timeout (#77).**
  `DecodeLimits::with_timeout(Duration)` builder sets a maximum
  wall-clock time for the entire decode operation. Returns
  `AfpError::Timeout { elapsed_ms, limit_ms }` if the limit is
  exceeded. Checked per-packet (~1 ns overhead). Use in Python FFI /
  multi-tenant services to prevent adversarial inputs from hanging
  decode workers indefinitely. Default: `None` (no limit).

- **In-memory matching subsystem (`audiofp::matching`).**
  `WangMatcher`, `HaitsmaMatcher`, `PanakoMatcher`, `NeuralMatcher`
  (feature-gated), plus `match_best` / `match_ranked` convenience
  functions and transient `WangIndex` / `HaitsmaIndex` / `PanakoIndex`
  accelerators for 1:N queries. Purely in-memory — no persistence, wire
  format, or DB adapters.
- **`PanakoMatcher` — full 2-D Hough + RANSAC.** Tempo-invariant
  matching with a sparse `(scale, offset)` accumulator, neighbourhood
  consolidation, and optional deterministic RANSAC line-fitting; reports
  a meaningful `MatchResult::time_scale` (reciprocal of the fitted
  scale, clamped to `[0.5, 2.0]`).
- **`HaitsmaIndex` — 1:N sub-fingerprint LUT accelerator.** Probes each
  query frame against a combined `u32 → (ref_id, frame_pos)` LUT and
  verifies the best per-reference alignment with the exact-BER path.
- **`PanakoIndex` — 1:N 2-D Hough accelerator.** Per-reference sparse
  accumulator over a shared inverted index of Panako triplets.
- **`WangMatcher` / `HaitsmaMatcher` implement `Default`.**
- **`benches/matching.rs`** — Criterion benchmarks for Wang/Haitsma/
  Panako 1:1 and small `WangIndex` (N=100) queries.
- **Pipeline E2E + adversarial test suites (30 tests).**
  `tests/pipeline.rs` (self-match, offset recovery, tempo speed-up,
  1:N catalog identification, unrelated rejection for all three
  algorithms) and `tests/adversarial.rs` (silence/DC, wrong sample
  rate, short input, clipping, determinism, empty catalog, no false
  positives), backed by four deterministic audio generators in
  `tests/common/audio_gen.rs`.
- **`serial_roundtrip` fuzz target (0.4.0 audit).** Exercises the
  binary serialization layer against untrusted bytes — every public
  deserializer (`WangFingerprint` / `PanakoFingerprint` /
  `HaitsmaFingerprint` `from_bytes`, `FingerprintEnvelope::peek`) must
  return `Ok`/`Err` and never panic — plus roundtrip integrity for all
  three algorithms and cross-algorithm blob rejection. Registered in
  `fuzz/Cargo.toml` and the CI fuzz-smoke loop (9 → 10 targets).

### Changed

- **Matching hot path uses `HashMap` under `std`** (default). Without
  `std`, the same code paths fall back to `BTreeMap` via an internal
  alias so `no_std + alloc` builds keep working.
- **`WangMatcher` hot path uses `SortedPostings`** — a flat sorted array
  (`hashes` / `starts` / `anchors`) with binary-search lookup replaces
  `HashMap<u32, Vec<u32>>`, eliminating per-unique-hash allocations and
  pointer chasing.
- **`match_best` early-exits** when a reference scores 1.0.
- **`PanakoMatcher` soft-fails on frame-rate mismatch** instead of
  silently converting offsets with the reference rate.
- **`wide` pinned to exactly `=1.6.0` (0.4.0 audit).** 1.6.1 (and
  1.5.0) are broken with `safe_arch` 1.1.0 (missing AVX-512
  intrinsics); 1.6.0 is yanked upstream but is the only known-good
  release. A caret requirement would let a fresh downstream
  `cargo update` resolve to broken 1.6.1 — the exact pin (already
  allow-listed in `deny.toml`) prevents that.

### Performance
- **5.5× faster Haitsma BER computation** — `hamming_at_offset` now
  processes frames in chunks of 64 without a per-element early-abort
  branch, enabling LLVM to auto-vectorize the XOR+POPCNT inner loop.
  Haitsma 1:1 self-match: 96 µs → 18 µs.
- **AHash replaces SipHash** for all internal `HashMap` usage in the
  matching layer. 2-5× faster hashing on integer keys (compile-time-rng
  initialised). Panako 1:1: 315 µs → 264 µs (19% faster).
- **Branchless RANSAC inlier counting** in `PanakoMatcher` — extracted
  into `count_inliers_wide()` using `(condition) as u32` accumulation.
  LLVM auto-vectorizes with `target-cpu=native`. Panako 1:1: 264 → 261 µs.
- **`target-cpu=native`** in `.cargo/config.toml` for bench/release
  builds — unlocks hardware POPCNT, AVX2, and FMA for `wide` and FFT.

- **Pre-sized `HashMap` in `WangMatcher::match_one` and `WangIndex::build`**
  — eliminates repeated rehashing during index construction.
- **O(B²) → O(B) consolidation** in `WangIndex::query` via sorted
  sliding window (previously quadratic in number of distinct offsets;
  now linear regardless of hash-collision density).
- **O(B²) → O(B·W) consolidation** in `PanakoMatcher::match_one` —
  sorted accumulator with bounded-radius scan replaces full pairwise
  comparison. Typical W is 5-15 for musical inputs, so effectively
  linear.
- **Pre-sized `HashMap` in `HaitsmaIndex::build_lut`** — eliminates
  ~10 re-hashes during LUT construction for typical reference lengths.
- **Early-abort on perfect score** in `WangIndex::query`,
  `HaitsmaIndex::query`, and `PanakoIndex::query` — when a reference
  scores `>= 1.0`, return immediately without evaluating remaining
  candidates.
- **`min_votes` pre-filter** in `WangIndex::query` — references whose
  total raw vote count is below `cfg.min_votes` are skipped before the
  expensive consolidation + prominence + score pipeline.
- **`min_votes` pre-filter** in `PanakoIndex::query` — sum of all bin
  votes for a reference is checked before consolidation.
- **`mem::take` instead of `hist.clone()`** when offset tolerance is 0.
- **Sorted Vec + dedup** replaces `HashMap<u32, ()>` for contrib counting.
- **`SortedPostings`** — Wang 1:1 self-match on 5 s audio: ~107 µs
  (single allocation vs N+1 per unique hash).
- **O(B²) → O(B·W) consolidation in `PanakoIndex::query`** — sorted
  `(scale_bin, offset_bin)` accumulator with bounded-radius scans
  replaces the full pairwise comparison, matching `PanakoMatcher`.
  Also makes peak selection deterministic (sorted order instead of
  `HashMap` iteration order).
- **Pre-sized `HashMap` in `PanakoIndex::build`** — matches
  `WangIndex`; eliminates repeated rehashing during index construction.
- **Streaming Panako emit no longer clones per anchor** — the emit
  closure now consumes the finalized `PendingAnchor` by value and sorts
  its targets in place, restoring the zero-copy emit path from the
  streaming-core extraction.
- **SIMD magnitude spectra** — `ShortTimeFFT::magnitude_flat` and
  `process_frame` now compute `sqrt(re² + im²)` 8-lane via
  `wide::f32x8::sqrt` instead of scalar `libm::sqrtf`. ~4× faster
  (513 bins: 2056 → 486 ns); bit-identical on default builds (hardware
  IEEE sqrt vs libm's musl-derived software sqrt).
- **SIMD L2 normalisation** — the neural embedder's embedding
  normalisation accumulates sum-of-squares via `wide::f32x8` (~5.4×
  faster at 1024 dims: 2453 → 457 ns).
- **SIMD neural-matcher dot product** — `NeuralMatcher`'s cosine
  kernel (the `Nq·Nr` inner loop of SlidingMax and DTW) is vectorised
  8-wide (~8× faster at 256 dims: 510 → 62 ns).
- **`u32` reference ids in inverted-index posting lists (#125)** —
  `WangIndex` / `HaitsmaIndex` / `PanakoIndex` store `(u32, u32)` /
  `(u32, u32, u32, u32)` postings instead of `usize`-wide ids: 33 %
  smaller posting memory on 64-bit, better cache density. Build panics
  if the reference count exceeds `u32::MAX`; public `query` signatures
  are unchanged.
- **Threshold calibration suite (#104)** — new
  `tests/threshold_calibration.rs` measures same-track cross-codec vs
  cross-track score/prominence distributions over the real CC0 catalog
  and pins that the shipped defaults separate every pair. The measured
  margins are documented in `ROBUSTNESS.md`.
- **O(N²) → O(N) `StreamingNeuralEmbedder` push** — see Fixed: the
  per-window front drain is replaced by a read cursor with one
  compaction per call.
- **Watermark inference on an optimised graph** — see Fixed:
  `into_optimized()` added to the cached per-length plan.
- **Deserialisation no longer double-writes the payload** —
  `from_bytes` zero-filled the destination `Vec` and immediately
  overwrote it with `copy_from_slice`; it now builds the `Vec<T>` with
  a single copy via `bytemuck::pod_collect_to_vec` (bytemuck's
  `extern_crate_alloc` feature).

### Documentation

- **Comment cleanup across the crate** — removed redundant comments
  that merely restated the code (band-energy loop, bin-index arithmetic,
  SIMD power expression, LUT probing, stop-hash pruning, decoder
  packet-skip branches, watermark runnable caching), merged a duplicated
  `StreamingHaitsma` doc block, and added genuine comments where
  non-obvious logic was unexplained (Wang's inclusive target-zone
  emission bound, Panako's post-truncation `(t, f)` re-sort, Haitsma's
  `< 2` frames threshold, polyphase kernel scaling and `wrapping_sub`
  safety, histogram OOM cap / vote wrap, RANSAC iteration budget,
  half-bin scale slack, BER-prominence sentinels, DTW tempo ratio, the
  decoder's `ResetRequired` and track-id handling, peak-picker deque /
  plateau `>=` semantics, serialisation trailing-bytes tolerance, mel
  normalisation floor).
- **Fixed misleading doc comments** — `decode_to_mono` had swapped
  `# Example` / `# Security` sections; the neural module example claimed
  16 kHz while passing `HZ_8000`; the `MatchResult::prominence` doc
  misstated the Haitsma matcher's formula (it uses `median_BER / (BER +
  ε)`, only the index path uses `0.5 / BER`).
- **Stale crate-doc sections refreshed** — the crate-root "Panics in
  streaming APIs" section still described the pre-#63 world where
  `StreamingNeuralEmbedder::push` panicked on inference errors; the
  `fp` module doc referenced modules that never existed
  (`fp_classical::Wang`, `neural::ResonaFp`); the feature table omitted
  `rayon`; the neural/watermark doc TOML snippets said `version = "0.3"`.
- **`DecodeLimits::timeout` documents its cooperative nature (0.4.0
  audit).** The deadline is only checked between packets; time spent
  inside container probing, a single packet decode, or the resample
  step is not interruptible, so a pathological stream can still exceed
  the timeout. The field doc and the check site now say so explicitly
  and point callers needing a hard wall-clock guarantee at a watchdog
  thread.
- **README feature table gained the missing `rayon` row** and
  **SECURITY.md's supported-versions table now lists 0.4.x** (0.4.0
  audit).

### Fixed

- **Haitsma BER selection is rate-normalized across offsets.** Both the
  exact-BER path and the LUT verification path compared candidates by
  **absolute** Hamming totals (and passed that total as the
  early-abort bound). Candidates at different offsets have different
  overlap lengths, so a short overlap with a small raw total but a
  *worse* bit-error rate could suppress a longer, strictly better-rate
  alignment — yielding wrong offsets, inflated scores, or dropped true
  matches on noisy audio. Selection and pruning now operate on BER
  (`hamming ÷ overlap·32`); clean bit-exact matches are unaffected.
- **Panako Hough offsets are 1-frame precise.** Offset votes were binned
  at `offset_tolerance_frames`-frame granularity and then consolidated
  over ±tol *bins* — an effective ±tol²-frame window that quantized
  the reported offset to multiples of `tol` whenever
  `offset_tolerance_frames > 1`. Binning is now 1-frame with a ±tol-bin
  (= ±tol-frame) consolidation window, matching `WangMatcher`'s
  semantics. Default `tol = 1` behaviour is unchanged.
- **`PanakoMatchConfig` degenerate scale grids are normalized at
  construction.** `scale_bins = 0` / inverted / non-finite scale bounds
  were only caught by a `debug_assert`; release builds silently divided
  by a 0/∞/NaN bin width and collapsed every Hough vote into saturated
  bins. `PanakoMatcher::new` and `PanakoIndex::query` now substitute
  the default scale grid (0.80–1.25, 24 bins) for the degenerate
  fields, identically in every build.
- **1:N index queries are deterministic under `std`.** `WangIndex`,
  `HaitsmaIndex`, and `PanakoIndex` iterated candidate references in
  `ahash` HashMap order, so exact (score, prominence) ties — and which
  perfect-scoring reference wins the early-exit — varied per process
  run. Candidates are now visited in ascending reference id.
- **`WangIndex` prominence matches `WangMatcher`'s dense-range
  semantics.** The index divided the background by *occupied* offset
  bins while the 1:1 matcher's dense histogram dilutes it with the
  empty bins in between — systematically understating index prominence
  and rejecting matches the matcher accepts on wide, sparse vote
  spreads. The denominator is now the vote-offset span width.
- **`HaitsmaIndex` verifies up to 8 candidate offsets per reference**
  (most-hit first, BER-normalized bounds) instead of only the single
  most-hit delta. A repeated motif concentrating exact LUT hits at a
  wrong offset no longer masks the true, low-BER alignment.
- **`WangMatcher` histogram hardening.** The offset-range computation
  now stays in `u64` until after the cap (a >4-Gi-bin span used to
  truncate through `as usize` on 32-bit targets and fold distant
  offsets together), and the ±tolerance consolidation uses an O(1)
  sliding window instead of a transient `u64` prefix array that
  tripled peak memory on adversarial inputs.
- **Decoder: mid-stream `ResetRequired` is recovered.** A codec
  parameter change (AAC config update, track switch) mid-file used to
  fail the whole decode; the decoder is now reset and the packet
  retried once before giving up. The f32 conversion buffer is also
  reallocated when the stream's audio spec (rate/channel layout)
  changes mid-file instead of reusing a stale layout.
- **`decode_to_mono_at_limited` enforces `max_samples` on the returned
  buffer.** The limit bounded native-rate decoding, so an upsample
  could legally return up to `target_sr/native_sr`× the cap; the
  output is now re-checked after resampling.
- **`max_push_samples: Some(0)` is bumped to `Some(1)`** by the
  classical config sanitizer, matching the other zero-value limits —
  `Some(0)` silently truncated every streaming push to empty. The
  `max_pending_anchors` docs now state explicitly that evicted anchors
  and their hashes are **lost**, not deferred.
- **CRITICAL: streaming `flush` is now idempotent and `push` after `push` after
  `flush` no longer duplicates hashes.**
  `IncrementalPeakDetector::flush` kept no record of rows it had
  already drained, so a second `flush()` (or a `push()` after a
  mid-stream `flush()`, both legal per the
  `StreamingFingerprinter::flush` lifecycle contract) re-emitted the
  last `neighborhood_t` spectrogram rows, re-created their buckets,
  and re-emitted their hashes — silently inflating vote counts and
  corrupting downstream matching for any consumer with retry logic or
  a push-flush-push cycle. The detector now tracks a `last_emitted`
  row cursor: `push_row` never re-ripens rows a flush already drained,
  and a second `flush` emits nothing. `StreamingWang` and
  `StreamingPanako` inherit the fix through the shared stream core;
  `StreamingHaitsma` and the neural streamer were already idempotent.
  Pinned by new detector-level (idempotency, no-re-emission,
  reset-restarts-cursor) and stream-level (double-flush, no
  pre-flush anchors after continuation) tests.
- **Codec features pull their companion decoders (#0.4.0 audit).**
  `std-mp4` enabled only the isomp4 *demuxer*, so real M4A files parsed
  but failed to decode — it now also enables `symphonia/aac`. Likewise
  `std-aiff` gains `symphonia/pcm` (AIFF payloads are PCM) and
  `std-alac` gains `symphonia/isomp4` (ALAC ships inside MP4/M4A).
- **`WatermarkDetector` runs an optimised tract plan** — the cached
  per-length plan skipped `into_optimized()` (the neural embedder never
  did), leaving watermark inference on unoptimised graphs. Same outputs,
  markedly faster repeated-`detect` inference.
- **`StreamingNeuralEmbedder` large-push cost is linear, not quadratic**
  — the per-window front `drain(..hop)` memmoved the entire remaining
  carry on every emitted embedding (an hours-long single push
  approached hundreds of GB of memmove). A read cursor now advances per
  window and compacts once per `push` call. Output is bit-identical
  (pinned by the chunk-size-invariance and offline-equivalence tests).
- **`from_bytes` rejects corrupt frame rates** — a malicious/corrupt
  blob could inject NaN / negative / zero `frames_per_sec`, which then
  propagated silently into downstream offset math. The header frame
  rate must now be finite and positive.
- **`from_bytes` payload sizing is overflow-safe on 32-bit** —
  `hash_count * size_of::<T>()` is now a checked multiply, so a
  truncated blob can no longer wrap the expected length and pass
  validation.
- **`NeuralEmbedderConfig` upper bounds** — extreme-but-finite
  `window_secs` / `n_mels` / `n_fft` values (e.g. `window_secs = 1e30`)
  previously passed validation and aborted on the OOM/overflow of the
  front-end allocation; they now fail fast with `AfpError::Config`
  (`window_secs ≤ 3600`, `n_mels ≤ 8192`, `n_fft ≤ 2²⁰`, and
  `n_mels × n_frames ≤ 2²⁸` cells).
- **Cheap checks before the O(n) finiteness scan** — the neural
  embedder and watermark detector now reject wrong-rate / oversized
  input before scanning every sample for NaN.
- **`SincResampler` honours the output-length contract on short inputs
  (0.4.0 audit).** For inputs shorter than the filter width
  (`2 * half_taps + 1`), the left/middle/right output regions could
  overlap, so the boundary loops double-emitted samples and `process`
  returned more than the documented `ceil(n_in * to_sr / from_sr)`
  outputs (e.g. 48 kHz→8 kHz on 10 samples returned 3 instead of 2;
  8 kHz→48 kHz on 10 samples returned 119 instead of 60). The region
  bounds are now clamped into an exact partition of the output range.
  Long-input behaviour is unchanged. Pinned by length-contract and
  bounded-gain regression tests.
- **Decoder rejects a zero sample rate instead of panicking (0.4.0
  audit).** A malformed container reporting `sample_rate = 0` flowed
  into `decode_to_mono_at` and panicked inside
  `SincResampler::new(0, _)`. It now surfaces as an I/O error
  ("sample rate is zero (malformed container)"). Pinned by a crafted
  zero-rate WAV regression test.
- **`decode_to_mono_at_limited` projects the post-resample size before
  allocating (0.4.0 audit).** The post-resample `max_samples` re-check
  above ran *after* the upsampled buffer was allocated, so a hostile
  container claiming a tiny native rate (e.g. 1 Hz → 48 kHz, a
  48,000× blow-up) could OOM the process before any limit fired. The
  projected output length is now checked before resampling; the
  post-resample check remains as a defensive re-verification.
- **Decoder enforces the sample budget before allocating the conversion
  buffer (0.4.0 audit).** The `max_samples` check ran after the f32
  conversion buffer was allocated for the packet, so a malformed packet
  reporting a huge frame count could blow the memory budget regardless
  of the limit. The budget is now checked against the decoded frame
  count first.
- **`FingerprintEnvelope::peek` rejects corrupt frame rates (0.4.0
  audit).** The finite-and-positive frame-rate validation lived only in
  the full-parse path, so `peek` — which stops at the 18-byte header —
  returned garbage metadata for NaN/zero/negative-fps blobs despite its
  documented contract. Validation now lives in the shared header read,
  so `peek` and `from_bytes` enforce it identically.
- **Serialization fails loudly instead of silently truncating huge hash
  counts (0.4.0 audit).** `to_bytes` stores the hash count as a `u32`
  and used to silently clamp larger counts in release builds (only a
  `debug_assert` guarded it), producing a corrupt blob. It now panics
  with an explicit message — unreachable in practice (billions of
  hashes) but no longer silent corruption. Documented in a `# Panics`
  section.
- **Timeout error fields use saturating conversions (0.4.0 audit).**
  `AfpError::Timeout`'s `elapsed_ms` / `limit_ms` are now built with
  saturating `u128 → u64` conversions instead of truncating `as` casts.

## [0.3.9] - 2026-08-02

### Changed

- **Safe SIMD via `wide` crate (replaces all `unsafe` arch intrinsics).**
  Window application (`apply_window_wide`), power-spectrum computation
  (`compute_power_wide`), and dB log-magnitude conversion
  (`power_to_db_wide`) now use `wide::f32x8` — portable safe SIMD that
  auto-dispatches to AVX2/SSE/NEON without any `unsafe` code. Removes
  all `#[target_feature]` functions and `core::arch` imports from
  `stft.rs`. Bit-exact output preserved (golden tests unchanged).
- **Vectorized dB conversion in Wang and Panako (offline + streaming).**
  The per-frame `max(v, floor).log2() * factor` loop is now processed
  8 elements at a time via `f32x8::max()` + `f32x8::log2()`. Both
  offline `extract` and streaming `push` benefit.
- **SIMD dot product in polyphase resampler (`resample.rs`).**
  The 65-tap kernel convolution in the middle (safe) loop uses `f32x8`
  fused multiply-add — 8 iterations + 1 scalar tail instead of 65
  scalar multiply-adds.
- **SIMD dot product in mel filterbank CSR application (`mel.rs`).**
  `log_mel_from_power` accumulates band energies 8-wide via
  `f32x8::mul_add`. Bands with 10-50 non-zero bins now run in 1-6
  SIMD iterations.
- **SIMD band-difference bit-packing in Haitsma (`haitsma.rs`).**
  `pack_frame_bits` computes 32 band differences in 4 `f32x8`
  iterations and extracts sign bits, replacing a 32-iteration scalar
  loop.
- **Folded `inv_dc_gain` into polyphase kernel table (`resample.rs`).**
  All kernel coefficients are pre-multiplied by `1/dc_gain` at
  construction, eliminating one `f32` multiply per output sample in the
  resampling hot loop.
- **Removed dense `matrix` field from `MelFilterBank`.**
  Saves ~256 KB heap per instance. The `matrix()` getter now
  reconstructs on demand from the sparse CSR representation.
  Hot path uses sparse representation exclusively.
- **Pre-sized `VecDeque` in Lemire peak picker.**
  Both `PeakPicker` and `IncrementalPeakDetector` allocate the
  monotonic deque to its maximum capacity at construction, eliminating
  capacity-check branches in the inner rolling-max loop.
- **Crate package size reduced from 7.2 MiB to ~215 KiB.** Excluded
  `tests/assets/` (8 MiB of real audio) and `fuzz/` from the published
  crate. Downstream users don't need test audio.
- **`MelFilterBank::new()` no longer allocates a dense matrix.**
  The sparse CSR representation is now built directly by computing
  analytical bin bounds per mel band (`first_bin`/`last_bin` from Hz
  values), eliminating a temporary `n_mels × n_bins` allocation
  (~512 KB for 128-mel / 2048-FFT configs). Bit-exact output preserved.
- **`PeakPicker::pick()` uses `mem::take` instead of `clone` + `clear`.**
  Eliminates a full Vec clone on every offline peak-pick call.
- **`new()` delegates to `try_new().expect()` in `MelFilterBank`,
  `ShortTimeFFT`, and `SincResampler`.**  Validation logic lives in one
  place per type; `new()` is a one-liner. No API change — same panics
  on invalid config.
- **Extracted `l2_normalize_inplace` helper in neural embedder.**
  Deduplicates the L2-norm logic that appeared in both
  `embed_window_into` and `embed_batch_into`.
- **Shared `map_model_open_io` / `map_model_load_err` helpers in
  `error.rs`.**  Eliminates duplicate model-loading error mappers
  between `neural/embedder.rs` and `watermark/detector.rs`.
- **Streaming `push()` / `flush()` use `mem::take` instead of
  `Vec::new()` + `append`.**  All three streaming fingerprinters
  (Wang, Panako, Haitsma) now return the internal buffer with O(1)
  pointer swap rather than allocating + copying.
- **Removed redundant `.clear()` after `.drain(..)` in Wang/Panako
  streaming `push_with` / `flush_with`.**
- **Removed trivial SIMD wrappers** (`dot_mel_wide`, `dot_mel_sq_wide`,
  `dot_f32_wide`) — call sites now invoke `dsp::dot_wide` /
  `dsp::dot_sq_wide` directly.

### Added

- **In-memory matching subsystem** (`audiofp::matching`) — `WangMatcher`,
  `HaitsmaMatcher`, `NeuralMatcher` (feature-gated), plus `match_best` /
  `match_ranked` and a transient `WangIndex` for 1:N Wang queries. Purely
  in-memory: no persistence, wire format, or DB adapters. See `USAGE.md`.
- **`benches/matching.rs`** — Criterion benches for Wang/Haitsma 1:1 and
  small `WangIndex` (N=100) queries (`cargo bench --bench matching`).
- **`wide` 1.5 dependency** (no_std compatible, zero `unsafe`). Provides
  `f32x8` used across the DSP pipeline.
- **Lightweight fingerprint serialization (`serial` module) (#117).**
  `WangFingerprint::to_bytes()` / `from_bytes()`,
  `PanakoFingerprint::to_bytes()` / `from_bytes()`,
  `HaitsmaFingerprint::to_bytes()` / `from_bytes()` — compact binary
  wire format with 8-byte magic, version byte, algorithm ID, and Pod
  hash payload. `FingerprintEnvelope` struct for metadata inspection.
  New `AfpError::Deserialize` variant for format errors.
- **Extraction progress callback (#122).**
  `Wang::extract_with_progress`, `Panako::extract_with_progress`,
  `Haitsma::extract_with_progress` — callback receives monotonic
  `f32` in `[0.0, 1.0]`. Called every ~500 ms of audio. Plain
  `extract()` delegates with a no-op closure (no overhead).
- **Decoder integrity mode (#76).**
  `DecodeLimits::strict()` builder sets `integrity_mode: true` — any
  per-packet decode error (corrupt frame, I/O glitch) becomes fatal
  instead of being silently skipped. Default remains `false`
  (backwards-compatible skip behavior).
- **Fuzz harnesses for decoder wrappers (#84).**
  `fuzz/fuzz_targets/decode_bytes.rs` (raw bytes → decode pipeline),
  `fuzz/fuzz_targets/decode_resample.rs` (arbitrary bytes + target
  sample rate → decode + resample). Both wired into CI fuzz smoke job
  (now 9 targets total).
- **macOS and Windows in CI test matrix (#29).**
  `test` and `no-std-test` jobs now run on `ubuntu-latest`,
  `macos-latest`, and `windows-latest` with `fail-fast: false`.
- **Codec robustness documentation (#87).**
  `ROBUSTNESS.md` documents the CC-BY corpus, overlap methodology,
  published numbers, and local reproduction commands.
  `scripts/codec_robustness.sh` helper to run and format results.
- **SIMD `dot_sq_wide` for `MelFilterBank::log_mel` (#97).**
  Added `dot_sq_wide(a, b)` to `dsp/mod.rs` — computes `sum(a[i] *
  b[i]²)` via `f32x8` (square + fused multiply-add, 8 elements per
  iteration). `log_mel` (magnitude input) now runs through this
  vectorised path instead of a scalar loop. Bit-exact output preserved
  (verified by the existing `log_mel_from_power_matches_log_mel` test).
- **Batched neural ONNX inference (`batch_size`) (#96).**
  `NeuralEmbedderConfig` gains `batch_size: usize` (default 1 —
  existing behaviour unchanged). When `batch_size > 1`, offline
  `extract()` fills a `[batch, n_mels, n_frames]` tensor and invokes
  the ONNX runtime once per batch, amortising per-run overhead across
  multiple windows. A dedicated `batch_runnable` plan is built at
  construction for the configured batch size; streaming continues to
  use the single-window plan. Falls back to single-window inference
  for partial tail batches. Verified parity with single-window output
  at 1e-6 tolerance.

### Removed

- **All `unsafe` SIMD code** (`apply_window_avx2`, `apply_window_neon`,
  `compute_power_avx2`) — replaced by safe `wide`-based equivalents.

### Performance

Offline extract (`cargo bench --bench extract`, 30 s synthetic audio,
Intel i5-1135G7):

| Algorithm | v0.3.8 (before) | This release | Improvement |
|-----------|----------------:|-------------:|:-----------:|
| Wang      | 99 ms           | 73 ms        | **-26%**    |
| Panako    | 104 ms          | 77 ms        | **-26%**    |
| Haitsma   | 47 ms           | 41 ms        | **-13%**    |

Streaming push (small-chunk benchmarks):

| Path            | Improvement vs v0.3.8 |
|-----------------|-----------------------|
| Wang (small)    | **-8%**               |
| Wang (large)    | **-32%**              |
| Panako (small)  | **-9%**               |
| Haitsma (small) | **-7%**               |
| Haitsma (large) | **-21%**              |

Neural front-end (`cargo bench --features neural --bench neural_frontend`):

| Path                          | Improvement |
|-------------------------------|:-----------:|
| `log_mel_pipeline_1s_window`  | **~-9%**    |

The `dot_sq_wide` SIMD path in `log_mel` eliminates per-element scalar
squaring + accumulation in favour of 8-wide fused multiply-add. Batched
inference (`batch_size > 1`) additionally amortises ONNX runtime
per-run overhead — improvement factor depends on model size and batch
depth (measured 2-4× on small embedding models).

## [0.3.8] - 2026-07-24

### Added

- **`try_new()` on `MelFilterBank`, `ShortTimeFFT`, `SincResampler` (#86).**
  Fallible constructors returning `Result<Self, AfpError::Config>` for
  callers who prefer error handling over panics. The existing `new()`
  methods remain unchanged (panic on invalid params).
- **`WatermarkConfig::max_input_samples` (#81).** When set, `detect()`
  rejects inputs exceeding the cap with `AfpError::InputTooLarge` before
  any inference. Default `None` (unbounded). Matches the pattern on
  classical fingerprinter configs.
- **Expanded Symphonia codec/format support (#112).** Enabled `adpcm`,
  `alac` (Apple Lossless), and `mkv` (Matroska/WebM) features. Files in
  these formats now decode out-of-the-box without user feature-flag work.
- **`SECURITY.md` + threat model (#74).** Documents responsible disclosure,
  trust boundaries (untrusted audio vs trusted ONNX), and production
  defaults (`DecodeLimits`). Linked from README.
- **`CODE_OF_CONDUCT.md`, issue/PR templates, CONTRIBUTING MSRV 1.93 (#83).**
- **`AfpError::NonFiniteSample` + PCM policy (#75).** Offline `extract` /
  watermark `detect` reject NaN/Inf. Streaming `push` sanitizes them to
  `0.0` (infallible API). Helper module `pcm`.
- **`max_push_samples` on Wang, Haitsma, and neural streaming (#80).**
  Matches Panako: truncate hostile `push` chunks when set.
- **OOM protection — `max_input_samples` on all offline fingerprinter configs (#68).**
  `WangConfig`, `PanakoConfig`, and `HaitsmaConfig` each gain a
  `max_input_samples: Option<usize>` field. When set, `extract()` rejects
  inputs larger than the cap with `AfpError::InputTooLarge` before any
  allocation. Safe defaults (30 min at each algorithm's required rate)
  ship out of the box; pass `None` to disable. `NeuralEmbedderConfig` also
  exposes the field (default `None` for BYO-model scenarios where the
  caller knows the model's limits). A 4 GB malicious upload now returns
  `InputTooLarge` instead of OOM.
- **OOM protection — decoder file-size and PCM caps.**
  `decode_to_mono_limited` / `decode_to_mono_at_limited` accept a
  `DecodeLimits` struct with `max_bytes` (rejects on-disk size before
  opening the stream) and `max_samples` (bounds decoded mono PCM so
  compressed formats cannot inflate past a sample budget).
  `DecodeLimits::bytes(n)`, `DecodeLimits::samples(n)`, and
  `DecodeLimits::both(bytes, samples)` cover the common cases.
  The base `decode_to_mono` / `decode_to_mono_at` remain unlimited
  (`DecodeLimits::default()`). Closes #68.
- **`AfpError::InputTooLarge` error variant.** Structured error reporting
  the configured limit and the actual input size (samples, bytes, or
  hashes depending on the check). `Display` text includes both numbers
  and a hint about raising the limit.
- **`audiofp::prelude` module (#14).** Convenience glob import
  (`use audiofp::prelude::*`) that pulls in all three classical
  fingerprinters with their config/hash types, both core traits,
  error types, and value types. Includes a doc-test showing the
  shortest path from zero to a fingerprint.
- **`StreamingFingerprinter::required_sample_rate()`.**  Callers can now
  query the expected sample rate at construction rather than feeding
  wrong-rate samples silently. Implemented for all four streaming types.

### Changed

- **`Haitsma::try_new()` and `StreamingHaitsma::try_new()`.**
  Fallible constructors returning `Result<Self, AfpError>` on invalid
  `fmin`/`fmax`/Nyquist config. The existing `new()` remains infallible
  (panics on bad config) for backward compatibility.
- **`#[non_exhaustive]` removed from config and fingerprint structs.**
  Originally added in this cycle but reverted — it breaks the documented
  `WangConfig { fan_out: 5, ..Default::default() }` pattern for external
  crates. Deferred to 0.4.0.
- **`max_pending_anchors` doc recommends `Some(10_000)` for untrusted
  input.** Default remains `None` (unbounded) for backward compatibility.
- **`target_zone_f` clamped to `[1, 512]`; `min_anchor_mag_db` clamped to
  `[-200.0, 0.0]`** in all Wang/Panako constructors.
- **Zero-value `Option` limits** (`max_input_samples`, `max_hashes`,
  `max_pending_anchors` = `Some(0)`) are silently clamped to `Some(1)`.
- **`native-tls` banned in `deny.toml`** (pulls openssl transitively).

### Changed

- **Matching hot path uses `HashMap` under `std`** (default). Without
  `std`, the same code paths fall back to `BTreeMap` via an internal
  alias so `no_std + alloc` builds keep working.
- **`PanakoMatcher` / `PanakoIndex` documented as stubs** — they always
  return non-match / empty until Phase 3 (2-D Hough) lands. Prefer
  `WangMatcher` for constant-tempo identification today.
- **UB in `PeakPicker` scratch buffers.** `prepare_vec_uninit` used
  `unsafe { v.set_len(new_len) }` leaving uninitialized `f32`s that
  could be read before overwrite under aggressive LLVM optimizations.
  Replaced with safe `clear()` + `resize()`.
- **`decode_to_mono` Security docstring.** Now warns callers about
  decompression bombs and directs to `decode_to_mono_limited` for
  untrusted input.
- **Model load TOCTOU (#79).** Dropped `path.exists()` before Tract load;
  missing files map to `ModelNotFound`, other failures to `ModelLoad`.
- **SIMD window length asserts (#82).** `debug_assert_eq!` on AVX2/NEON
  window helpers. `BufferOverrun` kept for future mic pipeline (#98).
- **`decode_to_mono_limited` returned `Config` instead of `InputTooLarge`.**
  Oversized files now match the documented `InputTooLarge` variant.
- **`PanakoConfig::max_pending_anchors` was dead.** Streaming Panako now
  evicts oldest-first, matching Wang.
- **`PanakoConfig::max_push_samples` was dead.** Streaming Panako now
  truncates hostile `push` chunks to the configured cap.
- **Panako `target_zone_t == 0` underflow** (P0): Setting `PanakoConfig::target_zone_t` to 0 caused `u32::MAX`-byte allocation via `saturating_add(u32::MAX - 1)`, guaranteed OOM. All four constructors (`Wang`, `StreamingWang`, `Panako`, `StreamingPanako`) now clamp `target_zone_t ∈ [1, 512]` and `fan_out ∈ [1, 64]`. `peaks_per_sec` is also capped at ≤ 500 to bound per-second allocation. Extreme config values (e.g. `u16::MAX`) are silently clamped rather than panicking — fully backward-compatible for all values within the default range.
- **Decoder `n_chans == 0` guard**: Corrupt packets reporting 0 channels previously caused division-by-zero producing NaN/Inf PCM samples. Malformed packets are now silently skipped (same policy as recoverable decode errors).
- **Streaming `push_with` / `flush_with` zero-alloc overrides**: All three classical streaming fingerprinters (`StreamingWang`, `StreamingPanako`, `StreamingHaitsma`) now override the default trait implementations with genuine zero-allocation callback loops. Previously fell back to the trait default which allocates a `Vec` on every call, contradicting the documented contract.
- **Streaming `push()` / `flush()` pooled Vec retention**: `push()` and `flush()` now use `out.append(&mut self.emitted)` instead of `core::mem::take(&mut self.emitted)` followed by `extend(drain(..))`. The pooled allocation stays with the struct across calls — no per-push reallocation. Measured: Wang extract 2 s −13.8 % wall time, streaming Wang small-chunk −29.5 % (cumulative vs pre-pool baseline).
- **`reset()` on all three streaming types**: `StreamingWang`, `StreamingPanako`, and `StreamingHaitsma` each expose a `reset()` method that clears all internal state (buffered audio, pending peaks/anchors, frame counter). Reusing a single instance across independent streams now works correctly without stale data bleed.

### Performance

- **Optimize PeakPicker scratch buffers**: Replaced zero-filling resizes of reused vectors (`max_buf`, `temp_2d`, `col_in`, `col_out`) with conditional `set_len` when capacity is already sufficient, saving millions of writes on every extraction.
- **Avoid heap allocations in Haitsma**: Introduced a reused `power_buf` buffer within `Haitsma` to eliminate `Vec<f32>` allocations in `stft.power_flat` on every extraction.
- **Optimize suffix_max in Panako**: Replaced full zero-filling resizes of `suffix_max` in triplet generation with a conditional check, truncating and zeroing only the last element.

### Testing

- **21 new tests**: Constructor clamping (defaults preserved, zero clamped to 1, extreme capped to safe bounds, clamped config still produces hashes), streaming reset+replay correctness (×3), `push_with` ≡ `push`+`flush` output parity (×6), and `flush_with` ≡ `flush` parity (×3). Test count: 227 → 280 (`cargo test --all-features --tests`).
- **Real audio golden regression tests (×6).** Byte-exact hash snapshots
  for Wang, Panako, and Haitsma on `piano.ogg` and `speech.ogg`. Any
  code change that alters hash output on real audio breaks these.
- **E2E real audio tests (×17):** segment/offset matching (hash subset
  verification), gain invariance (0.1× quiet, 3× clipped), 10×
  determinism (all 3 algorithms), Panako ±5% time-stretch robustness,
  decoder edge cases (short audio, empty file, corrupt header,
  `DecodeLimits` byte/sample caps, stereo/5.1 multichannel downmix,
  odd sample rates 11025/22050/44100 Hz).
- Test count: 280 → 303 (`cargo test --all-features --tests`).
- **Codec round-trip robustness tests (×12).** Same music ("Galway" by
  Kevin MacLeod, CC-BY 3.0) encoded as FLAC, MP3, OGG-Vorbis, AAC/M4A,
  and WAV. Asserts minimum Jaccard / bit-similarity thresholds vs. the
  FLAC lossless reference for all three algorithms. Test assets committed
  to `tests/assets/galway.*`.

### Documentation

- **Zero-deps README quick start** before the file-decode example (#38).
- **Pinned performance numbers** to v0.2.0 / v0.3.4 with a YMMV note and
  streaming bench pointer (#51).
- **USAGE.md**: wrap copy-paste snippets in `fn main()`, replace undefined
  `whole_song` / `audio_capture_iter` placeholders (#55); add async /
  batching / watermark-model download guidance (#50); document
  `WatermarkResult::localization` contract (#59).
- **Examples**: `dsp_starter`, `neural_embed` (`neural`),
  `watermark_detect` (`watermark`) (#39).

## [0.3.7] - 2026-07-08

### Changed

- **MSRV raised to 1.93.0** due to tract-onnx 0.23 requiring Rust ≥ 1.91.

- **Bump `symphonia` 0.5.5 → 0.6.0.** Symphonia 0.6 restructured its
  audio, format-reader, and codec APIs. The decoder module has been
  rewritten to use the new `Audio` trait, `GenericAudioBufferRef`,
  `Probe::probe()`, and `FormatReader::default_track()` entry points.
  No public API change; `decode_to_mono` / `decode_to_mono_at` continue
  to return the same results. Closes #49 (partial).

- **Bump `tract-onnx` 0.22.1 → 0.23.3.** Updates to
  `TypedSimplePlan` (two-param `SimplePlan`), `to_plain_array_view`,
  and `as_slice_mut_unchecked`. Resolves RUSTSEC-2026-0009 transitive
  advisory via updated `time` dep chain. Closes #49.

- **`AfpError::Io` is now a structured `IoError` (std only).** The
  `Io` variant carries `IoError { path: Option<PathBuf>, kind:
  ErrorKind, source: io::Error }` instead of a bare `String`. Users
  can now inspect `kind` for retry decisions, pattern-match on the
  path, and access the `#[source]` chain. A `From<std::io::Error>` impl
  is provided for ergonomic `?` propagation. The `no_std` variant
  remains `Io(String)`. Closes #18.

  **Migration:** if you pattern-match on `AfpError::Io(msg)` where `msg`
  was a `String`, change to `AfpError::Io(e)` — `e` is now an `IoError`
  which implements `Display` (so `format!("{e}")` still works).
  `AfpError::Io(_)` patterns compile unchanged.

### Added

- **`fingerprint_batch_parallel`** (gated on the `parallel` feature) —
  rayon-backed batch extraction that runs `Fingerprinter::extract` in
  parallel across multiple CPU cores. Each item gets its own fingerprinter
  instance via a `Fn() -> F` factory closure, avoiding `&mut self`
  contention. Results are returned in input order. Enable with
  `--features parallel`; `rayon` is an optional dependency, so the core
  crate remains zero-extra-dependency by default. A doc-test and unit
  test verify that parallel output matches sequential output for 100
  items.

### Performance

- **SIMD `fill_windowed` window-application inner loop (closes #21).**
  The per-frame `dst[i] = src[i] * win[i]` multiply (executed once per
  STFT frame, ~320K multiplies for 5 s of Wang audio) now dispatches to
  an explicit AVX2 path on x86-64 (`f32x8` packed muls via
  `_mm256_mul_ps`) and a NEON path on aarch64 (`vld1q_f32` /
  `vmulq_f32`). Runtime detection via `is_x86_feature_detected!`
  (std-only; no_std falls back to scalar). Scalar tail-handling for
  `n_fft % 8 != 0`.

  Measured on Intel i5-1135G7 (vs. previous iteration):

  | Benchmark            | Before   | After    | Δ            |
  | -------------------- | -------- | -------- | ------------ |
  | Wang extract 30 s    | 91.3 ms  | 81.0 ms  | **−11.3 %** |
  | Panako extract 30 s  | 101.0 ms | 84.2 ms  | **−16.6 %** |
  | Haitsma extract 30 s | 37.0 ms  | 39.1 ms  | noise        |
  | Wang extract 5 s     | 11.8 ms  | 11.3 ms  | **−4.2 %**  |
  | Panako extract 5 s   | 14.4 ms  | 11.6 ms  | **−19.4 %** |

- **Mel filterbank: `log10f` → `LOG10_2 * log2f`.** Three hot-path
  `log10f` calls (`hz_to_mel` HTK branch, `log_mel`, `log_mel_from_power`)
  replaced with `core::f32::consts::LOG10_2 * log2f(…)`. `log2f` lowers to
  `fyl2x` on x86-64, avoiding the ~30-cycle `log10f` software
  implementation. Measured: Panako extract 2 s 4.38 ms → 4.38 ms (within
  noise at small sizes), but compounding across the full mel matrix and all
  three fingerprinters produces measurable end-to-end gains.

- **`SLANEY_LOGSTEP` precomputed as `const`.** The Slaney mel scale's
  `logf(6.4) / 27` was evaluated at runtime on every `hz_to_mel` /
  `mel_to_hz` call. Now a compile-time constant `0.068_751_97`, saving a
  transcendental + division in the Slaney hot path.

- **Wang `detect_rows_range` eliminates per-row `Vec<Peak>`.** Each row
  previously allocated a local `Vec<Peak>` and batch-extended into
  `bucket_pending`. Now pushes peaks directly into the bucket's
  `Vec<Peak>` via `entry(bucket).or_default().push()`, matching the
  pattern already used by `StreamingPanako::detect_rows`. Removes one
  heap allocation per spectrogram row.

- **Panako `pack_triplet` eliminates `libm::roundf`.** The Slaney-tempo
  `β` quantiser called `libm::roundf(dt_bc / dt_ac * 31.0)` per
  triplet — a software-float round invoked hundreds of times per anchor.
  Since both `dt_bc` and `dt_ac` are always ≥ 0, a fast integer cast
  `(x + 0.5) as i32` is equivalent for positive values, saving ~15
  cycles per triplet on softfloat platforms. Also removed the now-unused
  `use libm::roundf;` import.

- **`PeakPicker::pick` pools the candidates buffer.** The intermediate
  `Vec<Peak>` used to collect local-maxima before adaptive thresholding
  was freshly allocated on every `pick()` call. Now stored as a struct
  field (`candidates`) and reused across calls, matching the existing
  pooling pattern for `max_buf`, `temp_2d`, `col_in`, `col_out`, and `dq`.

- **Haitsma offline `extract` pools energy and frame buffers.** The
  offline `extract` method allocated a fresh `energies` (`Vec<[f32; 33]>`)
  and `frames` (`Vec<u32>`) on every call — two heap allocations per
  extraction. Now stored as struct fields (`energies_buf`, `frames_buf`)
  and reused across calls via `.clear()` + `.push()`, returning ownership
  to the caller via `core::mem::take`. This is especially beneficial for
  batch-extraction workloads that call `extract` repeatedly on short
  clips.

  `cargo bench` results (cumulative, vs published 0.3.6):

  | Benchmark              | 0.3.6   | Current | Δ            |
  | ---------------------- | ------- | ------- | ------------ |
  | Wang extract 30 s      | 88.2 ms | 82.7 ms | **−6.2 %**  |
  | Panako extract 30 s    | 90.7 ms | 85.9 ms | **−5.3 %**  |
  | Haitsma extract 30 s   | 44.7 ms | 36.9 ms | **−17.4 %** |
  | Wang extract 2 s       | 4.5 ms  | 4.2 ms  | **−6.7 %**  |
  | Panako extract 2 s     | 5.1 ms  | 4.3 ms  | **−15.7 %** |
  | Haitsma extract 2 s    | 2.4 ms  | 1.95 ms | **−18.8 %** |
  | Wang streaming 256     | 11.9 ms | 11.3 ms | **−5.0 %**  |
  | Panako streaming 256   | 13.0 ms | 12.1 ms | **−6.9 %**  |
  | Haitsma streaming 256  | 8.9 ms  | 6.4 ms  | **−28.1 %** |
  | Wang streaming 1 s     | 11.3 ms | 11.3 ms | noise        |
  | Panako streaming 1 s   | 12.9 ms | 11.8 ms | **−8.5 %**  |
  | Haitsma streaming 1 s  | 8.4 ms  | 6.4 ms  | **−23.8 %** |

- **STFT: eliminate per-frame FFT scratch allocation.** `ShortTimeFFT`
  now stores a dedicated `fft_scratch: Vec<Complex<f32>>` and uses
  `realfft`'s `process_with_scratch()` instead of `process()`. The latter
  allocates temporary scratch on every call; the new path reuses the
  pre-allocated buffer. Eliminates ~1875 hidden heap allocations for a
  30 s clip at 62.5 fps.

- **STFT: vectorization-friendly norm_sqr.** All power-spectrum loops
  replaced with the canonical `zip().for_each(|o, c| *o = c.re * c.re +
  c.im * c.im)` pattern that LLVM auto-vectorizes reliably, instead of
  `enumerate` + method call on `Complex::norm_sqr()`.

- **Resampler: multiply by precomputed reciprocal.** `SincResampler`
  now stores `inv_dc_gain = 1.0 / dc_gain` and multiplies per output
  sample instead of dividing. Float division is ~4× slower than
  multiplication on modern CPUs.

- **Resampler: zip+sum inner loop.** The safe middle loop uses the
  canonical `slice.iter().zip(kernel).map(|(&s,&k)| s*k).sum()` pattern
  for optimal auto-vectorization by LLVM (FMA on AVX2).

- **Watermark detector: cache `Runnable` plan.** `WatermarkDetector`
  now caches the Tract `SimplePlan` (runnable) directly. Previously, even
  on a cache-hit (same input length), `detect()` would deep-clone the
  `TypedModel` and rebuild the runnable on every call. Now the runnable is
  reused by reference — only a length change triggers a rebuild.

- **Streaming: pool output Vec.** Both `StreamingWang` and
  `StreamingPanako` now store an `emitted` buffer on the struct and return
  it via `core::mem::take`. Eliminates one `Vec` allocation per `push()`
  call (~62/sec in steady state).

- **Streaming: replace BTreeMap with sorted Vec.** The `bucket_pending`
  field in both streaming fingerprinters used a `BTreeMap<u32, Vec<Peak>>`
  with at most 3 entries. Now a sorted `Vec<(u32, Vec<Peak>)>` with
  linear search — fewer allocations and better cache locality for
  ≤3-element collections.

- **Panako streaming: cap targets at 2×fan_out.** `PendingAnchorPanako`
  targets are now capped at `2 × fan_out` by magnitude. Prevents O(N²)
  pair enumeration for dense audio where an anchor's target zone could
  accumulate 60+ peaks.

- **Panako triplet early-exit via suffix-max array.** The offline triplet
  enumeration builds a suffix-max array over target magnitudes. The outer
  loop skips any `b` where `b.mag + suffix_max[j+1]` can't beat the
  current heap minimum; the inner loop breaks early when the remaining
  `c` candidates can't win. Reduces O(T²) enumeration to near-linear in
  practice for typical peak densities.

- **Haitsma: range-based band accumulation.** Precomputed `band_ranges`
  replaces the per-bin `if b != NO_BAND` branch with contiguous
  `row[start..end].iter().sum()` calls. Enables SIMD auto-vectorization
  on the energy sums.

- **Haitsma: branchless bit packing.** `pack_frame_bits` now uses
  `((diff > 0.0) as u32) << (31 - b)` instead of an `if` branch,
  compiling to `fcmp` + `setg` + `shl` + `or` — branchless and
  vectorizable by LLVM.

- **Peak picker: preserve Vec capacity.** `PeakPicker::pick()` now
  uses `clone()` + `clear()` instead of `core::mem::take()` for the
  candidates buffer, retaining the allocation for the next call.

- **Resampler: replace modulo with `.min(steps - 1)`.** Polyphase
  step selection used `% steps` (integer division) as a safety clamp.
  Now uses `.min(steps - 1)` (branch-predicted comparison) — avoids an
  integer division per output sample.

- **Bessel I₀: f64 accumulation.** `modified_bessel_i0` now accumulates
  the series sum in `f64` and casts back to `f32` at the end. Improves
  precision for large `β` values at negligible cost (construction-only
  path).

- **STFT windowing: iterator pattern.** The fast-path windowing loop
  in `fill_windowed` now uses `iter_mut().zip().for_each()` for more
  reliable autovectorization by LLVM.

- **Neural frontend: `debug_assert` for window length.** Release builds
  no longer pay for a per-window bounds-check assertion that is
  structurally guaranteed by all callers.

- **Deduplicate `DB_LOG2_FACTOR`.** Moved the shared constant to
  `dsp::mod` and imported it in both `wang.rs` and `panako.rs`.

- **Remove dead code.** Removed unused `window_start` computation in
  `IncrementalPeakDetector::push_row`.

- **Inline hints.** Added `#[inline]` to `MelScale::hz_to_mel`,
  `MelScale::mel_to_hz`, and `make_window` for reliable cross-crate
  inlining.

### Fixed

- **`decode_to_mono_at` validates `target_sr > 0`.** Previously, passing
  `target_sr = 0` would panic inside the resampler. Now returns
  `AfpError::Config("target sample rate must be > 0")` immediately.

### Testing

- **Fixed `haitsma_hash_roundtrip` fuzz target.** The assertion
  `u32::from_le_bytes(frame.to_le_bytes()) == frame` was a tautology.
  Now uses `bytemuck::pod_read_unaligned(bytes_of(&frame))` and adds a
  determinism check (same input → same output).

- **Added Panako tempo-robustness test.** Property test stretches audio
  by factors in [0.96, 1.04] and asserts non-zero hash overlap,
  validating Panako's defining ±5% tempo tolerance.

- **Added watermark integration test suite.** Seven tests exercise the
  public `WatermarkDetector` API: config validation, error variants, and
  model-load failure paths.

## [0.3.6] - 2026-06-30

A performance patch release. Optimisations across the DSP, classical,
and resampling paths, verified deterministic (goldens unchanged). No
public API changes, no hash output changes.

### Added

- **`SincResampler::process_into(input, out)`.** Writes resampled
  output into a caller-owned `&mut Vec<f32>` (cleared and repopulated)
  instead of allocating a new buffer. Same semantics as `process` but
  lets callers in a hot loop reuse the allocation across chunks.

### Performance

- **`[profile.release]` with LTO and `codegen-units = 1`.** No release
  profile existed — the crate shipped with Cargo's defaults
  (`codegen-units = 256`, no LTO). Adding `lto = "fat"` and
  `codegen-units = 1` enables cross-crate inlining of hot-path functions
  (`log10f`, `norm_sqr`, DSP inner loops). `[profile.bench]` inherits the
  same settings so benchmark numbers reflect the production build.
  This is the single highest-leverage change in this release.

- **Wang/Panako front-ends: `10·log10f(x)` → `DB_LOG2_FACTOR·x.log2()`.**
  Each per-frame log-magnitude loop was calling `libm::log10f` (~30
  cycles per call). `f32::log2` lowers to a single `fyl2x` instruction
  on x86-64. The identity `10·log10(x) = (10/log₂10)·log2(x) ≈ 3.0103·log2(x)`
  is exact in real arithmetic; f32 rounding preserves golden-regression
  outputs. Both offline and streaming paths unified via a named constant
  `DB_LOG2_FACTOR` in each classical fingerprinter. Affects Wang, Panako,
  and Haitsma. Measured: Wang extract 5 s 12.2 ms → 11.3 ms (**−7.6 %**),
  30 s 89.9 ms → 81.0 ms (**−9.9 %**); Panako extract 2 s 4.95 ms →
  4.41 ms (**−10.9 %**); Haitsma extract 30 s 45.8 ms → 41.9 ms
  (**−8.4 %**). Streaming push also benefits: Panako large-chunk
  13.1 ms → 11.7 ms (**−10.4 %**), Haitsma small-chunk 8.6 ms →
  7.2 ms (**−16.3 %**).

- **Wang `build_hashes` uses linear-insert top-K instead of BinaryHeap.**
  For the default `fan_out = 10` (≤ 16 in practice), maintaining a
  sorted `Vec<Peak>` via `partition_point` + `insert/pop` has lower
  constant factors than `BinaryHeap` + drain + re-sort. Combined with
  the existing `partition_point` zone-bound pre-slice, the inner loop
  goes from O(total_peaks) to O(K · peaks_in_zone).

- **Panako `build_triplet_hashes` uses binary search for zone bounds.**
  `partition_point()` pre-slices the target list, reducing the
  O(|targets|²) pair-enumeration scope. Measured: Panako extract 2 s
  4.95 ms → 4.41 ms (**−10.9 %**).

- **`PeakPicker::pick` hoists config reads outside the frame×bin loop.**
  `self.cfg.min_magnitude` and `self.cfg.target_per_sec` are read once
  per call instead of per candidate cell. `candidates` is pre-allocated
  to an upper bound (`ceil(fps) × target_per_sec × 2`) so the hot loop
  never resizes. The explicit `_pad: 0` field assignment in every
  `Peak` constructor is replaced with `..Peak::zeroed()`, saving one
  unnecessary store instruction in the inner loop.

- **`rolling_max_2d_pooled` column pass uses block-tiling.** Each
  column read was striding `n_cols` apart — a separate cache line per
  row, thrashing L1 on large spectrograms (e.g. 1024-bin Haitsma).
  Now processes columns in 64-wide tiles, keeping `temp` data
  L1-resident across each tile.

- **`StreamingWang::detect_rows_range` hoists dB threshold and bunches
  bucket entries.** `self.cfg.min_anchor_mag_db` read once per call;
  `peak_row_max` sliced once per row. Peaks are accumulated into a
  local `Vec` and batch-extended into `bucket_pending` via
  `.extend()` — one hash-table entry per row instead of per peak.

- **`SincResampler` splits boundary and middle regions.**
  Previously checked `if idx < 0 || idx >= n_in` on every output sample.
  Now precomputes the safe middle range where the kernel is fully inside
  the input and iterates it without bounds checks. The boundary regions
  (first and last ~`half_taps / ratio` samples) retain the check.
  Middle region covers >95 % of output samples for typical resampling
  ratios.

- **`#[inline]` on hot-path functions.** Added to `pack_frame_bits`
  (Haitsma), `pack_triplet` (Panako), and `rolling_max_1d` (peaks).
  These are called in tight loops across module boundaries; `#[inline]`
  ensures cross-module inlining even without LTO.

### Fixed

- **Streaming Wang/Panako allocation-elimination gains are now measurable.**
  The 0.3.5 streaming benchmarks showed ~1 % gains from pooled
  `to_finalize` buffers — within noise. With LTO enabled, the same
  code shows **4–16 % streaming throughput improvement** (Wang
  small-chunk: 12.1 ms → 11.7 ms, −4 %; Panako large-chunk:
  13.1 ms → 11.7 ms, −10.4 %), confirming the allocation elimination
  was worthwhile once the compiler can inline the surrounding code.

## [0.3.5] - 2026-06-22

A correctness, ergonomics, and doc-completeness patch release. Public
API surface gains one constructor (`AudioBuffer::new`), two promoted
DSP primitives (`IncrementalPeakDetector`, `rolling_max_2d_pooled`),
and `#[must_use]` on the two extraction trait methods. No hash output,
no error contract, and no audio-format support changes.

### Performance

- **`StreamingWang` and `StreamingPanako` no longer allocate a
  `Vec<u32>` on every `push` for bucket finalisation.**
  `finalize_buckets` previously called `self.bucket_pending.keys()
  .filter(…).cloned().collect()` — a fresh heap allocation on every
  `push` (called every ~256 samples in the streaming benchmark). Both
  structs now own a pooled `to_finalize: Vec<u32>` field, `clear()`ed
  and reused per call. The `flush` path uses the same pooled buffer.
  The `bucket_pending` map is bounded to ≤ 3 entries in steady state
  (pinned by `streaming_state_stays_bounded_under_long_input`), so
  the pooled buffer never grows after warmup. An index-based loop
  (rather than `drain(..)`) sidesteps the borrow conflict where
  `drain` would hold `&mut self.to_finalize` across the
  `self.finalize_bucket` call. Matches the 0.3.3 pooled-`VecDeque`
  and 0.3.2 pooled-embedding-scratch patterns. (Issue #26.)

  `cargo bench --bench streaming` A/B (50 samples, 5 s measurement,
  5 s synthetic input, default config):

  | Benchmark              | Before    | After     | Δ          |
  | ---------------------- | --------- | --------- | ---------- |
  | Wang (256-chunk)       | 12.808 ms | 12.694 ms | **−0.89%** |
  | Wang (1 s-chunk)       | 12.684 ms | 12.661 ms | noise      |
  | Panako (256-chunk)     | 13.452 ms | 13.363 ms | **−0.66%** |
  | Panako (1 s-chunk)     | 13.459 ms | 13.347 ms | **−0.83%** |
  | Haitsma (256-chunk)    | 7.259 ms  | 7.280 ms  | noise      |
  | Haitsma (1 s-chunk)    | 7.222 ms  | 7.252 ms  | noise      |

  The improvement is modest (~1 % on the small-chunk streaming path
  where `finalize_buckets` is called most often) — the allocation
  was real (one `Vec<u32>` per push) but small. For realtime audio
  callbacks where allocator jitter matters more than steady-state
  throughput, eliminating a per-push alloc is the right call
  regardless of the bench delta. Haitsma is unaffected (it doesn't
  use `bucket_pending` / `finalize_buckets`).

### Fixed

- **`StreamingNeuralEmbedder::push` panic is now documented at the
  crate level.** The `StreamingFingerprinter` trait declares
  `push` as infallible, but the neural streaming implementation
  panics on ONNX inference errors. Added a `Panics in streaming
  APIs` section to the crate-level docs (`src/lib.rs`) pointing
  at [`neural::StreamingNeuralEmbedder::try_push`] for callers
  that need to surface inference failures (audio callbacks,
  `tokio::spawn` workers, etc.). Classical streaming fingerprinters
  (Wang / Panako / Haitsma) never panic on valid input. (Issue #6.)

  **Note on issue #7 (hot-path Tensor allocation per window):**
  investigated but **not fixed** in this release. tract 0.22.1's
  `Tensor::clone()` is a deep copy (`self.deep_clone()`), not a
  refcount bump, so the "cache + clone" approach proposed in the
  issue would not save the allocation. Every `Tensor` construction
  path in tract 0.22.1 (`uninitialized`, `from_shape`, `from_raw`,
  `from_slice_align`) allocates a fresh backing buffer. A genuine
  zero-alloc fix requires either a tract API change (e.g.
  `from_raw_vec`) or unsafe interior mutability — deferred until
  tract 0.23 (tracked in issue #49). The misleading comment in
  `embed_window_into` has been corrected to document this
  limitation honestly. (Issue #7.)

### Added

- **`AudioBuffer::new(samples, rate)`.** Constructor that matches
  the field layout by name; equivalent to the existing struct
  literal. Recommended in new code; the literal form is kept for
  backward compatibility. (Issue #16.)

- **Public `IncrementalPeakDetector` and `rolling_max_2d_pooled`.**
  Both were `pub(crate)` but battle-tested, unit-tested, and useful
  as building blocks for any streaming spectrogram pipeline (tempo,
  onset, beat tracking). `IncrementalPeakDetector` was the engine
  behind the 0.3.4 streaming-perf fix; exposing it lets external
  pipelines reuse the amortised O(n_bins) per-row rolling max
  without copy-pasting ~130 lines. (Issue #19.)

- **`#[must_use]` on `Fingerprinter` and `StreamingFingerprinter`.**
  Both `extract` and `push`/`flush` return values that are often
  silently discarded (`fp.extract(buf);`, `s.push(chunk);`) — a
  real bug pattern. The trait-level attribute flags this at
  compile time. (Issue #16.)

### Documentation

- **`Peak::mag` units now documented.** The previous "Magnitude at
  the peak." left callers guessing whether the value was linear
  power, dB, or something else. Now spells out: Wang/Panako feed
  dB (`10·log10(power)`), and `audiofp`'s `PeakPicker` has no
  opinion on units otherwise. (Issue #40.)

- **`MelFilterBank::new` `fmin` precondition documented.** The
  docstring now clarifies that `fmin = 0` is accepted (both Slaney
  and HTK mel scales handle 0 Hz without hitting `log(0)` — Slaney's
  linear branch covers `hz < 1000`, HTK evaluates `log10(1 + 0) = 0`),
  and explains why [`HaitsmaConfig`](crate::classical::HaitsmaConfig)
  independently requires `fmin > 0` (its log-spaced band edges use
  `powf(fmax / fmin, …)`, which is a different code path). An
  `assert!(fmin >= 0)` was added to guard against negative inputs.
  (Issue #40.)

- **`Haitsma` MSB-zero bit-packing divergence from paper is now
  called out in the module docstring.** The paper packs bands in
  natural index order; this implementation uses band 0 → bit 31
  ("MSB-zero"). Stable for `haitsma-v1` hashes, but worth
  documenting explicitly — same shape as the recent
  `panako-v2` docstring fix. (Issue #41.)

- **Wang, Haitsma, and AudioSeal citations now include stable
  identifiers.** Wang (2003) is cited with the canonical
  Columbia PDF URL; Haitsma (2002) is cited with full
  author/title/venue; AudioSeal (San Roman et al., 2024) is
  cited with the arXiv preprint ID. Matches the post-0.3.4
  Panako citation style. (Issue #42.)

- **`push_with` / `flush_with` default-impl allocation behaviour
  is now explicit in the docstring.** The previous wording
  claimed the methods are "zero-allocation"; the default impl
  delegates to `push`/`flush` (which allocate a `Vec`) and is
  not zero-alloc unless the implementor overrides. The
  docstring now says so plainly. (Issue #53.)

## [0.3.4] - 2026-06-14

### Changed

- **Internal: removed all production `.unwrap()` calls.** Six `unwrap()`
  sites in the streaming hot path are replaced with explicit `Option`
  handling:
  - `dsp::peaks::rolling_max_1d` — `dq.front().unwrap()` reads are guarded
    by `if let Some(&front) = dq.front()`. Callers pre-zero the output
    slice, so a missing deque entry leaves the value at 0.0 (the Lemire
    invariant guarantees the deque is non-empty at these sites).
  - `IncrementalPeakDetector::push_row` / `flush` — same `if let` pattern
    for the per-column vertical deques.
  - `StreamingPanako` / `StreamingWang` `emit_finalized_anchors` — switched
    from peek-then-`pop_front().unwrap()` to a pop-and-push-front pattern
    that re-queues the anchor when its target zone is not yet finalised.
  Public API and signatures unchanged; behaviour bit-identical
  (`streaming_offline_equivalence` and 1-sample-per-push tests still pass).

- **Internal: added regression tests for the refactor above plus broader
  coverage gaps.** 28 new unit tests across the affected paths:
  - 8 in `dsp::peaks` + `classical/{wang,panako}` pin the
    `IncrementalPeakDetector` per-row output and the
    `emit_finalized_anchors` re-queue / emit-all / idempotent
    invariants directly. A forgotten `push_front` (or any other
    re-queue mistake) is now caught at the site of the change, not
    indirectly via `streaming_offline_equivalence`.
  - 4 in `classical/{wang,panako,haitsma}` + `neural/embedder` pin
    `Fingerprinter::name()` and the streaming `latency_ms()` against
    documented values. A silent rename of any fingerprinter would
    now break a unit test rather than only the regression goldens.
  - 11 constructor panic tests in `dsp::mel` (5), `classical/haitsma`
    (2), and `dsp::resample` (4). Every documented panic on
    `MelFilterBank::new`, `Haitsma::new`, `linear()`, and
    `SincResampler::with_quality` is now `#[should_panic]`-pinned.
  - 5 in `dsp::stft` and `error`: direct tests for
    `ShortTimeFFT::power_flat` (`power == |magnitude|²`) and
    `power_flat_into` (zero-alloc on the hot path), plus
    `Display` text for the two previously-untested `AfpError`
    variants (`UnsupportedChannels`, `Io`).
  - 1 in `neural/embedder`: end-to-end `NeuralEmbedder::extract`
    happy path using the in-process passthrough tract model —
    verifies window count, `embedding_dim`, L2 normalisation, and
    `t_start` arithmetic. Replaces the prior gap where the offline
    `extract` was only smoke-tested via the streaming passthrough
    fixture.

  Only production change: `SincQuality` now derives `PartialEq` so
  its getter test can assert equality. Library test count: 112 → 183
  (`--features neural`), 112 → 139 (default features).

### Performance

- **Streaming peak detection is now truly incremental (~16× faster).**
  `StreamingWang::push` and `StreamingPanako::push` previously called
  `rolling_max_2d_pooled` over the full 31-row spectrogram window on
  every newly-ripe frame, even though only a single row's result was
  consumed — ~31× redundant work per frame (the dominant streaming cost
  noted in 0.3.3). A new `IncrementalPeakDetector` caches the horizontal
  rolling-max once per row when it's appended and maintains per-column
  vertical Lemire deques across pushes, producing the 2-D rolling max
  for the single ripe row in amortised O(n_bins) instead of
  O(n_rows × n_bins). Per-push CPU no longer depends on
  `neighbourhood_t`. Bit-exact streaming/offline parity is preserved —
  all existing equivalence tests pass unchanged, including the
  1-sample-per-push pathological case.
  `cargo bench --bench streaming` results (5 s synthetic, default config):

  | Fingerprinter | Before     | After     | Δ           |
  | ------------- | ---------- | --------- | ----------- |
  | Wang (256)    | 226 ms     | 13.3 ms   | **−94.1 %** |
  | Wang (1 s)    | 229 ms     | 13.3 ms   | **−94.2 %** |
  | Panako (256)  | 231 ms     | 14.2 ms   | **−93.9 %** |
  | Panako (1 s)  | 233 ms     | 14.3 ms   | **−93.9 %** |

  Streaming throughput now matches the offline extractors (~3 Melem/s).
  Closes #1.

## [0.3.3] - 2026-06-04

A correctness and realtime-allocation patch release for the classical
streaming paths. Public API and fingerprint outputs are unchanged for
normal inputs; the release tightens deterministic selection, decoder
buffer handling, and the allocation guarantees documented for streaming
push.

### Performance

- **Streaming `push` is now genuinely allocation-free after warmup.**
  `dsp::peaks::rolling_max_1d` allocated a fresh `VecDeque` on every
  call, and `rolling_max_2d_pooled` invokes it `(n_rows + n_cols)` times
  — so each streaming `detect_rows` performed hundreds of small heap
  allocations per frame, contradicting the README's "allocation-free hot
  path after warmup" claim. A single pooled `VecDeque<usize>` is now
  owned by each caller (`PeakPicker`, `StreamingWang`, `StreamingPanako`)
  and threaded through `rolling_max_2d_pooled` into `rolling_max_1d`,
  where it is `clear()`ed (capacity retained) instead of reallocated.
  Bit-exact streaming/offline parity is preserved — verified by the
  existing `streaming_offline_equivalence` and 1-sample-per-push tests.
  This removes per-frame allocator traffic (a realtime-jitter win for
  audio-callback use); steady-state throughput is unchanged at default
  config, where peak-picking dominates per-frame cost —
  `cargo bench --bench streaming` shows all deltas within the ~3–4 %
  run-to-run noise floor. The larger streaming cost is the redundant
  full-window rolling-max recompute tracked in #1.

### Fixed

- **Adaptive per-second peak selection is now a total order.** Both the
  offline `adaptive_per_second` and the streaming `finalize_bucket`
  truncate to `peaks_per_sec` by magnitude; neither had a positional
  tiebreak, so two peaks of *exactly* equal `f32` magnitude straddling
  the cap could be resolved differently by `sort_unstable` in the two
  paths — a latent gap in the "bit-exact under arbitrary chunking"
  guarantee. All three truncation sorts now break ties on
  `(t_frame, f_bin)` (unique per peak), making selection deterministic
  and identical across offline/streaming. No golden change (synthetic
  inputs produce no exact ties); new unit test
  `adaptive_per_second_breaks_exact_mag_ties_by_position` pins the
  contract.

- **File decoder reallocates its conversion buffer when a packet grows.**
  `io::decode_to_mono` sized the `f32` conversion buffer from the first
  decoded packet only. A later packet that decodes to more frames than
  the first (legal for some containers) would have decoded into an
  undersized buffer. The buffer is now rebuilt whenever a packet's
  capacity exceeds the current buffer.

### Changed

- **Internal: collapsed duplicated top-K heap wrappers.** Wang's
  borrowed `MinByMag` and Panako's borrowed `MinByScore` were
  byte-identical in `Ord` logic to their owned
  `MinByMagOwned` / `MinByScoreOwned` siblings, differing only in
  borrowed-vs-owned storage. Since `Peak` is `Copy`, the offline
  builders now use the owned variants too, removing the borrowed structs
  and their hand-written `PartialEq`/`Eq`/`Ord` impls (~55 lines). Also
  hoisted Slaney's `min_log_mel` (a pure compile-time division) to a
  `const` instead of recomputing it per `hz_to_mel` / `mel_to_hz` call.
  No public API or output change — goldens unaffected.

[0.3.3]: https://github.com/themankindproject/audiofp/compare/v0.3.2...v0.3.3

## [0.3.2] - 2026-05-26

A correctness + clarity patch release driven by an end-to-end internal
audit. Behaviour is bit-for-bit unchanged for any input that completed
successfully on 0.3.1 — regression goldens are untouched. The visible
changes are clearer error messages, an additional convenience constant,
and tightened guarantees in places where 0.3.1's docs overstated.

### Added

- **`SampleRate::HZ_5000`** — convenience constant for the rate
  [`Haitsma`] consumes. The previously verbose
  `SampleRate::new(5_000).unwrap()` form is now obsolete; every
  in-tree example, test, bench, and fuzz target has been migrated.

### Changed

- **`AfpError::UnsupportedSampleRate`** message dropped its
  `(supported: 8000, 11025, 16000, 22050, 44100, 48000)` parenthetical.
  The list was misleading: each fingerprinter has its own required
  rate, and Haitsma's 5 kHz was conspicuously absent. Callers should
  consult `Fingerprinter::required_sample_rate()` (now linked from the
  variant's docstring) for the per-algorithm answer.

- **`SampleRate` `HZ_*` constants** are now built from safe `const`
  `Option::unwrap` instead of `unsafe { NonZeroU32::new_unchecked }`.
  Same compile-time evaluation, no runtime cost, no `unsafe`.

- **`SampleRate::HZ_8000` docstring** corrected from "the rate
  audiofp's classical fingerprinters consume" (false — Haitsma uses
  5 kHz) to "the rate Wang and Panako consume". `HZ_16000` doc clarifies
  it is also the AudioSeal watermark default.

- **`WatermarkDetector` typed-model cache** now stores
  `Option<(usize, TypedModel)>` instead of `Option<TypedModel>`. If a
  later `detect()` call passes a different-length buffer, the typed
  plan is transparently rebuilt for the new length instead of letting
  Tract emit a cryptic shape error. Equal-length repeat calls still
  reuse the cached plan as before.

### Fixed

- **Wang `Δt` packing** truly clamps to `[1, 16383]` instead of
  masking the low 14 bits. The module docstring already advertised
  clamping as the contract; the code now matches. The fix is
  bit-identical for all `Δt` values that arise under default
  `target_zone_t` (≤ 63), so regression goldens are unaffected. A new
  unit test `dt_field_clamps_to_14_bit_ceiling_not_wraparound` pins
  the contract for any future change.

- **`StreamingNeuralEmbedder::try_push_with` is now genuinely
  zero-allocation per call.** The embedding scratch `Vec<f32>` is now
  a struct field allocated once at construction with capacity
  `embedding_dim`, instead of a fresh `Vec::with_capacity(...)` on
  every call. The README claim of "zero-alloc try_push_with" is now
  technically accurate. New regression test
  `try_push_with_does_not_reallocate_embedding_scratch` proves the
  buffer's capacity is preserved across 50 emits and 10 separate
  pushes.

- **`examples/hash_matcher.rs`** no longer hardcodes
  `FRAMES_PER_SEC = 62.5`. The example now reads `frames_per_sec`
  from the produced `WangFingerprint`, so it stays correct if Wang's
  frame rate is ever changed in a future major version.

### Documentation

- **README** dropped stale `(0.2.0)` and `(0.3.0)` parenthetical
  version markers from the feature bullets — they had become
  misleading rather than informative. Watermark and neural bullets
  now describe the cache + scratch behaviour the new code actually
  delivers.

[`Haitsma`]: https://docs.rs/audiofp/0.3.2/audiofp/classical/struct.Haitsma.html
[0.3.2]: https://github.com/themankindproject/audiofp/compare/v0.3.1...v0.3.2

## [0.3.1] - 2026-05-16

A performance-focused patch release with a new zero-allocation streaming
API and a breaking change to `SincQuality` struct literals.

### Added

- **`StreamingFingerprinter::push_with` / `flush_with`** — zero-allocation
  callback variants that invoke `FnMut(TimestampMs, &Frame)` per emitted
  frame instead of allocating a `Vec`. Default implementations delegate
  to `push`/`flush` so existing trait implementors are unaffected.

- **`ShortTimeFFT::power_flat_into`** — writes the power spectrogram
  directly into a caller-owned `&mut Vec<f32>`, avoiding the intermediate
  allocation of `power_flat`.

### Performance

- **`WatermarkDetector` caches `TypedModel`** after the first `detect()`
  call, skipping `with_input_fact + into_typed` on subsequent invocations.

- **`Wang::extract` / `Panako::extract`** use `power_flat_into` with
  in-place log-magnitude conversion, eliminating a `clear + resize +
  copy` per call.

- **`f_a_q` hoisted** out of per-target loops in `build_hashes` and
  `build_hashes_for_anchor` (wang.rs) — computed once per anchor instead
  of once per target pair.

- **`StreamingPanako` pools** its triplet scratch `Vec` across
  `build_triplets_for_anchor` calls, and uses `MinByScoreOwned` to avoid
  lifetime-erased heap allocation.

- **`SincResampler` precomputes** a polyphase kernel table at construction
  time, replacing per-sample `sinc × Kaiser` evaluation with a table
  lookup during `process()`.

### Changed

- **`SincQuality`** gains a required `polyphase_steps: u16` field
  (default 256). This is a **breaking change** for explicit struct
  literal constructions: `SincQuality { half_taps: 32, kaiser_beta: 8.6 }`
  no longer compiles. `SincQuality::default()` continues to work.

[0.3.1]: https://github.com/themankindproject/audiofp/compare/v0.3.0...v0.3.1

## [0.3.0] - 2026-04-28

A feature release: ships the neural fingerprinting module the `neural`
feature has been a placeholder for since 0.1, and a measured streaming
hot-path perf fix in the classical fingerprinters.

### Added

- **`audiofp::neural`** (gated on the `neural` feature) — a generic
  ONNX log-mel audio embedder. Bring your own model: any ONNX file
  whose first input is `[1, n_mels, n_frames] f32` and whose first
  output is a flat `f32` embedding vector works against the documented
  contract. Two top-level types:

  - `NeuralEmbedder` (impl `Fingerprinter`) for whole-buffer
    extraction. Slides analysis windows across the input and emits
    one `NeuralEmbedding { vector, t_start }` per window.
  - `StreamingNeuralEmbedder` (impl `StreamingFingerprinter`) with
    a bounded sample carry, `try_push` for error-propagating
    inference, and **`try_push_with(samples, |t, &[f32]| ...)`** for
    zero-allocation streaming where the embedding is handed to the
    callback by reference.

  Reasonable defaults (16 kHz, 1024 FFT, 320 hop, 128 mels, 1 s
  non-overlapping windows, Slaney mel, Hann window, L2-normalised
  output) via `NeuralEmbedderConfig::new(model_path)`.

  All expensive work (model typing, optimisation, runnable plan
  construction) happens **once** in `new()` with a fully-concrete
  input shape — the watermark detector's per-call `clone +
  optimize + runnable` pattern is explicitly avoided.

- **`MelFilterBank::log_mel_from_power`** — log-mel from a power
  spectrum (`re² + im²` per bin). Equivalent to `log_mel(sqrt(p))`
  but skips the redundant per-bin square when the upstream is
  `ShortTimeFFT::power_flat` / `process_frame_power`. Verified
  bit-equivalent to the existing `log_mel` on squared input.

- **Memory-bound regression tests** for all three classical streaming
  fingerprinters (`streaming_state_stays_bounded_under_long_input` ×
  3). Push 30 s of audio in 256-sample chunks and assert tight
  ceilings on every internal buffer, including the rolling spectrogram
  rows, bucket-pending map, and pending-anchors deque.

- **`benches/streaming.rs`** — Criterion microbenches for the
  classical streaming push throughput, two patterns each (small
  256-sample chunks ≈ realtime mic; large 1 s chunks ≈ offline
  batch). Captures the cost shape that's easy to regress and that
  gates further perf work.

- **`benches/neural_frontend.rs`** — Criterion microbenches for the
  neural front-end (log-mel pipeline, strided tensor write, L2
  normalise). Used to validate the ≥ 5 % bench-driven bar on perf
  changes; documented the bench-driven decisions in `future.md`.

### Performance

- **`StreamingHaitsma::push` large-chunk: -25 %.** A 1 s push at
  default config (HAITSMA_HOP = 64, HAITSMA_N_FFT = 2048) used to
  call `sample_carry.drain(0..HOP)` *inside* the per-frame loop —
  78 frames × ~5 KB shifted per drain = ~770 KB of cumulative
  memmove per push. Replaced with an offset cursor and a single
  drain at the end of the call. Bench: 10.44 ms → 7.78 ms.

- **`StreamingWang::push` and `StreamingPanako::push`** received the
  same drain-once-per-push refactor *and* lost a per-frame
  `frame_scratch.clone()` (a fresh `Vec<f32>` allocated every frame
  purely to satisfy a borrow conflict, replaced with a new
  `append_frame_scratch_row` method that copies via disjoint field
  borrow). Both changes are within bench noise at the default config
  (peak picking dominates per-frame cost there) and were kept on
  correctness grounds: drain is now O(N) instead of O(N²) per push,
  and per-frame allocator traffic is gone.

### Changed

- **`neural` feature** now actually pulls in `tract-onnx` and exposes
  `audiofp::neural`. Previously a no-op placeholder. Users with
  `default-features = false, features = ["neural"]` will now see the
  module.

- **Streaming push internals**: `sample_carry` is drained exactly
  once per `push()` call instead of per frame. No observable
  semantic change — bit-exactness with `extract` preserved across
  all chunk sizes (verified by `streaming_chunk_size_invariant` and
  `streaming_with_one_sample_chunks_still_matches_offline`).

### Documentation

- Updated `future.md` with two new entries: §1.1 (neural fingerprinter)
  marked done with three deferred follow-ups (batched offline
  inference, SIMD log-mel matvec, and the bench-driven skip list);
  §2.8 marked done with §2.8.1 documenting the streaming hot-path fix.

## [0.2.1] - 2026-04-27

### Added

- **`examples/hash_matcher.rs`** — runnable demo of the time-aligned
  voting algorithm that turns Wang landmark hashes into actual
  fingerprint matching. Multi-track enrollment, per-query Δt-histogram
  scoring, top-5 results with offset and a confident-match heuristic:

  ```bash
  cargo run --example hash_matcher --release -- ref1.flac ref2.flac -- query.mp3
  ```

- **Property-based tests** (`tests/property.rs`) via `proptest`. Four
  invariants checked under a randomly-generated mix of seed and chunk
  patterns:

  - `StreamingWang` ↔ `Wang::extract` hash multisets match.
  - `StreamingPanako` ↔ `Panako::extract` hash multisets match.
  - `StreamingHaitsma` ↔ `Haitsma::extract` frame sequences match.
  - `Wang::extract` is deterministic (twice on same input → identical).

  Default 16 cases per property; bump with
  `PROPTEST_CASES=2000 cargo test --test property`.

## [0.2.0] - 2026-04-27

A performance-focused minor release, driven by a hot-path audit using
the recon code-intelligence MCP server.

### Performance

Measured on Intel i5-1135G7 (2.40 GHz) over a 30 s synthetic input:

| Algorithm | 0.1.1   | 0.2.0   | Δ        |
| --------- | ------- | ------- | -------- |
| Wang      | 109 ms  |  99 ms  | **-9.6 %**  |
| Panako    | 109 ms  | 104 ms  | **-4.7 %**  |
| Haitsma   |  65 ms  |  47 ms  | **-27.4 %** |

Wins compound across the 7 changes below; Haitsma sees the biggest lift
because it's FFT-bound and benefits most from the new contiguous
spectrogram and skip-sqrt path.

### Added

- `dsp::stft::ShortTimeFFT::magnitude_flat` — returns the magnitude
  spectrogram as a single contiguous `Vec<f32>` of shape
  `(n_frames, n_bins)` plus the dimensions, instead of the
  per-frame-allocated `Vec<Vec<f32>>` that `magnitude` returns. One
  allocation per call instead of one per frame, and downstream consumers
  can slice it without indirection.
- `dsp::stft::ShortTimeFFT::power_flat` — same shape but emits
  `re² + im²` instead of `sqrt(re² + im²)`, useful when the next stage
  is `log10` (since `20·log10(sqrt(p)) ≡ 10·log10(p)`) or any
  power-domain operation. Saves a per-bin `sqrt` over the entire
  spectrogram.

### Changed

- **Breaking:** `dsp::peaks::PeakPicker::pick` now takes `&mut self` so
  it can re-use its internal scratch buffers across calls. If you
  previously held a `PeakPicker` behind `&self`, store it as
  `Mutex<PeakPicker>` or use one picker per producing thread. This
  eliminates three `Vec::new() + resize()` allocations per `pick`
  invocation.
- **Hash output regenerated.** `Wang`, `Panako`, and `Haitsma` now
  consume `power_flat` directly and apply the algebraically-equivalent
  `10·log10(power)` instead of `20·log10(sqrt(power))`. The two forms
  agree mathematically, but `f32` rounding through one less operation
  produces last-bit differences in the resulting hashes. Goldens in
  `tests/goldens/{wang_v1,panako_v2,haitsma_v1}.bin` were regenerated.
- `dsp::stft::ShortTimeFFT::fill_windowed` takes a fast inner path with
  no per-sample bounds or reflect check when the window slot lives
  entirely inside the input buffer (≈ 99 % of frames in any non-edge
  audio). Slow path retained for the boundary cases.
- `Wang`, `Panako`, and `Haitsma` cache a `PeakPicker` (and pooled
  log-magnitude `Vec<f32>`) as struct fields instead of constructing
  them on every `extract` call.
- `Wang::build_hashes` and `Panako::build_triplet_hashes` now use a
  size-bounded `BinaryHeap` (`O(N log K)`) for the per-anchor top-K
  selection instead of a full sort followed by `truncate`
  (`O(N log N)`). Output is unchanged because the kept K elements are
  re-sorted deterministically before emission.

- **`StreamingWang`, `StreamingPanako`, and `StreamingHaitsma` are now
  fully incremental.** The previous implementation re-ran the entire
  offline pipeline on every push (`O(N²)` total CPU in stream length).
  The new implementation:

  - Wang / Panako: maintain a rolling log-power spectrogram window of
    `2·neighborhood_t + 1` rows; detect peaks frame-by-frame as each
    becomes ripe (full forward neighbourhood visible); accumulate
    candidates per 1-second bucket and finalise them with the offline
    adaptive threshold once the next bucket starts; grow per-anchor
    target heaps incrementally; emit hashes when an anchor's target
    zone is fully observed.
  - Haitsma: trivially incremental — each output bit-frame depends only
    on the current and previous frames' band energies, so we keep one
    32-element `prev_energy` array and emit immediately per new frame.

  Per-push CPU is now proportional to the number of new samples,
  independent of total stream length. **Bit-exact equivalence with
  `extract` is preserved — verified by the existing equivalence tests
  including the 1-sample-per-push pathological case.**

## [0.1.1] - 2026-04-27

### Added

- **Criterion benchmark harness** (`benches/extract.rs`). Runs each
  classical fingerprinter (`Wang`, `Panako`, `Haitsma`) over 2 s, 5 s,
  and 30 s of deterministic synthetic input. Reproducible numbers via a
  seeded xorshift32 generator (matching the regression-golden test's
  input). Run with `cargo bench --bench extract`.

- **Synthetic robustness tests** (`tests/robustness.rs`). Six tests
  verifying each classical fingerprinter retains a calibrated minimum
  hash overlap (Jaccard for Wang/Panako, bit similarity for Haitsma)
  under two reproducible perturbations: SNR-based additive noise and
  a 1-pole IIR lowpass. Synthetic only — real codec round-trips
  (MP3/AAC/Opus through ffmpeg) are still on the roadmap.

### Changed

- **README performance section** replaces the previous "design notes"
  placeholder with measured numbers from the criterion harness on
  Intel i5-1135G7 (2.40 GHz). Sample timings: Wang/Panako 30 s in
  ≈ 109 ms (≈ 275× realtime), Haitsma 30 s in ≈ 65 ms (≈ 462× realtime).

## [0.1.0] - 2026-04-26

Initial release of `audiofp`, an audio fingerprinting SDK for Rust.

### Added

- **Core types and traits**:
  - `AfpError` (`#[non_exhaustive]`) covering audio length / sample-rate / channel /
    config / model load / inference / I/O / buffer-overrun failures, with `Display`
    impls suitable for end-user diagnostics.
  - `Result<T>` alias for `core::result::Result<T, AfpError>`.
  - `SampleRate` (NonZeroU32 newtype) with `HZ_8000` … `HZ_48000` constants and
    `SampleRate::new(u32) -> Option<Self>` for arbitrary rates.
  - `AudioBuffer<'a>` borrowed mono PCM view + `TimestampMs` ordered timestamp.
  - `Fingerprinter` and `StreamingFingerprinter` traits — the two extraction shapes
    every algorithm in the crate exposes.

- **Three classical fingerprinters** (`audiofp::classical`), each with offline
  (`Fingerprinter`) **and** streaming (`StreamingFingerprinter`) variants:
  - **`Wang`** / **`StreamingWang`** — Shazam-style anchor-target landmark pairs at
    8 kHz. STFT `n_fft = 1024`, `hop = 128`, Hann; 31×31 dB-domain peak picker
    capped at 30 peaks/s; 32-bit hash `f_a_q (9) | f_b_q (9) | Δt (14)`.
    `latency_ms() = 2_256`.
  - **`Panako`** / **`StreamingPanako`** — Six 2021 triplet hashes at 8 kHz with
    same front-end as Wang. Anchors paired with two targets; tempo-invariant β
    (5 bits) ratio robust to ±5 % time stretch.
    Hash: `sign (2) | mag_order (2) | β (5) | Δf_ab (8s) | Δf_bc (8s) | reserved (7)`.
    `latency_ms() = 2_784`.
  - **`Haitsma`** / **`StreamingHaitsma`** — Haitsma–Kalker / Philips robust hash at
    5 kHz. STFT `n_fft = 2048`, `hop = 64`; 33 log-spaced bands from 300–2000 Hz;
    32 sign bits per frame `n ≥ 1` comparing band-difference deltas with the
    previous frame. "MSB-zero" packing (band 0 → bit 31). `latency_ms() = 409`.

- **DSP primitives** (`audiofp::dsp`), all `no_std + alloc`:
  - `windows` — periodic Hann / Hamming / Blackman generators
    (`fftbins=True`-equivalent for librosa parity).
  - `stft::ShortTimeFFT` — pre-planned real-input STFT via `realfft`, with reusable
    scratch and an optional librosa-style reflect-padding (`center: true`).
  - `mel::MelFilterBank` — slaney-normalised triangular filters with HTK and Slaney
    hz↔mel conventions; `log_mel` matches librosa's `feature.melspectrogram` +
    `power_to_db` defaults.
  - `peaks::{Peak, PeakPicker}` — 2-D peak picker built on Lemire's monotonic-deque
    sliding max (amortised O(N · M) regardless of neighbourhood size), plus a
    per-second adaptive cap. `Peak` is `bytemuck::Pod` for direct mmap / FFI.
  - `resample::{linear, SincResampler, SincQuality}` — straight linear resample for
    cheap paths; windowed-sinc (Kaiser) with auto anti-aliasing cutoff for quality.

- **Audio file decoding** (`audiofp::io`, gated on `std`):
  - `decode_to_mono(path) -> Result<(Vec<f32>, u32)>` and
    `decode_to_mono_at(path, target_sr) -> Result<Vec<f32>>` via Symphonia.
  - Multi-channel files are downmixed to mono by averaging channels per frame.
  - Resampling at `decode_to_mono_at` uses the SDK's built-in `SincResampler`.
  - Supports MP3, AAC (in MP4), FLAC, OGG-Vorbis, WAV, raw PCM. Recoverable
    per-packet decode failures are silently skipped (resilient to corrupt blocks).

- **Watermark detection** (`audiofp::watermark`, gated on `watermark` feature):
  - `WatermarkDetector` — AudioSeal-compatible ONNX wrapper built on `tract-onnx`.
  - `WatermarkConfig::new(path)` constructor with AudioSeal defaults
    (`message_bits = 16`, `threshold = 0.5`, `sample_rate = 16_000`).
  - `WatermarkResult { detected, confidence, message: u32, localization: Vec<f32> }`.
  - Loader holds the model with no fixed input shape; each `detect()` concretises
    the input length, runs inference, and decodes `[detection, message_logits]`
    outputs (LSB-first bit packing).

- **Streaming / offline equivalence**: every streaming fingerprinter is verified
  bit-for-bit against the offline extractor under randomised chunking, including
  the 1-sample-per-push pathological case.

- **Test coverage**: 111 unit tests + 20 doctests across:
  - error variants, value types, trait shapes
  - DSP determinism + librosa-aligned conventions
  - Lemire 2-D rolling max validated against an O(N · M · K²) reference
  - hash bit-field decoding for Wang and Panako (sign, mag_order, β, Δf clamping)
  - Haitsma "MSB-zero" packing (band 0 → bit 31, band 31 → bit 0)
  - Wav round-trip decoding (16-bit int, 32-bit float, mono + stereo, multiple SRs)
  - Watermark error paths (missing file, corrupt protobuf, invalid config)

- **Tooling**:
  - `rustfmt.toml`, `clippy.toml`, `deny.toml` (license allowlist, MPL-2.0
    exception explicit for Symphonia).
  - `rust-toolchain.toml` pins MSRV to 1.85.0.
  - GitHub Actions `ci.yml` runs `cargo fmt --check`, `cargo clippy
    --all-targets --all-features -- -D warnings`, and `cargo test
    --all-features` in parallel jobs.

### Cargo features (added in this release)

- `std` (default): pulls in `symphonia` and exposes `audiofp::io`.
- `watermark`: pulls in `tract-onnx` + `ndarray`, exposes `audiofp::watermark`.
- `neural`: reserved for the upcoming ONNX neural fingerprinter (no surface yet).
- `mimalloc`: installs `mimalloc` as the process-wide `#[global_allocator]`.
  Off by default — libraries should not pick the allocator on behalf of their
  downstream binaries.

### Known limitations

- **Embedded build (Cortex-M).** `rustfft` (transitive dep of `realfft`)
  unconditionally enables `num-traits/std`, so the no_std DSP path only
  compiles on hosted targets. True bare-metal support will require swapping
  the FFT backend (`microfft` is the planned target).
- **Streaming implementation.** `StreamingWang` / `StreamingPanako` /
  `StreamingHaitsma` rerun the offline pipeline on each `push()` to guarantee
  bit-exact parity with `extract`. This is correct but quadratic in stream
  length — an incremental implementation is on the roadmap.
- **No mic capture / live audio orchestrator.** `cpal`-based capture and the
  `Pipeline<F: StreamingFingerprinter>` driver from the spec are deferred to a
  later release. For now, drive `StreamingFingerprinter::push` directly from
  whatever capture mechanism your application uses.
- **No neural fingerprinter yet.** The ONNX-based Resona-FP head is reserved
  as feature `neural` but not yet implemented.
- **No constant-Q transform.** None of the classical fingerprinters need it;
  it's deferred until a downstream consumer requires one.
- **No bundled regression goldens.** Bit-exact regression goldens against
  committed v1 outputs aren't included; codec robustness benchmarks against a
  held-out corpus are also pending.

[Unreleased]: https://github.com/themankindproject/audiofp/compare/v0.3.9...HEAD
[0.3.9]: https://github.com/themankindproject/audiofp/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/themankindproject/audiofp/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/themankindproject/audiofp/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/themankindproject/audiofp/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/themankindproject/audiofp/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/themankindproject/audiofp/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/themankindproject/audiofp/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/themankindproject/audiofp/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/themankindproject/audiofp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/themankindproject/audiofp/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/themankindproject/audiofp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/themankindproject/audiofp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/themankindproject/audiofp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/themankindproject/audiofp/releases/tag/v0.1.0
