# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/themankindproject/audiofp/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/themankindproject/audiofp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/themankindproject/audiofp/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/themankindproject/audiofp/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/themankindproject/audiofp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/themankindproject/audiofp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/themankindproject/audiofp/releases/tag/v0.1.0
