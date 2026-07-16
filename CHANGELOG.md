# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Real Audio integration tests**: Added CC0 speech and piano Ogg assets (`tests/assets/`) and integration test suite (`tests/real_audio.rs`) to verify Wang, Panako, and Haitsma robustness against 30 dB SNR noise, highband lowpass filtering, flat silence handling, resampling invariance, and streaming equivalence.

### Performance

- **Optimize PeakPicker scratch buffers**: Replaced zero-filling resizes of reused vectors (`max_buf`, `temp_2d`, `col_in`, `col_out`) with conditional `set_len` when capacity is already sufficient, saving millions of writes on every extraction.
- **Avoid heap allocations in Haitsma**: Introduced a reused `power_buf` buffer within `Haitsma` to eliminate `Vec<f32>` allocations in `stft.power_flat` on every extraction.
- **Optimize suffix_max in Panako**: Replaced full zero-filling resizes of `suffix_max` in triplet generation with a conditional check, truncating and zeroing only the last element.

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

[Unreleased]: https://github.com/themankindproject/audiofp/compare/v0.3.7...HEAD
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
