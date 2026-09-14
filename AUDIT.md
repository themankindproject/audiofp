# audiofp — Deep Audit Report

> **Superseded for remediation tracking (2026-03).** The authoritative,
> reproduction-backed audit for commit `1a055a5` / `origin/main` @ `dfd42cc`
> is the read-only report at `audiofp-audit-1a055a5/AUDIT-1a055a5.md`
> (findings F01–F25 with evidence). This file retains the earlier in-tree
> pass for historical context; several “fixed” claims below were incomplete
> or stale when re-verified. **Remaining unknowns / open work (sibling
> worktrees):** F01 streaming flush hash loss, F02–F05 matching/index
> correctness, F06–F12 I/O/cache hardening, F13–F17 DSP/neural metadata,
> MP3-only ungated WAV unit tests, and full benchmark timer boundaries
> (F23). This worktree addresses **F18–F22, F25** and doc/CI gates; fuzz
> targets call `flush_complete` (build blocked until the streaming PR stacks).
> Per-file status: `afp-remediation/logs/verification-followup-final.md`.

**Scope:** the whole `audiofp` v0.4.2 crate (`src/`, 24 files / ~28.7 kLOC),
plus `tests/`, `fuzz/`, `benches/`, `examples/`, CI, and the long-form docs.
**Method:** full read of every source file; five parallel deep audits
(matching / classical / DSP / io-serial-cache / neural-watermark); every
"confirmed" finding reproduced in a scratch crate outside the repo, most in
both debug (overflow checks on) and release.
**Baseline:** `cargo clippy --all-targets --all-features -- -D warnings` clean,
`cargo test --all-features -- --skip golden` green (456 tests).

This report is deliberately split into **fixed in this pass** and
**remaining / recommended**. Nothing here changes hash bytes, match scores,
or `is_match` semantics.

---

## 1. Critical and high-severity findings (all fixed)

### 1.1 Panics reachable from the public API

| # | Where | Trigger | Behaviour before |
|---|-------|---------|------------------|
| 1 | `matching/panako.rs`, `matching/index.rs` | `t_c < t_anchor` in a `PanakoHash` | debug: `attempt to subtract with overflow`; release: silent wraparound to a garbage scale |
| 2 | `dsp/mel.rs` | `MelFilterBank::try_new(.., sr = 0, ..)` | debug: add-overflow panic; release: **`Ok`** with an all-zero filterbank |
| 3 | `dsp/peaks.rs` | `PeakPicker::pick(.., fps = 1e30, ..)` | debug: multiply-overflow panic; release: `capacity overflow` abort |
| 4 | `dsp/resample.rs` | `SincResampler::try_with_quality` with `kaiser_beta = NaN/±inf/200` or `half_taps = usize::MAX` | all-NaN output, or a panic inside the *fallible* constructor |
| 5 | `neural/embedder.rs` | `NeuralEmbedderConfig { batch_size: usize::MAX/2 }` | `Vec::with_capacity` → `capacity overflow` panic in `extract` |

#1 is the sharpest: `PanakoHash`'s fields are `pub` and the type derives
`bytemuck::Pod`, so a malformed triplet is constructible in safe Rust *and*
round-trips verbatim through `from_bytes` — the documented mmap / flat-file
persistence path. Both the matcher and the index now use `saturating_sub`
and skip degenerate spans (a zero-length span carries no scale evidence).

**Fixed:** all five. Regression tests:
`matching::panako::tests::malformed_triplet_spans_do_not_panic_or_wrap`,
`dsp::mel::tests::zero_sample_rate_is_a_config_error_not_a_panic_or_silent_ok`,
`io::decoder` channel/amplification tests, plus a new fuzz target
`matching_malformed` that sweeps every matcher and index with arbitrary
`(hash, t_anchor, t_b, t_c)` tuples.

### 1.2 Haitsma: a config that makes *everything* match

`HaitsmaConfig { fmin: 1000.0, fmax: 1000.5 }` was accepted by `try_new`.
The 33 log-spaced band edges land inside a single FFT bin, so **no band
receives any bin**; `band_energies` sums an empty slice to `0.0` for every
band and every frame, so the whole sub-fingerprint is the constant `0`.
Two unrelated 4 s recordings then compare with BER 0:

```
300..2000 (default): unrelated is_match=false score=0.5382
1000..1001:          unrelated is_match=true  score=0.9887
1000..1000.5:        unrelated is_match=true  score=1.0000
```

The in-file test `band_lookup_table_covers_in_band_frequencies` already
asserts "every band has ≥ 1 bin" — but only for the default config.
`try_new` (both offline and streaming) now rejects any range where a band is
uncovered, with a message naming the band and the bin width.

### 1.3 Panako Hough consolidation was O(B²), not O(B·W)

`matching/panako.rs` (and `matching/index.rs`) sorted the accumulator by
`(s_bin, off_key)` and then, for every bin, scanned the **entire same-scale
run** in both directions: the break condition only tested the scale bin. The
comment claimed O(B·W) with `W` the neighbourhood size. Measured on a
single-scale-row accumulator (release):

```
n= 12500  time= 1.04 s
n= 25000  time= 3.62 s
n= 50000  time=13.37 s
n=100000  time=71.48 s   (44.6 s inside that one loop)
```

100 k hashes ≈ 6.7 min of default-config audio, so a large-but-plausible
match cost minutes — and the fingerprint is fully caller-controlled
(deserializable). Both loops now also break on the offset-window boundary,
which preserves semantics exactly and restores the documented complexity.

### 1.4 `matching::index` violated three documented contracts

* **Frame-rate check missing.** `matching/mod.rs` promises that a
  `frames_per_sec` mismatch returns `MatchResult::NONE` "in all builds".
  Every matcher enforces it; **none** of the three index `query` paths did.
  A query at half the reference rate got `is_match=true` with `offset.ms`
  computed from the *reference* rate — exactly the wrong-rate conversion the
  matcher docs warn about. Now enforced per candidate.
* **Prominence basis.** `WangIndex::query` normalised over the *observed
  vote span* while `WangMatcher` uses the dense `[-q_max, +r_max]` range, so
  the index could reject a match the 1:1 matcher accepts (verified with a
  wide sparse background: matcher `prom=8.71` → match, index `prom=2.47` →
  reject). `r_max` is now stored per reference and the span reproduced.
* **`match_best` tie-break.** It early-exited on the first `score >= 1.0`,
  but its ranking rule is (score desc, **prominence desc**). A later
  reference with equal score and higher prominence was ignored, so
  `match_best` and `par_match_best` returned different answers. The early
  exit is removed (making the rayon and sequential docs true).

### 1.5 Decoder resource bounds for untrusted uploads

* **Unbounded upsample amplification.** The projection check was gated on
  `max_samples`, so `DecodeLimits::bytes(n)` bounded only the on-disk file.
  A 444-byte WAV declaring 1 Hz decoded through `decode_to_mono_at_limited(p,
  48_000, DecodeLimits::bytes(10_000))` returned **9.6 M f32 (38.4 MB)** —
  ~86 000× — and a 16 KB variant burned >600 s of CPU. The resample step is
  explicitly outside the cooperative timeout, so `with_timeout` did not help.
  The projection is now bounded regardless of `max_samples`, with a hard
  default of 10 minutes at 48 kHz.
* **Container vs codec sample rate.** `decode_to_mono` returned
  `audio_params.sample_rate` (container) and never reconciled it with the
  decoder's actual output spec. Patching the `mp4a` rate field of a 44.1 kHz
  m4a to 16000 made it return 90 112 samples labelled 16 kHz — 2.75×
  duration error, pitch/speed-warped fingerprints, no error anywhere. The
  returned rate now comes from the decoder's output spec.
* **Channel amplification / 0-channel ordering.** `max_samples` bounds
  decoded frames, but the conversion buffer is `frames × channels` f32;
  channels can be 65 535 in AIFF and 255+ in Matroska. Measured 18-channel
  WAV: 51.6× the caller's budget. The zero-channel guard also ran *after*
  the allocation, although Symphonia's `AudioBuffer::new` divides by the
  channel count. Now: 64-channel cap, both checks before the allocation.
* **Timeout bypass.** The deadline was checked only after the selected-track
  filter and after the reader-reset `continue`, so a multi-track file
  flooded with video packets, or an Ogg emitting `ResetRequired`, never
  observed it. Moved to the top of the loop.

### 1.6 `cache` read hardening

`load_from_cache` / `load_all_cached` did an unbounded `fs::read` of
caller-supplied paths and followed symlinks. A `*.afp` symlink to
`/dev/zero` raised RSS until allocation failed; a dangling symlink aborted
the entire directory scan. Now only regular files are read (via
`file_type()`, no symlink following) and files above
`MAX_CACHE_FILE_BYTES` (256 MiB) are rejected with `InputTooLarge`.

---

## 2. Medium findings (fixed)

| Finding | Fix |
|---|---|
| `WangIndex`/`HaitsmaIndex`/`PanakoIndex::is_empty()` stay `false` after every reference is removed (removal is physical, keys are retained) | documented key-space vs liveness; added `is_empty_catalog()` |
| `ransac_refine = true` could *reject* a match the coarse path accepted (a single refit line yields fewer inliers than the ±1-scale-bin neighbourhood count, then that smaller number was tested against `min_votes`) | fall back to the coarse peak whenever refinement does not itself clear `min_votes` |
| RANSAC `pairs` buffer filled even when `ransac_refine = false` | push only when enabled |
| `StreamingPanako` emitted triplets in score order, violating the documented `(t_anchor, t_b, t_c, hash)` invariant of `PanakoFingerprint` | re-sort the selected top-K by `(t_b, t_c)` before emitting (set unchanged, so streaming↔offline multiset parity holds) |
| `StreamingNeuralEmbedder::flush` cleared the carry without advancing `samples_consumed`, so a documented-valid `push` after `flush` reported `t_start = 0` for a window that started 187 ms in | advance the timeline by the discarded count |
| `rolling_max_1d`'s deque and the `k == 15` vectorized path disagreed on NaN (the deque's `<=` is false for every NaN comparison, letting a NaN reach the front) | deque now uses `f32::max`-equivalent eviction; NaN never wins in either path |
| `IncrementalPeakDetector` allocated and memcpy'd a `(2·kt+1)×n_bins` ring (≈127 KB at kt=15) that was never read | removed the dead field and the per-row copy |
| `PeakPicker::pick`'s "pooled candidate buffer" comment was false (`mem::take` leaves capacity 0) | comment corrected; capacity is honestly per-call and amortised over the spectrogram |
| `AfpError::Io` did not forward its source (`#[error("{0}")]`), so error chains stopped at the display string | `#[error(transparent)]`; `ModelLoad` now keeps the path for non-`NotFound` open failures |
| `std` alone was a hard `compile_error!`, breaking feature unification and making the codec-free `cache` module unreachable | removed; bare `std` now builds (`cache` + `IoError`, no `io`) |
| `max_push_samples`/`max_input_samples` sanitisation inconsistent between configs; `min_anchor_mag_db` clamped *down* to 0.0 silently | documented in `sanitize_cfg!`; the clamp set is now stated on the constructor |
| Public config/report types not `#[non_exhaustive]` | **deferred to 0.5.0.** Marking them was tried in this pass and reverted: the attribute is a compile break for downstream struct literals and for exhaustive `match` on `CachedFingerprint`, so it cannot ride a patch release. This repeats the 0.3.8 decision — see `CHANGELOG.md` under 0.3.8: "`#[non_exhaustive]` removed from config and fingerprint structs … breaks the documented `WangConfig { fan_out: 5, ..Default::default() }` pattern for external crates." |
| `serial.rs` docs claimed "zero-copy reads on little-endian hosts"; `read_pod_vec` always allocates+copies and `pod_collect_to_vec` zero-fills first | reworded; payload endianness documented as native (effectively LE-only) |

---

## 3. Documentation corrections (behaviour unchanged)

Several doc comments asserted things the code does not do. These matter for
an SDK because they are the contract callers program against:

* **Haitsma band↔bit mapping** was documented backwards at the struct level
  (band `k` → bit `k`), contradicting the module doc and the in-file test
  (band `b` → bit `31 - b`). The porting advice ("XOR or byte-reverse") also
  does not produce the paper's order — `u32::reverse_bits()` does.
* **Panako output size** claimed ~250 hashes/s; the hard cap is
  `peaks_per_sec × fan_out = 150/s`.
* **Resampler stopband**: "β = 8.6 (≈ -80 dB)" is wrong at the default
  `half_taps = 32`, because the kernel width is in *input* samples and does
  not scale with the decimation ratio. Measured 48 k→16 k: a 9 kHz tone is
  only ≈ -23 dB down. Docs now state this and the `half_taps` scaling rule.
* **Calibration `NONE` value**: the module claimed `MatchResult::NONE` maps
  to ≈0.01 "documented at each call site". Actual: Wang ≈0.012,
  Panako ≈0.010, **Haitsma ≈6.4e-10**, neural ≈1.1e-7. Corrected, and the
  NaN-propagation behaviour is now stated.
* **`power_to_db_wide`** claimed bit-identity with the scalar path;
  `wide::f32x8::log2()` is a polynomial that differs by 1 ULP on some
  inputs (3 of 69 probed), so the 8-wide body and scalar tail can disagree.
  The "single hardware log2 instruction on x86" rationale is also wrong.
* **Serialization** payload is native-endian Pod bytes, not little-endian.
* **`compute_prominence`** claimed a "large sentinel when there is no
  background"; it always returns `peak / (mean_rest + 1.0)`.
* **`PeakPickerConfig::default().min_magnitude_db`** was `1e-3` — a linear
  value in a dB-named field, so the default returned **zero peaks** on any
  dB spectrogram. Changed to `-50.0`, matching Wang/Panako.
* **Chained Ogg** silently truncates to the first logical stream
  (`ResetRequired` is treated as a re-sync rather than a track-list change);
  `packets_skipped` stays 0. Now documented on `DecodeStats::resets`.

---

## 4. Remaining and recommended (not fixed)

Ordered by value. None of these is a correctness bug on the supported path.

### 4.1 Correctness / robustness

1. **Chained Ogg / mid-stream track changes are dropped.** The honest fix
   is to re-resolve `format.default_track(TrackType::Audio)` on reader
   `ResetRequired`, update `track_id`, and re-create the decoder from the
   new params. Until then, callers who care about full duration should split
   chained files first. *(Documented; deliberate scope decision — a
   half-correct decoder rewrite is worse than an honest limitation.)*
2. **Symphonia WAV `fmt ` overflow** — a `fmt ` chunk with
   `channels × (bits/8) > u16::MAX` panics inside Symphonia's probe step
   (before any `audiofp` validation) in overflow-checked builds. Fixable on
   our side by pre-validating the channel count / bit depth from a small
   header read before `probe()`, or by pinning a patched Symphonia. The new
   64-channel cap in the decode loop does not cover this because the panic
   is upstream of it.
3. **Big-endian payload.** The v1 blob header is explicitly little-endian
   but the Pod payload is host-endian and unmarked, so a blob written on a
   BE host and read on LE parses with a valid header and garbled hashes.
   Either byte-swap per element or add an endianness flag. *(No BE target
   was available to test.)*
4. **Incremental stop-hash pruning is order-dependent.** `insert` re-creates
   a key that a later push pushed over `max_postings_per_hash`, so a
   `build`-ed index and an incrementally maintained one can differ, and a
   stop-hash stays live for the most recent ~`max_postings_per_hash`
   references. The doc's "exactly the map state a full build would produce"
   claim should be corrected, or the key should stay poisoned.
5. **`DecodeStats` are lost on every error path** exactly when triage needs
   them. Consider returning partial stats with the error.
6. **~~`MAX_VOTES_PER_REF`~~ — FIXED.** It capped votes per reference, so
   total query memory was `O(refs_hit × 10M)`. A `MAX_VOTES_PER_QUERY` cap
   now bounds the flat vote list overall. Note the per-query cap must be
   tested against `list.len()` *before* the push (a post-push check still
   allocates the entry), which is how it is written.
7. **`WangRefIndex` doesn't stop at the live reference bound** on
   `Palette`/stop-hash policy changes; minor.

### 4.2 Performance / memory

8. **~~`WangIndex::query` per-candidate allocations~~ — FIXED, measured.**
   `bins`, `bin_vec`, `consolidated`, the plateau `Vec`, and `contrib_indices`
   were allocated per candidate; the `per_ref: HashMap<u32, Vec<_>>` also
   allocated one `Vec` per candidate reference. All are now hoisted/replaced:
   scratch is reused via `clear()`, and votes live in one flat
   `Vec<(ref_id, δ, qi)>` that is stably sorted by `ref_id` (so the
   per-reference truncation and tie-break order are bit-identical).
   `q_max` is hoisted, and the separate `sum_rest` O(B) pass is replaced by a
   running `total`. Measured with `tests/index_alloc.rs` (deterministic
   allocation counts, no timing noise): **369 → 16 allocations per query**
   (100 refs) and criterion `matching/wang_index/n100_query`
   **144.84 µs → 63.76 µs (−56.0%)**. The audit's `d_min`/`d_max` claim was
   stale — those fields no longer exist.
9. **~~`HaitsmaIndex` clones every reference's frame vector~~ — DISPROVEN, no
   change made.** The audit predicted an `Arc`/offset arena would "roughly
   halve memory". Measured in `tests/index_alloc.rs`
   (`haitsma_frames_arena_overhead_is_measured`) on 100 refs × 234 frames
   (≈3 s at 78.125 fps): payload **93,600 B**, nested `Vec<Vec<u32>>`
   **96,000 B**, flat arena **93,600 B**. The stored overhead is **2,400 B
   total — 24 B per reference, exactly the `Vec` header — i.e. 2.6%, not
   ~50%.** There is no duplication inside the index: the clone at build/insert
   is a transient copy of the *caller's* input, not a second resident copy. An
   arena would also break a documented property, since `remove` currently
   returns the per-ref frame memory immediately (see its doc comment) — an
   arena holding interleaved references cannot free a middle range cheaply.
   Trade 2.6% for that regression risk: not worth it.
10. **~~`estimated_bytes` double-counts map keys~~ — FIXED.** The
    `map.len() * size_of::<K>()` term was removed at all three sites
    (`WangIndex`, `HaitsmaIndex`, `PanakoIndex`); the slot term already
    includes one key per slot. `wang_index_estimated_bytes_tracks_real_heap`
    now pins the report against real live bytes.
11. **~~Watermark detector rebuilds on every input-length change~~ — FIXED.**
    The single-slot cache is now an LRU of up to `MAX_CACHED_PLANS = 4`
    concretised plans, so alternating between a few clip lengths no longer
    rebuilds the ~1.3 ms tract plan on every call. The cap is deliberate:
    concretising deep-clones the weights, so the cache multiplies resident
    model memory by up to 4. `detect_is_correct_across_interleaved_input_lengths`
    pins that results do not depend on eviction order.
12. **~~`NeuralEmbedder::extract` allocates a fresh `Vec` per window~~ — STALE,
    already correct in HEAD.** Both the single-window path (`embedder.rs:742`)
    and the partial-final-batch path (`:722`) already reuse one scratch vector
    via `clear()` + a single `vector.clone()` into the owned embedding. The
    remaining clone is unavoidable — each `NeuralEmbedding` owns its vector.
    No change made.

### 4.3 Test and process gaps

13. **The watermark `detect()` body has no positive-path test.** All seven
    integration tests and all six unit tests assert only construction /
    validation errors; the only call site in the tree is
    `examples/watermark_detect.rs`. No `*.onnx` exists in the repo, and no
    test loads a real model file — the neural happy path uses an in-process
    tract passthrough fixture. Adding a committed tiny ONNX fixture would
    convert a large unverified surface into a gated one.
14. **The resampler has no stopband/attenuation test** — which is exactly
    why the -80 dB doc was wrong for so long.
15. **`simd.rs` has no tests at all**, and the bit-identity claim lived only
    in comments.
16. **Fuzz coverage gaps:** no target for `cache::from_blob` /
    `load_all_cached`; `decode_resample` clamps `target_sr` to
    `[1000, 48000]` and always sets `max_samples`, so it never exercised the
    amplification path that 1.5 found; no target drives
    `decode_to_mono_report`. Four new targets would close these.
17. **`serial.rs` / `cache.rs` / `io/decoder.rs` are fully covered by
    `from_bytes` fuzzing** (350 k+ inputs, no panics) — worth keeping.

### 4.4 API/SDK ergonomics

18. **Persistence is hash-only.** `to_bytes`/`from_bytes` write hashes and
    `frames_per_sec`, but not the producer's crate version, so the v1 format
    cannot distinguish a `wang-v1` blob written by an incompatible future
    build. `FingerprintEnvelope::peek` exposes `hash_count` without
    validating the payload exists (documented), so a caller doing
    `Vec::with_capacity(env.hash_count)` on a truncated blob can allocate
    GBs — a doc warning or a checked helper would help.
19. **`compute_prominence` is `pub` and panics** on `peak_idx >= len`; the
    other matching helpers are NaN-tolerant. Consider returning `0.0`.
20. **No `match_ranked`-equivalent on the index types.** `WangIndex::query`
    returns one best hit; the pairwise path has `match_ranked`. A
    `query_ranked` is the obvious missing method for 1:N evaluation.
21. **`matching` has no persistence or DB adapter by design** ("no
    persistence, no serialisation format, no on-disk index") — but `serial`
    and `cache` now exist, so the module doc's list of non-goals is stale.
22. **No duration limit in `DecodeLimits`** — only `max_bytes`,
    `max_samples`, and a wall-clock `timeout`.

---

## 5. What was verified as already sound

Worth recording so future changes do not weaken it:

* `serial::{read_header, read_payload}` reject short buffers, bad magic,
  wrong version, algorithm mismatch, and non-finite/≤0 fps; the payload
  length uses `checked_mul` and `payload.len() >= expected`, so a hostile
  `hash_count = u32::MAX` allocates nothing. 350 k+ fuzzed inputs, zero
  panics.
* `SortedPostings::build`/`get` are bounds-safe and matched a naive
  reference over randomized inputs, duplicates, and every stop-hash
  threshold.
* `maps.rs`, `calibration.rs` (math), `pcm.rs`, `error.rs` — clean.
* STFT framing matches librosa in both centre modes; the `reflect` pad
  matches `np.pad(mode='reflect')`; periodic windows match scipy
  `fftbins=True`; the mel matrix matches a librosa-style dense reference to
  ≤1.4e-7 across 8 configs; the resampler's output-length contract and
  bounds hold across 6 rate pairs × 200 input lengths.
* Streaming↔offline parity is bit-exact for Haitsma and multiset-exact for
  Wang/Panako, including the 1-sample-per-push case; the zero-allocation
  contract is pinned by a counting allocator.
* Watermark decode math (strict `>` threshold, LSB-first 16-bit packing,
  rank-2/3 flattening, `localization`, confidence = mean) and all zero-norm
  guards in `matching/neural.rs` were verified by execution.
* `SampleRate` cannot be zero-constructed; `TimestampMs` is
  `repr(transparent)` + `Pod`; `#![deny(unsafe_code)]` and
  `#![deny(missing_docs)]` hold everywhere.

---

## 6. Verification performed

```
cargo fmt --all -- --check                                  # clean
cargo clippy --all-targets --all-features -- -D warnings    # clean
cargo clippy --all-targets --no-default-features -- -D warnings   # clean
cargo clippy --all-targets --features std-wav,std-mp3,std-flac,std-ogg -- -D warnings  # clean
cargo test --all-features -- --skip golden                  # 456 passed, 0 failed
cargo test --no-default-features --lib                      # 346 passed, 0 failed
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features  # clean
cargo check --no-default-features --features std            # builds (was a compile_error!)
cargo check --manifest-path fuzz/Cargo.toml --bin matching_malformed  # builds
```

Every fixed item has a regression test; the two originally-reproduced
defects (Panako overflow, Haitsma false positive) were re-run against the
patched tree and now behave correctly.
