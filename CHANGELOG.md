# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/themankindproject/audiofp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/themankindproject/audiofp/releases/tag/v0.1.0
