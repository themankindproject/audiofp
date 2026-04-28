# Roadmap to Production-Grade

This document inventories everything that would need to be done to take
`audiofp` from "0.2.1, useful but minimal" to "production-grade SDK that
you'd happily depend on at scale". Items are grouped by area and tagged
with rough effort and priority.

Legend:

- **Priority:** P0 = blocks "production-ready" claim, P1 = important but
  not blocking, P2 = nice-to-have, P3 = research-grade
- **Effort:** S = hours, M = days, L = weeks, XL = months

---

## 1. Major Features

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 1.1 | ~~**Neural fingerprinter** (`audiofp::neural`)~~ | ✅ done | — | Shipped as a generic ONNX log-mel embedder (user-supplied model). Build-once-runnable, `embed_window_into` zero-alloc primitive, `try_push_with` zero-alloc streaming, passthrough test fixture for end-to-end coverage. See § 1.1.1 for deferred follow-ups. |
| 1.1.1 | **Neural: batched offline inference** | P1 | M | Currently `extract` runs `runnable.run()` once per analysis window. Most ONNX embedders accept a batch dim; concretising input as `[N, n_mels, n_frames]` and running once over stacked windows is a 5–20× win on small models where per-call overhead dominates. Design: add `batch_size` to `NeuralEmbedderConfig` (default 1 for current behaviour), build runnable with that fixed batch dim, `extract` gathers batch_size windows of log-mel into one tensor, slices the `[batch_size, embedding_dim]` output back out. Streaming stays per-window. |
| 1.1.2 | **Neural: SIMD log-mel matvec** | P2 | M | The dominant front-end cost is the scalar `acc += w * p` loop in `MelFilterBank::log_mel_from_power` — at default settings, ~7 ms for one second of audio (47 frames × 128 mels × 513 bins of MAC). f32x4 manual unroll under `cfg(target_arch = "x86_64")` + `std::arch::x86_64` should give 2–4×. Crate-wide DSP improvement, not embedder-specific. |
| 1.1.3 | **Neural: skipped speculative perf items** | — | — | Decided against: (a) `tract::SimpleState` reuse for the runnable — the strided write + L2 norm together are <11 µs vs ~7 ms front-end + ms-scale inference, so even halving them saves <0.1 % wall time. (b) `unsafe get_unchecked` on the strided tensor write — same 8 µs ceiling, and rustc/LLVM elides the bounds checks on `dst[m * n_frames + f]` already given the loop bounds. Bench evidence in `benches/neural_frontend.rs`. |
| 1.2 | **Mic capture orchestrator** (`audiofp::stream`) — `cpal` capture + bounded ring buffer + `Pipeline<F: StreamingFingerprinter>` | P0 | M | Lets users wire microphone → realtime fingerprints in a few lines. |
| 1.3 | **Constant-Q transform** (`dsp::cqt`) — Brown-Puckette cascaded downsampling | P2 | M | Useful for pitch-aware front-ends and the neural head. Defer until a downstream consumer needs it. |
| 1.4 | **Multi-channel processing** — first-class stereo / surround handling | P1 | M | Currently mono-only. Real-world audio is rarely mono; the file decoder downmixes but the fingerprinters can't take advantage of the extra information. |
| 1.5 | **Voice activity detection (VAD)** — gate fingerprinting on speech presence | P2 | S–M | Useful for podcast / call recording deduplication. |
| 1.6 | **Music / speech / silence classifier** | P2 | M | Pre-route audio to the right fingerprinter or skip silence entirely. |
| 1.7 | **Cover / remix detection** | P3 | L | Built on the neural head; needs a chroma / harmonic-analysis layer. |
| 1.8 | **Self-supervised fingerprint learning** | P3 | XL | Train your own ID head from unlabeled audio; reduces dep on Meta's model. |

---

## 2. Performance & Scale

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 2.1 | **Embedded FFT swap** — replace `rustfft` with `microfft` behind a feature flag for true Cortex-M support | P1 | M | Today the `no_std` build only runs on hosted targets because `rustfft → num-traits/std` is unconditional. |
| 2.2 | **Hand-tuned SIMD** for the windowed-multiply and log-power loops | P2 | M | `realfft` already SIMD-accelerates the FFT; the surrounding pre/post loops are scalar. AVX2 / NEON intrinsics would close the gap. |
| 2.3 | **GPU batch fingerprinting** via `wgpu` for catalog-scale enrollment | P2 | L | A 10M-track ingestion run would benefit from offloading STFT + peak-pick to the GPU. |
| 2.4 | **Async API surface** for non-blocking decode + extract | P2 | M | Currently every API is sync. Async would integrate cleanly with `tokio` / `async-std` users. |
| 2.5 | **Multi-threaded batch helper** — `fingerprint_batch_parallel` using `rayon` | P1 | S | Trivial to add; saves users having to wire up rayon themselves. |
| 2.6 | **Per-platform tuning profiles** for Apple Silicon, x86_64 AVX2/AVX-512, ARMv8 | P2 | M | Conditional compilation paths to pick the optimal kernel per host. |
| 2.7 | **Memory-mapped fingerprint storage** helpers | P2 | S | `bytemuck::Pod` types are already cast-safe; expose `Vec<WangHash> ↔ &[u8]` helpers. |
| 2.8 | ~~**Bounded-memory streaming buffer trim**~~ | ✅ done | — | Stale entry; the `accumulated` field referenced here was removed in the v0.2.0 streaming overhaul. Confirmed by audit: every streaming field is bounded (`sample_carry < n_fft`, rolling `spec` capped at `2·neighborhood + 1` rows, `bucket_pending` cleaned by `finalize_bucket`, `pending_anchors` drained by `emit_finalized_anchors`, Haitsma `pending` `mem::take`'d per push). Memory-bound regression tests now lock these invariants in (`streaming_state_stays_bounded_under_long_input` × 3). |
| 2.8.1 | ~~**Streaming hot-path: per-frame `drain` + per-frame `Vec::clone`**~~ | ✅ done | — | Adjacent perf issue found while auditing 2.8. The streaming `push` loops were doing `sample_carry.drain(0..HOP)` *per frame* (O(frames × buffer)) and Wang/Panako were `frame_scratch.clone()` *per frame* to break a borrow. Fixed: offset cursor + single drain at end of push; `append_frame_scratch_row` method using disjoint field borrow. **Haitsma large_chunk_1s: 10.4 ms → 7.78 ms (−25%)** at the bench-driven ≥ 5% bar. Wang/Panako changes are within bench noise at the default config (peak picking dominates per-frame cost) but are kept on correctness grounds: the drain becomes O(N) instead of O(N²) per push. The win **would** show on configs where peak picking is cheaper — e.g., `peaks_per_sec = 5` instead of 30, smaller `target_zone_t`, or a bigger `WANG_N_FFT`. Bench: `benches/streaming.rs`. |

---

## 3. Robustness & Testing

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 3.1 | **Real codec corpus** with quantitative robustness numbers | P0 | M + corpus | A held-out CC0 corpus + ffmpeg round-trips to MP3@128k, AAC@128k, Opus@32k, etc. Replaces the synthetic robustness tests with measured "≥ 60 % MP3 hash overlap" claims. |
| 3.2 | **`cargo-fuzz` harness** for the decoder and watermark wrapper | P1 | S–M | Symphonia and Tract are fuzzed upstream, but our wrappers' parsing of return values, shape mismatches, and edge cases aren't. |
| 3.3 | **Property tests for all hash bit-packing functions** | P1 | S | We have 4 properties; add round-trip pack/unpack tests for `WangHash`, `PanakoHash`, and Haitsma frame bit positions. |
| 3.4 | **Cross-platform CI matrix** — Linux + macOS + Windows | P0 | S | Currently CI runs only on `ubuntu-latest`. |
| 3.5 | **MSRV CI job** alongside stable | P1 | S | Run the test suite on the pinned MSRV (1.85.0) to catch accidental usage of newer-stable features. |
| 3.6 | **Big-endian + 32-bit target tests** | P2 | S | Probably "just works" because `bytemuck::Pod` types are POD, but worth verifying. |
| 3.7 | **Memory leak tests** via `miri` (interpreter) and Valgrind | P1 | M | Catches `unsafe` bugs and FFI / Drop issues. |
| 3.8 | **Concurrency stress tests** with `loom` | P2 | M | If we add async / threaded helpers, model-check them. |
| 3.9 | **Mutation testing** (`cargo-mutants`) | P2 | M | Measures whether our tests actually catch behavioural regressions. |
| 3.10 | **Code coverage** tracked over time | P1 | S | A coverage badge + per-PR coverage diff via `cargo-llvm-cov` or Codecov. |
| 3.11 | **Snapshot tests over real CC0 audio** | P1 | S–M | Today's regression goldens are synthetic. Real audio catches more bugs. |
| 3.12 | **Adversarial input stress**: extremely long inputs, all-NaN buffers, silent + tiny tones, etc. | P1 | S | Verify no panics, no quadratic slowdowns. |
| 3.13 | **Concurrency safety audit** — confirm `Send + Sync` boundaries are actually safe | P1 | S | Add `static_assertions` for the trait bounds we promise. |

---

## 4. Documentation & Developer Experience

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 4.1 | **mdBook documentation site** with tutorials | P1 | M | Per-algorithm deep-dives, end-to-end music-ID tutorial, deployment recipes. Hosted on GitHub Pages. |
| 4.2 | **Interactive WASM playground** | P2 | M | Drag-and-drop an audio file in a browser, see hashes computed live. Best discovery tool. |
| 4.3 | **Video walkthrough** of the SDK end-to-end | P2 | S | 5-minute demo for the README. |
| 4.4 | **API stability policy** documented in CONTRIBUTING | P0 | S | What "0.x → 1.0" means, deprecation policy, MSRV bumps. |
| 4.5 | **Migration guides** for breaking changes (0.1 → 0.2 already happened) | P1 | S each | Explicit "what changed and how to update". |
| 4.6 | **Algorithm whitepapers** linked to readable summaries | P2 | S | Pointers to Wang/Panako/Haitsma papers + plain-English summaries of what each captures. |
| 4.7 | **Comparison with `chromaprint` / `dejavu` / `audfprint`** with reproducible benchmarks | P1 | M | Quantitative head-to-head on the same corpus, not the README's table of "Yes / No" features. |
| 4.8 | **`docs.rs` examples module** that compiles | P1 | S | Per-algorithm example modules visible in docs.rs's sidebar. |
| 4.9 | **Plain-English error messages** for common misuses (wrong rate, short audio, missing model) | P2 | S | Already pretty good; review one more pass. |
| 4.10 | **`cargo-deadlinks` in CI** | P2 | S | Catches broken intra-doc links before they ship. |

---

## 5. Distribution & Bindings

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 5.1 | **CLI binary** (`audiofp-cli`) for batch processing | P1 | M | `audiofp enroll <dir>`, `audiofp match <query> --db <path>`, `audiofp inspect <hash-file>`. |
| 5.2 | **Python bindings** via `pyo3` + `maturin` | P1 | L | The biggest user base for audio tooling lives in Python. |
| 5.3 | **Node.js bindings** via `napi-rs` | P2 | M | Less critical than Python but valuable for web stacks. |
| 5.4 | **C ABI** via `cbindgen` | P2 | M | Lets non-Rust hosts (Go, Swift, …) consume `audiofp` directly. |
| 5.5 | **WASM build** via `wasm-bindgen` | P2 | M | Pairs with item 4.2 (playground) and lets users fingerprint in-browser. |
| 5.6 | **Docker image** for the CLI | P2 | S | One-liner deploy for batch jobs. |
| 5.7 | **Homebrew formula** for the CLI | P3 | S | Discoverability for macOS users. |
| 5.8 | **Cloud Run / Lambda templates** for serverless fingerprinting | P3 | M | Reference deployments for common cloud platforms. |
| 5.9 | **Versioned wire format** for hash storage (JSON + binary) | P1 | S | Documents the on-disk layout users can rely on. |

---

## 6. Operations & Observability

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 6.1 | **Structured logging** via `tracing` behind a feature flag | P1 | S | Spans on `extract`, `push`, `detect` — invisible by default, opt-in via `--features tracing`. |
| 6.2 | **Metrics hooks** for processing time, hash counts, error counts | P2 | S | `metrics` crate facade so users plug in Prometheus / StatsD / OTel. |
| 6.3 | **OpenTelemetry integration** example | P2 | S | Shows how to wire `tracing` → OTLP exporter. |
| 6.4 | **Memory limits** on input audio (cap allocations) | P1 | S | Decoding a maliciously-crafted 4 GB file shouldn't OOM the host. |
| 6.5 | **Decode timeout** (cap wall time on a single file) | P1 | S | Symphonia can hang on adversarial inputs. |
| 6.6 | **Sandboxed decoder mode** (subprocess) | P3 | M | Maximum-isolation option for hostile environments. |
| 6.7 | **`audiofp inspect` CLI subcommand** to debug a fingerprint | P2 | S | Hex-dump hashes, summarise per-second density, plot Δt histograms. |

---

## 7. Security

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 7.1 | **`SECURITY.md`** with responsible disclosure address | P0 | S | Required for serious adoption. |
| 7.2 | **`cargo-audit` in CI** for known-vulnerable deps | P0 | S | Catches advisories on every push. |
| 7.3 | **`cargo-deny check` in CI** | P0 | S | Already have `deny.toml`; just need it wired into the workflow. |
| 7.4 | **Threat model** for the SDK and CLI (audio-as-input, model-as-input, hash-as-output) | P1 | S–M | Documents what attacks the SDK does and doesn't defend against. |
| 7.5 | **Constant-time hash comparison helper** | P2 | S | If users want to use audio fingerprints in security-sensitive comparisons. |
| 7.6 | **Signed releases** (`cargo-crev` / minisign) | P2 | S | Supply-chain hardening for crates.io users. |
| 7.7 | **SBOM generation** for releases | P2 | S | Compliance requirement for some enterprises. |

---

## 8. Ecosystem Integrations

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 8.1 | **Hash database adapters** for FAISS, hnswlib, RocksDB, sqlite, Postgres, Redis | P1 | S each | Most users will store their hashes somewhere; canonical adapter crates remove that integration toil. |
| 8.2 | **`gstreamer-rs` plugin** for piping audio in | P3 | M | Lets `audiofp` run inside any GStreamer pipeline. |
| 8.3 | **`hound` / `creek` / `kira` interop docs** | P2 | S | Show how to pipe audio from popular Rust audio crates into `audiofp`. |
| 8.4 | **`rodio` / `ffmpeg-rs` examples** | P2 | S | Practical recipes for the common alternative file decoders. |

---

## 9. Project & Community

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 9.1 | **`CODE_OF_CONDUCT.md`** | P0 | S | Required for community contributions. |
| 9.2 | **GitHub issue + PR templates** | P0 | S | Reduces back-and-forth on triage. |
| 9.3 | **Public roadmap** (this file or a GitHub Project) | P1 | S | Lets users plan around upcoming releases. |
| 9.4 | **Release notes automation** (`release-plz` or `cargo-release`) | P1 | S | Removes manual error from version bumps. |
| 9.5 | **Conventional commits** enforcement | P2 | S | Already follow the convention; a CI lint would make it strict. |
| 9.6 | **Discussion forum** (GitHub Discussions or Discord) | P2 | S | Lower-friction help channel than issues. |
| 9.7 | **Conference talk** at Rust / RustAudio / ICASSP | P3 | M | Best discovery vehicle for an SDK. |
| 9.8 | **Blog post writeups** of the perf overhaul, incremental streaming, etc. | P2 | M | Long-form content that positions the project. |
| 9.9 | **Sponsorship / funding** path (GitHub Sponsors, OpenCollective) | P3 | S | Sustainability for non-trivial maintenance. |

---

## 10. Research & Long-Horizon

| # | Item | Priority | Effort | Notes |
|---|------|----------|--------|-------|
| 10.1 | **Train a from-scratch neural fingerprinter** to remove dependence on Meta's AudioSeal weights | P3 | XL | Independent ML pipeline with our own data + training code. |
| 10.2 | **Time-stretch / pitch-shift robustness** beyond Panako's ±5 % | P3 | L | Either via CQT + chroma or a neural domain-adaptation head. |
| 10.3 | **Adaptive-bitrate fingerprinting** (one fingerprint that survives 8 kbps mono → lossless) | P3 | L | Active research area. |
| 10.4 | **Federated / privacy-preserving matching** | P3 | XL | Match audio without transmitting it. |
| 10.5 | **Continuous learning** — update fingerprints as new content lands | P3 | L | For very-long-running deployments. |

---

## What "production-grade" means concretely

A reasonable bar:

- ✅ All P0 items closed (5 items: real codec corpus, cross-platform CI,
  API stability policy, security/CoC/templates files, cargo-audit/deny
  in CI, plus the two big features).
- ✅ Most P1 items closed (≈ 20 items).
- ✅ At least one binding (Python, since that's where the audience is).
- ✅ Public benchmarks vs `chromaprint` showing competitive numbers.
- ✅ One year of stable releases (semver kept).
- ✅ At least 100 GitHub stars and 5 production users willing to be
  named in a "who's using" list.

`audiofp 0.2.1` ships **none of the P0 items above** — it's a credible
**alpha-grade** SDK with strong perf characteristics and rigorous
correctness for what it does. The path to production-grade is the
P0 column, then P1, then bindings.
