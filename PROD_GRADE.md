# audiofp — Detailed Work Plan (Prod-Grade)

> Consolidated backlog from the Jul 2026 full audit, open GitHub issues,
> and [`future.md`](future.md). This is the actionable “what we can do”
> list — prioritized for making `audiofp` a production-grade SDK.
>
> **Crate today:** `audiofp` **v0.3.7** · MSRV **1.93.0** · MIT  
> **Repo:** https://github.com/themankindproject/audiofp  
> **Status:** Classical core is strong (bit-exact streaming, goldens, CI
> audit/deny). Gaps are hardening, API productization, measured
> robustness, and matching/storage UX.

---

## Legend

| Tag | Meaning |
| --- | ------- |
| **P0** | Blocks a credible “production-ready” claim |
| **P1** | Important for serious adopters |
| **P2** | Nice-to-have / scale polish |
| **P3** | Research / long-horizon |
| **S / M / L / XL** | Hours / days / weeks / months |
| **NB** | Non-breaking (can ship in 0.3.x) |
| **BR** | Breaking (target **0.4.0**) |
| **Audit** | From Jul 2026 security/architecture audit |
| **#N** | Open GitHub issue |

---

## What’s already solid (do not re-litigate)

- Bit-exact offline ≡ streaming for Wang / Panako / Haitsma (incl. 1-sample chunks)
- Golden regression hashes; property tests; streaming state bound tests
- Streaming peak path ~16× faster since 0.3.4 (`IncrementalPeakDetector`)
- Feature-gated Symphonia (`std`) and Tract (`neural` / `watermark`)
- `cargo audit` + `cargo-deny` in CI; license/source policy
- `fingerprint_batch_parallel` behind `rayon`
- Limited, documented `unsafe` (SIMD window + Tract tensor init); no FFI surface
- Docs pack (Jul 2026): zero-deps quick start, USAGE async/batching, examples
  `dsp_starter` / `neural_embed` / `watermark_detect`, localization contract

---

## 1. Architectural improvements

### 1.1 Trust boundaries & fail-safe APIs (P0)

| ID | Item | Effort | Break? | Source | Notes |
| -- | ---- | ------ | ------ | ------ | ----- |
| A1 | **Input budget API** — `max_samples` / `max_duration` on decode, extract, watermark detect, neural extract | S–M | NB if optional (default unlimited) | Audit · #68 · future 6.4 | Without this, untrusted uploads can OOM. Prefer new capped helpers + optional fields with default `None`. |
| A2 | **Max push chunk size** for streaming | S | NB | Audit | Steady-state carry is bounded; one hostile `push(&[f32; huge])` still allocates ~chunk. Reject or internal-chunk. |
| A3 | **Decode wall-clock timeout** | S–M | NB | future 6.5 | Symphonia can hang on adversarial inputs. |
| A4 | **ONNX model trust helpers** — max file size, optional content hash pin, “models must be trusted” docs | S | NB | Audit | Tract parses arbitrary ONNX = deserializer surface. |
| A5 | **NaN / Inf PCM policy** — reject or sanitize at extract/push entry | S | NB* | Audit · future 3.12 | *Rejecting values that previously “worked” is a soft behavior change; document clearly. |
| A6 | **Decoder integrity mode** — opt-in fail on corrupt packets instead of silent skip | S | NB | Audit | Today recoverable Symphonia errors are skipped; fingerprints may be incomplete with `Ok`. |
| A7 | **`SECURITY.md` + threat model** (audio / files / ONNX / hashes-as-output) | S | NB | future 7.1, 7.4 | Fingerprints are perceptual, not MACs. |
| A8 | **Wire up / keep `BufferOverrun`** or remove dead variant | S | NB or BR | Audit | Variant exists, unused in production paths. |

### 1.2 Config & constructor hygiene (P0/P1)

| ID | Item | Effort | Break? | Source | Notes |
| -- | ---- | ------ | ------ | ------ | ----- |
| A9 | **Panako `target_zone_t == 0` underflow** — clamp or `try_new` | S | NB if clamp / additive `try_new` | Audit | `as u32 - 1` wraps to `u32::MAX`. |
| A10 | **Cap public config knobs** (`fan_out`, `n_mels`, `n_fft`, `window_secs`, sinc taps) | S | NB if soft caps + `try_*` | Audit | Extreme configs → OOM/CPU. |
| A11 | **Replace `assert!` in library constructors with `Result`** (Haitsma, mel, resample, STFT) | M | BR for signatures that today return `Self` | #8 · Audit | Prefer `try_new` alongside deprecated panic path for one minor, then remove. |
| A12 | **SIMD window helpers: length asserts** | S | NB | Audit | Current call sites sound; asserts harden against future misuse. |
| A13 | **Model path TOCTOU** — drop `exists()`; map load errors | S | NB | Audit | Wrong error variant today, not privilege escalation. |
| A14 | **Decoder `n_chans == 0` guard** | S | NB | Audit | Avoid `/ 0` → Inf samples. |

### 1.3 Trait & type architecture (0.4.0) (P0/P1)

| ID | Item | Effort | Break? | Source | Notes |
| -- | ---- | ------ | ------ | ------ | ----- |
| A15 | **`StreamingFingerprinter::push` → `Result`** | M | BR | #63 | Neural `push` panics on Tract errors; classical stays infallible in practice. |
| A16 | **Fix dB vs linear `min_magnitude`** + rename to `min_magnitude_db` | S–M | BR | #2 · #62 | Correctness bug for Wang/Panako peak floors. |
| A17 | **`Fingerprinter::required_sample_rate() → SampleRate`** | S | BR | #61 | Type-safe rates end-to-end. |
| A18 | **Drop `AudioBuffer`; take `&[f32]` + `SampleRate`** | M | BR | #65 | Less wrapper ceremony. |
| A19 | **Hash timestamps → `TimestampMs` not raw `u32` frames** | M | BR | #66 | Align offline hash fields with streaming emission. |
| A20 | **Flat crate-root re-exports of major types** | S | BR (name collisions / rustdoc) | #64 | Ergonomics for apps. |
| A21 | **`audiofp::prelude` module** | S | NB | #14 | Can ship before 0.4. |
| A22 | **`fingerprint_file` one-shot helper** | S | NB | #15 | Decode + resample + extract. |
| A23 | **Split `std` into per-codec sub-features** | M | BR | #60 | Smaller binaries, less parser surface. |
| A24 | **STFT `process_frame*` return `Result`** | S | BR | #8 | No panic on length mismatch for public DSP. |
| A25 | **API stability policy** in CONTRIBUTING | S | NB | future 4.4 | What may break in 0.x vs 1.0; MSRV policy. |
| A26 | **0.4 migration guide** | S | NB | future 4.5 | Required when shipping A15–A20. |

### 1.4 Product / system architecture (P1)

| ID | Item | Effort | Break? | Source | Notes |
| -- | ---- | ------ | ------ | ------ | ----- |
| A27 | **Mic capture orchestrator** — `cpal` + bounded ring + `Pipeline<F: StreamingFingerprinter>` | M | NB (new module) | future 1.2 | Real-time product path. |
| A28 | **Versioned hash wire format** (JSON + binary) | S–M | NB | future 5.9 | Durable indexes across releases. |
| A29 | **Hash DB adapters** (sqlite / RocksDB / Redis / FAISS / hnsw) | S each | NB (new crates or features) | future 8.1 | Matching is out of scope today — biggest adoption gap. |
| A30 | **True `no_std` FFT** — `microfft` behind feature | M | NB | future 2.1 | Host-only `no_std` is half a promise. |
| A31 | **Multi-channel / stereo policy** | M | Possibly BR | future 1.4 | Beyond silent downmix. |
| A32 | **Neural batched offline inference** (`batch_size` in config) | M | NB (default 1) | future 1.1.1 | Architecture for throughput; see §2. |
| A33 | **Observability** — `tracing` feature spans on extract/push/detect | S | NB | future 6.1 | Invisible by default. |
| A34 | **CLI binary** (`enroll` / `match` / `inspect`) | M | NB | future 5.1 | Ops / demos. |
| A35 | **Python bindings** (pyo3 + maturin) | L | NB | future 5.2 | Where most audio tooling lives. |

### 1.5 Architecture diagram (target trust model)

```text
                    ┌─────────────────────────────────────┐
                    │           Caller / service           │
                    │  (must bound size, trust models)     │
                    └───────────────┬─────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
   │ Audio files  │         │ PCM f32      │          │ ONNX models  │
   │ (untrusted)  │         │ (untrusted)  │          │ (untrusted)  │
   └──────┬───────┘         └──────┬───────┘          └──────┬───────┘
          │ caps+timeout           │ NaN policy               │ size+hash
          ▼                        ▼                          ▼
   ┌──────────────┐         ┌──────────────┐          ┌──────────────┐
   │ io::decode   │────────▶│ classical /  │◀─────────│ neural /     │
   │              │         │ streaming    │          │ watermark    │
   └──────────────┘         └──────┬───────┘          └──────────────┘
                                   │
                                   ▼
                            Pod hashes / embeddings
                            (not cryptographic)
                                   │
                                   ▼
                         wire format → DB adapters
```

---

## 2. Performance optimizations

### 2.1 Highest ROI next

| ID | Item | Effort | Break? | Expected win | Source |
| -- | ---- | ------ | ------ | ------------ | ------ |
| P1 | **Batched neural ONNX** (`batch_size`) | M | NB | 5–20× on small models where `run()` overhead dominates | future 1.1.1 |
| P2 | **SIMD mel matvec** in `MelFilterBank::log_mel_from_power` | M | NB | 2–4× front-end (~7 ms per 1 s audio @ defaults) | future 1.1.2 |
| P3 | **Soft input caps + early reject** | S | NB | Protects p99 / avoids OOM death spirals | Audit · #68 |
| P4 | **Streaming / progressive decode** | M | NB | Avoid full-file `Vec<f32>` for long tracks | Audit |
| P5 | **Cap watermark/neural `T` + reuse typed plans aggressively** | S | NB | Avoid huge tensors + plan rebuild thrash | Audit |

### 2.2 Medium ROI

| ID | Item | Effort | Break? | Notes | Source |
| -- | ---- | ------ | ------ | ----- | ------ |
| P6 | SIMD log-power / more pre-FFT loops | M | NB | FFT already SIMD via `realfft`; surrounding scalar | future 2.2 |
| P7 | mmap / bytemuck cast helpers for hash slices | S | NB | Zero-copy enroll I/O | future 2.7 |
| P8 | Per-platform tuning profiles (Apple Silicon, AVX-512) | M | NB | Conditional kernels | future 2.6 |
| P9 | Watermark detect: fixed-length batching guidance + helper | S | NB | Plan cache already keyed by `T` | detector docs |

### 2.3 Scale / later

| ID | Item | Effort | Break? | Notes | Source |
| -- | ---- | ------ | ------ | ----- | ------ |
| P10 | GPU batch fingerprinting (`wgpu`) | L | NB | Only if catalog enroll hits 10⁶–10⁷ tracks | future 2.3 |
| P11 | Async-native decode/extract API | M | NB or BR | Sync + `spawn_blocking` is documented; native async is polish | future 2.4 |

### 2.4 Already done (do not reinvest)

| Item | Result |
| ---- | ------ |
| Incremental streaming peaks (0.3.4) | ~94% less wall time on Wang/Panako small-chunk benches |
| Streaming drain O(N) vs per-frame drain | Haitsma ~−25% on large chunks; correctness for Wang/Panako |
| Neural build-once Tract runnable | Large win vs per-call optimize |
| Allocation-free streaming hot path after warmup | Locked by bound tests |

### 2.5 Bench commands (reproduce)

```bash
cargo bench --bench extract
cargo bench --bench streaming
cargo bench --bench neural_frontend --features neural
```

Offline numbers in README are from **v0.2.0**; streaming wins from **v0.3.4**. Always re-bench on target hardware.

---

## 3. Truly production-grade (evidence + ops)

Production-grade ≠ more algorithms. It means **bounded resources, fail-safe APIs, measured robustness, multi-OS CI, and a matching story**.

### 3.1 Must-have (P0)

| ID | Item | Effort | Break? | Source | Done? |
| -- | ---- | ------ | ------ | ------ | ----- |
| G1 | Real **codec corpus** + published overlap numbers (MP3/AAC/Opus) | M + corpus | NB | future 3.1 | ❌ |
| G2 | **Cross-platform CI** (Linux + macOS + Windows) | S | NB | #29 · future 3.4 | ❌ |
| G3 | **Fuzz all 7 targets** in CI (not only `streaming_wang_equiv` 10s) | S | NB | #28 | ✅ |
| G4 | Fuzz harnesses for **decoder + watermark wrappers** | S–M | NB | future 3.2 · #84 | ❌ |
| G5 | **OOM / allocation limits** (#68) | S–M | NB | Audit | ✅ |
| G6 | Decode **timeout** | S–M | NB | future 6.5 · #77 | ❌ |
| G7 | **`SECURITY.md` + threat model** | S | NB | future 7.1, 7.4 · #74 | ❌ |
| G8 | **CoC + issue/PR templates** | S | NB | future 9.1, 9.2 · #83 | ❌ |
| G9 | **API stability policy** + 0.4 migration guide | S | NB | future 4.4, 4.5 · #85 | ❌ |
| G10 | **Miri** (or equivalent) on `unsafe` paths | M | NB | future 3.7 · #89 | ❌ |
| G11 | Keep **cargo-audit / cargo-deny** green | — | NB | future 7.2, 7.3 | ✅ |

### 3.2 Should-have (P1)

| ID | Item | Effort | Source |
| -- | ---- | ------ | ------ |
| G12 | Public benches vs **chromaprint** on same corpus | M | future 4.7 |
| G13 | **Python bindings** | L | future 5.2 |
| G14 | **CLI** for enroll/match/inspect | M | future 5.1 |
| G15 | Coverage tracking (`cargo-llvm-cov` / Codecov) | S | future 3.10 |
| G16 | Adversarial / NaN / huge-input stress tests | S | future 3.12 |
| G17 | Property tests for all hash pack/unpack paths | S | future 3.3 |
| G18 | Snapshot tests on real CC0 audio | S–M | future 3.11 |
| G19 | `tracing` feature | S | future 6.1 |
| G20 | Versioned wire format for hashes | S | future 5.9 |
| G21 | Constant-time hash compare helper (only if auth-ish use) | S | future 7.5 |
| G22 | Fix CONTRIBUTING **MSRV** (still says 1.85; reality 1.93) | S | Audit |

### 3.3 Nice / later (P2–P3)

| Area | Items |
| ---- | ----- |
| Docs / DX | mdBook site, WASM playground, algorithm whitepapers, deadlinks CI |
| Bindings | Node (napi), C ABI (cbindgen), WASM |
| Ops | Metrics facade, OTel example, SBOM, signed releases |
| Research | Cover/remix detection, own neural trainer, federated matching |
| Community | Discussions, release automation, sponsors |

### 3.4 Definition of done (prod-grade bar)

From [`future.md`](future.md) § “What production-grade means”, updated:

- [ ] All **P0** items in this file closed (harden + evidence + security meta)
- [ ] Most **P1** architecture/perf items closed or explicitly deferred with rationale
- [ ] At least one binding (**Python** preferred)
- [ ] Public benchmarks vs chromaprint with reproducible methodology
- [ ] Semver kept for ≥ 1 year of releases; migration guides for breaks
- [ ] Named production users / “who’s using” (aspirational)

**Today (0.3.7):** credible **library-alpha / strong classical core** — not yet the full prod-grade bar above.

---

## 4. Open GitHub issues map

> Epic tracker: [#85](https://github.com/themankindproject/audiofp/issues/85) (0.4.0 + full backlog links).  
> Labels: every issue is tagged `breaking-change` or `non-breaking`; plus `security` / `matching` / `bindings` / `meta` / `ci` / `tests` / `performance` as relevant.

### Shipped / closed (reference)

| # | Title | Maps to |
| - | ----- | ------- |
| [#14](https://github.com/themankindproject/audiofp/issues/14) | `prelude` module | A21 ✅ |
| [#15](https://github.com/themankindproject/audiofp/issues/15) | `fingerprint_file` helper | A22 ✅ |
| [#28](https://github.com/themankindproject/audiofp/issues/28) | Fuzz all 7 targets in CI | G3 ✅ |
| [#38](https://github.com/themankindproject/audiofp/issues/38)–[#55](https://github.com/themankindproject/audiofp/issues/55), [#59](https://github.com/themankindproject/audiofp/issues/59) | Docs pack | ✅ |
| [#68](https://github.com/themankindproject/audiofp/issues/68) | OOM / allocation limits | A1 · G5 ✅ (watermark gap → #81) |

### Phase A — Harden (non-breaking)

| # | Title | Maps to |
| - | ----- | ------- |
| [#74](https://github.com/themankindproject/audiofp/issues/74) | SECURITY.md + threat model | A7 · G7 |
| [#75](https://github.com/themankindproject/audiofp/issues/75) | NaN/Inf PCM policy | A5 |
| [#76](https://github.com/themankindproject/audiofp/issues/76) | Decoder integrity mode | A6 |
| [#77](https://github.com/themankindproject/audiofp/issues/77) | Decode wall-clock timeout | A3 · G6 |
| [#78](https://github.com/themankindproject/audiofp/issues/78) | ONNX model trust helpers | A4 |
| [#79](https://github.com/themankindproject/audiofp/issues/79) | Model path TOCTOU fix | A13 |
| [#80](https://github.com/themankindproject/audiofp/issues/80) | `max_push_samples` Wang/Haitsma/neural | A2 |
| [#81](https://github.com/themankindproject/audiofp/issues/81) | Watermark + config knob caps | A1 · A10 |
| [#82](https://github.com/themankindproject/audiofp/issues/82) | SIMD window asserts + BufferOverrun | A12 · A8 |
| [#83](https://github.com/themankindproject/audiofp/issues/83) | CoC + templates + MSRV 1.93 | G8 · G22 |
| [#84](https://github.com/themankindproject/audiofp/issues/84) | Fuzz decoder + watermark | G4 |
| [#29](https://github.com/themankindproject/audiofp/issues/29) | macOS + Windows CI | G2 |
| [#86](https://github.com/themankindproject/audiofp/issues/86) | `try_new` constructors (additive) | A11 |

### Breaking (bundle into 0.4.0) — epic [#85](https://github.com/themankindproject/audiofp/issues/85)

| # | Title | Maps to |
| - | ----- | ------- |
| [#2](https://github.com/themankindproject/audiofp/issues/2) / [#62](https://github.com/themankindproject/audiofp/issues/62) | dB vs linear magnitude | A16 |
| [#8](https://github.com/themankindproject/audiofp/issues/8) | STFT `Result` | A24 |
| [#60](https://github.com/themankindproject/audiofp/issues/60)–[#66](https://github.com/themankindproject/audiofp/issues/66) | API reshape pack | A15–A20, A23 |
| [#85](https://github.com/themankindproject/audiofp/issues/85) | 0.4 epic + migration guide | A25–A26 · G9 |

### Phase C — Evidence (non-breaking)

| # | Title | Maps to |
| - | ----- | ------- |
| [#87](https://github.com/themankindproject/audiofp/issues/87) | Codec robustness corpus | G1 |
| [#88](https://github.com/themankindproject/audiofp/issues/88) | Chromaprint bakeoff | G12 |
| [#89](https://github.com/themankindproject/audiofp/issues/89) | Coverage + Miri | G15 · G10 |
| [#90](https://github.com/themankindproject/audiofp/issues/90) | Adversarial + CC0 snapshots | G16 · G18 |

### Phase D — Product / scale (non-breaking)

| # | Title | Maps to |
| - | ----- | ------- |
| [#91](https://github.com/themankindproject/audiofp/issues/91) | Versioned hash wire format | A28 · G20 |
| [#92](https://github.com/themankindproject/audiofp/issues/92) | Matcher + sqlite adapter | A29 |
| [#93](https://github.com/themankindproject/audiofp/issues/93) | CLI enroll/match/inspect | A34 · G14 |
| [#94](https://github.com/themankindproject/audiofp/issues/94) | Python bindings | A35 · G13 |
| [#95](https://github.com/themankindproject/audiofp/issues/95) | `tracing` feature | A33 · G19 |
| [#96](https://github.com/themankindproject/audiofp/issues/96) | Batched neural ONNX | P1 |
| [#97](https://github.com/themankindproject/audiofp/issues/97) | SIMD mel matvec | P2 |
| [#98](https://github.com/themankindproject/audiofp/issues/98) | Mic capture orchestrator | A27 |

---

## 5. Suggested delivery phases

### Phase A — Harden (NB, ~2–3 weeks)

Goal: safe to call from a multi-tenant upload service with clear docs.

1. A1–A7, A9–A10, A12–A14 (caps, Panako clamp, SECURITY.md, NaN, integrity mode)
2. G3–G4, G2 (fuzz + CI matrix)
3. G5–G6, G8, G22 (OOM, timeout, templates, MSRV docs)
4. Close docs issues #38–#59 if not already

**Exit:** “Untrusted audio won’t silently OOM us; threat model published.”

### Phase B — API 0.4 (BR, ~2–4 weeks)

Goal: correct, ergonomic traits; one migration guide.

1. A15–A20, A23–A24 (Result streaming, dB fix, types cleanup)
2. A21–A22 if not shipped in A (`prelude`, `fingerprint_file`)
3. A25–A26 (stability policy + migration guide)
4. Semver bump **0.4.0** + CHANGELOG

**Exit:** “No known footguns in public constructors/traits; 0.3→0.4 guide exists.”

### Phase C — Evidence (NB, ~2–4 weeks + corpus)

Goal: numbers you can put on a sales / README slide.

1. G1 codec corpus + robustness tables
2. G12 chromaprint bakeoff
3. G15–G18 coverage + adversarial tests
4. G10 Miri on unsafe

**Exit:** “Measured ≥X% overlap on MP3@128k; competitive with chromaprint on corpus Y.”

### Phase D — Scale UX (NB, ongoing)

Goal: people can ship products without reinventing storage/match.

1. P1–P2 (neural batch + SIMD mel)
2. A28–A29 (wire format + DB adapters)
3. A27, A34–A35 (mic pipeline, CLI, Python)
4. A32–A33 (batch arch, tracing)

**Exit:** “Enroll 100k tracks, match queries, expose Python, observe in prod.”

---

## 6. Explicit non-goals (for now)

- Treating fingerprints as cryptographic authentication
- Bundling AudioSeal / third-party ONNX weights in the crate
- Guaranteeing wire compatibility with Six’s Panako or paper Haitsma bit order
- Bare-metal Cortex-M until `microfft` feature lands (A30)
- GPU path until catalog scale demands it (P10)

---

## 7. Quick reference — top 15 to do next

| Priority | ID | Issue | One-liner |
| -------- | -- | ----- | --------- |
| 1 | A7 | [#74](https://github.com/themankindproject/audiofp/issues/74) | `SECURITY.md` + threat model |
| 2 | A5 / A6 | [#75](https://github.com/themankindproject/audiofp/issues/75) / [#76](https://github.com/themankindproject/audiofp/issues/76) | NaN policy + decoder integrity |
| 3 | A3 | [#77](https://github.com/themankindproject/audiofp/issues/77) | Decode wall-clock timeout |
| 4 | A4 / A13 | [#78](https://github.com/themankindproject/audiofp/issues/78) / [#79](https://github.com/themankindproject/audiofp/issues/79) | ONNX trust + TOCTOU |
| 5 | G2 | [#29](https://github.com/themankindproject/audiofp/issues/29) | macOS + Windows CI |
| 6 | A16 | [#2](https://github.com/themankindproject/audiofp/issues/2) | Fix dB vs linear peak floor (0.4) |
| 7 | A15 | [#63](https://github.com/themankindproject/audiofp/issues/63) | Streaming `push` → `Result` (0.4) |
| 8 | — | [#85](https://github.com/themankindproject/audiofp/issues/85) | 0.4 epic + migration guide |
| 9 | G1 | [#87](https://github.com/themankindproject/audiofp/issues/87) | Codec robustness corpus |
| 10 | A28 / A29 | [#91](https://github.com/themankindproject/audiofp/issues/91) / [#92](https://github.com/themankindproject/audiofp/issues/92) | Wire format + sqlite matcher |
| 11 | A34 / A35 | [#93](https://github.com/themankindproject/audiofp/issues/93) / [#94](https://github.com/themankindproject/audiofp/issues/94) | CLI + Python bindings |
| 12 | P1 / P2 | [#96](https://github.com/themankindproject/audiofp/issues/96) / [#97](https://github.com/themankindproject/audiofp/issues/97) | Neural batch + SIMD mel |
| 13 | G12 | [#88](https://github.com/themankindproject/audiofp/issues/88) | Chromaprint bakeoff |
| 14 | G8 | [#83](https://github.com/themankindproject/audiofp/issues/83) | CoC + templates + MSRV |
| 15 | A2 | [#80](https://github.com/themankindproject/audiofp/issues/80) | Finish `max_push_samples` |

---

## 8. Related docs

| File | Role |
| ---- | ---- |
| [`future.md`](future.md) | Original long-horizon inventory (some items stale vs 0.3.7) |
| [`CHANGELOG.md`](CHANGELOG.md) | What already shipped |
| [`USAGE.md`](USAGE.md) | User-facing API guide |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Dev setup (fix MSRV: G22) |
| Audit canvas | `~/.cursor/projects/.../canvases/audiofp-full-audit.canvas.tsx` |

---

*Last updated: 2026-07-18 — GH backlog #74–#98 opened via orchestrator; §4/§7 synced.*
