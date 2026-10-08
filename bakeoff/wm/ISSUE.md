# Add classical (non-neural) audio watermarking: STFT-domain spread spectrum

## Summary

Add a **classical, non-neural audio watermarking subsystem** to `audiofp`, exposed as
`audiofp::watermark::classical`, behind an opt-in `classical-watermark` feature.

The subsystem should embed an invisible, keyed, robust identifier directly into the
signal domain and recover it **blindly** (no reference copy) with only the secret key.
It is not metadata embedding, appended bytes, container tricks, and it does not reuse
Wang / Panako / Haitsma fingerprints.

This issue reports the results of a **measured bakeoff** across three classical
candidate algorithms against a full codec + signal-processing + temporal attack matrix,
and proposes a concrete design for the winner.

> **Scope of this issue:** design + evidence only. No implementation is proposed here
> beyond the API sketch and module layout.

---

## 1. Motivation

`audiofp` already answers *"which recording is this?"* extremely well
(Wang / Panako / Haitsma, streaming + offline, in-memory matching, codec-robustness
infrastructure). It cannot answer *"which identity / provenance / asset ID was
embedded into this copy?"*.

The existing `audiofp::watermark` module is an **AudioSeal/ONNX detector wrapper**.
It detects a watermark produced by an external neural model; it cannot embed one,
and it carries a heavyweight `tract-onnx` dependency.

A classical embedder/detector pair is complementary:

| Capability | Fingerprinting | Classical watermarking |
|---|---|---|
| Question answered | *Which recording?* | *Which ID was embedded?* |
| Requires reference DB | Yes (1:N search) | No (blind, key only) |
| Payload | None (implicit identity) | Explicit configurable bit string |
| Survives "unknown" content | Only if enrolled | Any content, if embedded |
| Dependencies | FFT/DSP only | FFT/DSP only |

The two mechanisms should remain **conceptually and structurally separate**.

---

## 2. What was measured

A dev-only bakeoff harness (`bakeoff/wm/`, committed to the repository for
reproducibility, not part of the crate) compared
three classical transform-domain candidates on a shared corpus with a shared attack
matrix. Everything below is **measured**, not estimated.

### 2.1 Candidates

| # | Candidate | Domain | Spreading | Detector |
|---|---|---|---|---|
| A | `mag_dsss` | STFT **magnitude**, phase-preserving | Keyed bipolar chips over keyed bins, bit index cycling at frame rate | Blind: subtract per-bin long-term mean magnitude (host estimate), correlate residual against chips |
| B | `log_dsss` | STFT **log-magnitude** (dB) | Keyed chips, fixed frame block per bit | Blind: correlate dB magnitude against chips |
| C | `qim_logmag` | STFT **log-magnitude**, quantisation index modulation | Two interleaved lattices, `q = 3 dB` | Blind: nearest-lattice decision per bin |

All three reuse the crate's DSP philosophy: periodic Hann STFT, complex spectra,
weighted overlap-add resynthesis, integer-only keyed PRNG.

### 2.2 Corpus

Six CC-BY / CC0 tracks from the existing `tests/assets/` corpus (the two orchestral
excerpts from `tests/assets/catalog/`), decoded to mono
16 kHz PCM:

| Track | Duration | Content |
|---|---|---|
| `galway.flac` | 15.8 s | Music (Kevin MacLeod, CC-BY 3.0) |
| `freak.flac` | 16.0 s | Music (Kevin MacLeod, CC-BY 3.0) |
| `acidjazz.flac` | 16.0 s | Music |
| `digya.flac` | 16.0 s | Music |
| `grieg_morning.ogg` | 30.0 s | Classical |
| `mozart_figaro_over.ogg` | 30.0 s | Classical |

Payload: **64 bits**, deterministic. Sample rate 16 kHz. STFT: `n_fft = 2048`,
`hop = 512` → **31.25 frames/s**, ~1.5 kB of STFT coefficients per frame.

### 2.3 Attack matrix (32 attacks × 6 tracks = 192 trials per candidate)

- **Lossy codecs:** MP3 64/96/128 kbps, AAC 64/96/128 kbps, Opus 32/64/96/128 kbps,
  Vorbis q2/q4
- **Signal processing:** gain ×0.5, gain ×2.0, peak normalisation, low-pass 4 k / 6 k,
  high-pass 300, band-pass 300–5 k, EQ +6 dB @ 1 kHz, additive pink noise ≈30 dB SNR,
  requantisation to 8-bit, channel gain 0.7, clip/limiter drive
- **Resampling:** 16 k → 8 k / 22.05 k / 44.1 k → 16 k
- **Temporal:** lead trim 0.3 s, tail trim 3 s, mid-extract (10 s window), time-scale
  modification ±2 % (`atempo` 1.02 / 0.98)

Recovery criterion: **BER < 0.10** with the correct key, blind detection.

---

## 3. Headline result

| Candidate | Attacks recovered | Mean BER | Embed SNR | Clean LSD |
|---|---:|---:|---:|---:|
| **`mag_dsss`** | **151 / 192 (79 %)** | **0.068** | 23.1 dB | 1.1 dB |
| `log_dsss` | 0 / 192 (0 %) | 0.479 | 48.1 dB | 0.1 dB |
| `qim_logmag` | 0 / 192 (0 %) | 0.528 | 65.6 dB | 0.0 dB |

**`mag_dsss` is the clear winner.** The other two never recovered *even a clean
round-trip* — the failure is in blind detection, not embedding, and it is instructive:

- `log_dsss` correlates log-magnitude against zero-mean chips, but the host's own
  log-spectrum is large and structured. It swamps the watermark.
- `qim_logmag` is nearly distortion-free (LSD 0.0 dB, SNR 65.6 dB) precisely because
  its quantisation steps are too small to survive *any* subsequent processing — and
  also too small to be detected reliably blind. **Low distortion alone is worthless
  if the detector cannot see the watermark.**

The winning design's essential idea: **the detector must explicitly estimate and
subtract the host signal.** Embedding alone is not the hard part — blind detection is.

### 3.1 `mag_dsss` per-attack detail

| Attack | Mean BER | Recovered |
|---|---:|---:|
| mp3-64 | 0.042 | 5/6 |
| mp3-96 | 0.042 | 5/6 |
| mp3-128 | 0.039 | 5/6 |
| aac-64 | 0.036 | 6/6 |
| aac-96 | 0.042 | 5/6 |
| aac-128 | 0.042 | 5/6 |
| opus-32 | 0.060 | 5/6 |
| opus-64 | 0.039 | 5/6 |
| opus-96 | 0.042 | 5/6 |
| opus-128 | 0.036 | 5/6 |
| vorbis-q2 | 0.065 | 5/6 |
| vorbis-q4 | 0.047 | 5/6 |
| gain-0.5 | 0.034 | 6/6 |
| gain-2.0 | 0.039 | 5/6 |
| peak-norm | 0.034 | 6/6 |
| lowpass-4k | 0.031 | 6/6 |
| lowpass-6k | 0.039 | 5/6 |
| highpass-300 | 0.036 | 5/6 |
| bandpass-300-5k | 0.036 | 5/6 |
| eq-boost-1k | 0.036 | 5/6 |
| noise-30db | 0.057 | 5/6 |
| **resample-8k** | **0.159** | **3/6** |
| resample-22k | 0.036 | 5/6 |
| resample-44k | 0.036 | 5/6 |
| lead-trim-0.3s | 0.042 | 5/6 |
| **mid-extract** | **0.078** | **3/6** |
| tail-trim-3s | 0.052 | 5/6 |
| **tms-1.02** | **0.370** | **0/6** |
| **tms-0.98** | **0.422** | **0/6** |
| requant-8bit | 0.042 | 5/6 |
| channel-gain-0.7 | 0.034 | 6/6 |
| clip-drive | 0.034 | 5/6 |

### 3.2 What this means

**Strong (all recovered, BER ≈ 0.04):**
- Every lossy codec at every tested bitrate — **including Opus down to 32 kbps**
- All gain, normalisation, limiting, channel-gain and clip-drive variants
- Low-pass, high-pass, band-pass, EQ
- Additive noise at ~30 dB SNR
- 8-bit requantisation
- Moderate resampling (22.05 k, 44.1 k)
- Leading trim, trailing trim, arbitrary offset

**Weak / needs work (the design agenda):**
- **Time-scale modification (±2 %)** — fails completely. Spreading over *frequency*
  bins per frame has no mechanism to absorb a change in frame rate. A tempo-invariant
  or explicit-sync design is required.
- **8 kHz resampling** — 3/6. Halving the rate to 8 kHz moves the 4 kHz Nyquist well
  inside the embedded band — `bin_hi = floor(1025 × 0.75) = 768` → 768 × 16000/2048 =
  **6.0 kHz** — so the entire 4–6 kHz slice (≈ a third of the keyed bins) is filtered
  away. The embed band must be capped relative to the *lowest supported* rate.
- **Mid-extract (10 s)** — 3/6. Works, but with less margin; needs the repetition
  framing described in §5.

**The `5/6` pattern** is a *content* effect first and a capacity effect second. The
marginal track is `freak.flac`, which sits **above the 0.10 recovery criterion even
on a clean round-trip** (BER 0.109 ≈ 7 of 64 bits wrong) while the shorter 15.8 s
`galway.flac` passes clean at 0.047 — so duration alone cannot explain the pattern.
Flat `alpha` plus blind host estimation leaves a content-dependent margin: hostile
spectra need adaptive masking (phase 4) to lift, and ECC (phase 3) to turn the
residual bit errors into exact payload recovery. Capacity still sets the floor (see
§4: 15.8 s ≈ 494 frames ≈ 7.7 frames/bit for 64 bits) — it is just not the whole
story.

---

## 4. Capacity vs duration

Measured on a 60 s source, sweeping clip length:

| Clip | Frames | Frames/bit (64-bit) | BER 64-bit | BER 32-bit |
|---:|---:|---:|---:|---:|
| 4 s | 125 | 2.0 | 0.078 | **0.000** |
| 8 s | 250 | 3.9 | 0.062 | 0.000 |
| 12 s | 375 | 5.9 | 0.047 | 0.000 |
| 16 s | 500 | 7.8 | 0.047 | 0.000 |
| 20 s | 625 | 9.8 | 0.016 | 0.000 |
| 24 s | 750 | 11.7 | 0.016 | 0.000 |
| 30 s | 937 | 14.6 | **0.000** | 0.000 |
| 45 s | 1406 | 22.0 | 0.016 | 0.000 |
| 60 s | 1875 | 29.3 | **0.000** | 0.000 |

**Design consequences:**

- ≈ **4 frames/bit** is the practical floor for clean recovery.
- A **64-bit payload needs ≈8 s** of audio; a **32-bit payload works from ≈4 s**.
- The API must therefore **derive minimum input length from payload size** and return
  `AfpError::AudioTooShort { needed, got }` — not silently produce a weak watermark.
- Longer audio buys margin, which is exactly what is spent surviving attacks.

---

## 5. Synchronization

The base design cycles the payload bit index at the frame rate, so a detector that
assumes frame 0 carries bit 0 breaks on *any* temporal edit.

**Measured fix:** search all 64 bit-phase hypotheses and keep the strongest.

| Transform | BER |
|---|---:|
| Full clip, clean | 0.000 |
| Lead trim 0.3 s | 0.016 |
| Lead trim 3.0 s | 0.000 |
| Tail trim 3 s | 0.016 |
| Mid-extract, 10 s window | 0.062 |

**This works and is cheap in implementation**, but it has a critical side effect —
see §7. The recommended refinement is a **dedicated keyed sync pattern** (a known
chip sequence occupying a short frame block, repeated every N payload blocks) so the
detector does one correlation-based *search* rather than 64 full demodulations.

---

## 6. Perceptual quality

Measured with the final configuration (`n_fft=2048`, `hop=512`, 300 bins/frame,
`bin_lo=10`, `bin_hi_frac=0.75`) on two tracks, embedding strength `alpha` = fraction
of local bin magnitude:

| `alpha` | SNR (galway) | SNR (grieg) |
|---:|---:|---:|
| 0.10 | 33.5 dB | 34.1 dB |
| 0.20 | 27.5 dB | 28.1 dB |
| **0.35** (used) | **22.6 dB** | **23.2 dB** |
| 0.80 | 15.4 dB | 16.0 dB |

LSD at the working point is **≈1.0–1.2 dB** (measured separately, same config).
Distortion is **≈23–28 dB SNR / ≈1.1 dB LSD** at `alpha = 0.35`.

Note the SNR depends mildly on the key and payload (different keys select different
bins and chips), so these are representative values rather than exact per-config
constants — an API should not promise an SNR figure.

For context, the **16-bit PCM quantisation floor** was measured separately on the same
tracks at **≈90.5 dB SNR** — so the watermark sits roughly **67 dB above** the
quantisation noise floor. This is expected for a signal-domain watermark (modification
is unavoidable), and it is why the perceptual justification rests on the spread,
noise-like character of the perturbation rather than on low absolute amplitude.

**A more perceptually meaningful metric should be added** (e.g. PEAQ or a masking-based
measurement). SNR and LSD are reported here because they are cheap and reproducible,
but neither is a substitute for a perceptual model — see §13 open question on
adaptive masking.

The design should nevertheless add **adaptive masking** (currently absent): scale
`alpha` per bin by local spectral energy and exclude bins that are near-silent,
near-DC, or in codec-hostile regions. The measured results above use a *flat* alpha,
so adaptive masking should only improve the quality/robustness tradeoff.

---

## 7. ⚠️ False positives and threshold calibration

**This is the most important finding for the API design, and the easiest to get wrong.**

A detector that reports `mean correlation > threshold` will produce garbage. Measured
null vs signal distributions over the six-track corpus:

| Population | Mean statistic | SD | Max |
|---|---:|---:|---:|
| Signal: watermarked, correct key | 86.81 | 27.48 | 128.38 |
| **Null A: unwatermarked, correct key** | **37.64** | 7.95 | **47.86** |
| **Null B: watermarked, wrong key** | **38.92** | 7.95 | 47.99 |
| Null C: watermarked, 1-char key difference | 40.38 | 8.47 | 49.96 |

Two things to note:

1. **The null is not near zero — it is 37.6.** The sync phase search maximises the
   statistic over 64 hypotheses, which *inflates the null distribution*. A naive
   fixed threshold would either miss real watermarks or fire on unwatermarked audio.
2. **Raw separation is only 6.2 σ**, with a signal SD of 27.5 — the signal statistic
   varies enormously across content. A fixed threshold cannot work.

### 7.1 Recommended fix: self-normalising z-score

Compute the statistic for **every** phase hypothesis, then report the *z-score of the
winner within that ensemble* rather than the raw maximum:

```
z = (max_phase_statistic - mean(all_phase_statistics)) / sd(all_phase_statistics)
```

Measured effect:

| Statistic | Signal (mean ± SD) | Null (mean ± SD) | Separation |
|---|---|---|---:|
| Raw max | 86.81 ± 27.48 | 37.64 ± 7.95 | 6.19 σ |
| **z-score** | **6.84 ± 0.42** | **2.57 ± 0.38** | **11.27 σ** |

**Separation nearly doubles, and the signal SD collapses from 27.5 to 0.42** — the
statistic becomes content-independent, which is what makes a *portable* threshold
possible at all.

Per-track z-scores: signal 6.23–7.23, null 2.08–3.19. A threshold around **z = 5.0**
separates them on this corpus with margin.

**Calibration scope.** The z-score is taken over a *max over 64 phase hypotheses*
(the 64-bit payload's sync search), and the null's inflation depends on that count:
changing `payload_bits`, the framing/ECC of §9.4, or replacing the phase search with
the dedicated sync pattern of §5 **changes the null distribution and invalidates
z = 5.0**. Any such change requires re-running this section's measurements — a
calibration is only valid for the exact (payload length, sync design, corpus) it was
measured on.

### 7.2 Other security measurements

- **Wrong key:** mean BER **0.482** (indistinguishable from random) — detection is
  genuinely key-dependent, not a generic property of the audio.
- **Payload discrimination:** same audio, two different payloads →
  `wm(payload)` vs `payload` = 0.047 BER, vs inverted payload = **0.953** BER.
  Different payloads are cleanly distinguishable.
- **Clean recovery, correct key:** mean BER **0.0365**, max **0.109** — and that max
  (`freak.flac`) is *above* the 0.10 recovery criterion: *one track fails even the
  clean round-trip* (see §3.2). Exact payloads on such content require the ECC of
  phase 3; raw detection alone is not sufficient.

**Recommendation:** the detector should return a **score with a documented,
corpus-calibrated threshold**, and the crate should ship a calibration test that
measures TPR/FPR over watermarked / unwatermarked / wrong-key / unrelated-audio
populations. A bare `bool` is not sufficient.

---

## 8. Performance

Single core, release build, 16 kHz mono. Throughput measured by tiling a source to
the target length. (These figures come from a dedicated timing run; the archived
`out/experiments.txt` is a separate run landing within ~10 % on every number —
run-to-run variance on one machine; the ×-realtime ratios are what transfer.)

| Length | Embed | Detect (with sync) | Detect (no sync) |
|---:|---:|---:|---:|
| 5 s | 16.5 ms (**303× RT**) | 92.2 ms (54× RT) | — |
| 30 s | 108.4 ms (**277× RT**) | 614.7 ms (49× RT) | 41.6 ms (**720× RT**) |
| 60 s | 207.8 ms (**289× RT**) | 1242.5 ms (48× RT) | — |
| 300 s | 1075.4 ms (**279× RT**) | 6186.7 ms (49× RT) | — |

**Both targets in the feature request are comfortably met and exceeded:**

- Embed is **~280× realtime** — far above the ">> 1× realtime" target and the
  "100× realtime or better" stretch goal.
- Detect is **~48× realtime** with the 64-phase brute-force sync search, and
  **~720× realtime** without it.

**The sync search is the single dominant cost** — a 15× slowdown. This is a strong
argument for the dedicated sync-pattern design in §5: it should recover most of that
factor while making temporal robustness explicit rather than emergent.

Scaling is linear in duration (no O(N²) behaviour): 30 s → 300 s is 10× the audio for
10× the work.

---

## 9. Proposed design

### 9.1 Algorithm

**STFT-domain magnitude DSSS with per-bin host estimation.**

```
PCM
 │
 ├─ STFT (n_fft=2048, hop=512, periodic Hann, centred)   ← reuse dsp::stft
 │
 ├─ keyed bin permutation → select N bins/frame          ← keyed PRNG
 │
 ├─ per-frame: bit index cycles at frame rate
 │    cell.magnitude += sign(bit) · chip[k] · alpha · |cell|
 │    (phase preserved)
 │
 ├─ inverse STFT + weighted overlap-add                  ← NEW dsp::istft
 │
 └─ watermarked PCM
```

Detection:

```
PCM
 │
 ├─ STFT
 ├─ per-bin long-term mean magnitude  → host estimate
 ├─ residual = |cell| − host_mean[bin]      (chip averages out; host removed)
 ├─ normalise residual by host level        (gain-invariant)
 ├─ correlate residual against keyed chips
 └─ sync phase search → z-score → bits
```

**Why this wins:**
- Embedding is *relative* (`alpha · |cell|`), so it is inherently **gain-invariant** —
  confirmed by gain ×0.5 / ×2.0 / normalisation / channel-gain all recovering.
- Phase is preserved, so the modification is a magnitude-only perturbation that
  **lossy codecs tolerate** — confirmed by all four codecs recovering.
- Host estimation via the long-term per-bin mean is **cheap** (one pass, O(frames ×
  bins)) and needs no reference signal.
- Everything is FFT/float operations the crate already performs, so the DSP stack is
  reused rather than duplicated.

### 9.2 Required DSP addition

The crate has a forward STFT (`dsp::stft::ShortTimeFFT`) but **no inverse**. The
bakeoff confirmed a correct overlap-add synthesis is essential and non-trivial
(measured round-trip **140.5 dB SNR** in the prototype).

Proposed: add a reusable `dsp::istft` primitive rather than a watermark-private one.

```
audiofp::dsp::stft   — forward (exists)
audiofp::dsp::istft  — NEW: inverse STFT + weighted overlap-add
```

It must:
- reuse FFT plans and scratch buffers (no per-frame allocation)
- be deterministic and bit-identical across targets, consistent with the
  crate's existing cross-platform guarantees
- expose the same error conventions as `ShortTimeFFT` (`try_new` / `Config`)

### 9.3 Module layout

```
audiofp::watermark                  ← pub mod gate: any(watermark, classical-watermark)
├── classical            ← NEW, feature = "classical-watermark"
│   ├── mod.rs           config, embedder, detector, results
│   ├── spread_spectrum  the algorithm above
│   ├── coding           CRC + ECC + interleaving
│   ├── sync             keyed sync pattern / phase search
│   ├── psychoacoustic   adaptive masking (phase 4)
│   └── keying           keyed PRNG + keyed expansion
└── detector             existing ONNX detector (UNCHANGED; feature = "watermark")
    └── … also re-exported as `audioseal` for symmetry with `classical` —
           an alias, never a move (see below)
```

**Backwards compatibility:** `audiofp::watermark::{WatermarkConfig, WatermarkDetector,
WatermarkResult}` must keep working exactly as today, **and so must full paths through
`audiofp::watermark::detector::…`** — that module is public today. Do not move it
under `audioseal/`: re-export (`pub use detector as audioseal;`) if the symmetric name
is wanted. The existing `watermark` feature must not change meaning.

**Gating (required, currently missing):** `src/lib.rs` gates the whole module with
`#[cfg(feature = "watermark")]`, and `watermark = ["std", "dep:tract-onnx"]`. For
`audiofp::watermark::classical` to exist under `classical-watermark` alone, that gate
must become `#[cfg(any(feature = "watermark", feature = "classical-watermark"))]`,
with `mod detector;` additionally gated `#[cfg(feature = "watermark")]` inside
`watermark/mod.rs` and `mod classical;` gated on `classical-watermark`. The crate is
`no_std + alloc` by default, and the classical subsystem is FFT + heap only — so
`classical-watermark` must **not** imply `std` (unlike `watermark`).

### 9.4 API sketch

```rust
use audiofp::{SampleRate, watermark::classical::{
    SpreadSpectrumConfig, SpreadSpectrumEmbedder, SpreadSpectrumDetector,
}};

// key: &[u8] — any length, domain-separated expansion inside (§9.5).
let config = SpreadSpectrumConfig::new()
    .sample_rate(16_000)
    .payload_bits(64)
    .strength(0.35);

// `try_new`, not `new` — crate convention: `new()` panics on invalid
// params, `try_new()` returns `Result` (src/lib.rs, Error Handling).
let mut embedder = SpreadSpectrumEmbedder::try_new(config.clone(), key)?;
let watermarked = embedder.embed(&samples, SampleRate::HZ_16000)?;

let mut detector = SpreadSpectrumDetector::try_new(config, key)?;
let result = detector.detect(&samples, SampleRate::HZ_16000)?;

if result.detected() {   // derived: `score >= threshold` — not an independent field
    println!("payload = {:?}", result.payload);
    println!("score   = {:.2} (threshold {:.2})", result.score, result.threshold);
    println!("offset  = {:?}", result.sync_offset);
}
```

Notes:
- **Payload is not hard-coded to 16 or 32 bits** — the frame is
  `version | flags | length | payload | CRC`, then ECC + interleaving.
- **`rate` is passed at the call site** and validated against `config.sample_rate`
  (mismatch → `UnsupportedSampleRate`), exactly like the existing
  `WatermarkDetector::detect(samples, rate)` — one convention across the crate.
- `detect` returns a **score plus threshold**, never a bare `bool`;
  `detected()` is documented as derived from those two so it can never disagree.
- `result.sync_offset` reports where the watermark was found (localization).

### 9.5 Keying

- **In-crate ChaCha20-style keystream** (no new dependency). The bakeoff prototype
  used exactly this and it is deterministic and integer-only, so it is
  cross-platform stable.
- The key derives **three independent domains**: bin permutation, chip sequences, and
  the sync pattern — all via domain-separated key derivation.
- **Measured:** a one-character key difference drops the statistic into the null
  distribution (mean 40.38 vs signal 86.81) and yields BER 0.482.
- The key must never be embedded in the media or exposed through the API.

### 9.6 Error handling

Use `AfpError` / `Result<T>`:
- invalid config (bad `alpha`, zero payload, non-power-of-two `n_fft`) → `Config`
- unsupported sample rate → `UnsupportedSampleRate`
- input shorter than `payload_bits × 4 / frames_per_sec` → `AudioTooShort { needed, got }`
- non-finite PCM → `NonFiniteSample { index }`

No panics for user-configurable input.

### 9.7 Feature flags

```toml
classical-watermark = []          # no new dependencies, no std requirement
```

Reuses `rustfft`, `libm`, `wide`, `bytemuck`. Explicitly does **not** pull in
`tract-onnx`, and — unlike `watermark` — does not imply `std`: the subsystem is
FFT + `alloc` only, so it works under the crate's default `no_std + alloc` build
(see the gating note in §9.3; `src/lib.rs` must switch to `any(...)` or this
feature exposes nothing). The existing `watermark` feature stays as-is.

---

## 10. Known limitations (measured, not speculative)

| Limitation | Evidence | Mitigation path |
|---|---|---|
| **Time-scale modification ±2 %** | BER 0.37–0.42, 0/6 recovered | Dedicated sync pattern + tempo-invariant spreading; may require a separate design |
| **8 kHz resampling** | BER 0.159, 3/6 recovered | Cap embed band relative to the lowest supported rate |
| **Short clips** | Duration sweep — §4: 4 s clip → BER 0.078 (64-bit) vs 0.000 (32-bit) | Derive min-length from payload size; reject early |
| **Opus at low bitrate** | 32 kbps: BER 0.060 (recovers, least margin) | Stronger alpha in low-SNR bands; longer repetition |
| **Sync search cost** | detect 48× RT vs 720× RT without | Dedicated sync pattern instead of 64-phase search |
| **No adaptive masking yet** | Flat alpha used throughout | Phase-4 psychoacoustic module |

**Explicitly out of scope / not claimed:**
- cryptographic proof of ownership
- survival of arbitrary adversarial signal processing
- survival of speaker→microphone re-recording
- replacement of AudioSeal

---

## 11. Proposed implementation phases

| Phase | Deliverable | Evidence it is needed |
|---|---|---|
| 1 | `dsp::istft` inverse STFT + overlap-add | Prototype round-trip measured at 140.5 dB; required by everything else |
| 2 | Keyed PRNG, keyed bin selection, DSSS embed, blind detector | `mag_dsss` — 79 % attack recovery, 280× RT |
| 3 | Sync pattern, CRC, ECC, interleaving, repetition | Phase search fixed temporal attacks; sync pattern fixes its cost |
| 4 | Adaptive masking / perceptual strength | Flat alpha already at 23–28 dB SNR; masking should improve it |
| 5 | Streaming embed + detect | Feature requirement; measured linear scaling makes it tractable |
| 6 | Robustness benchmark harness | Reuse `tests/assets/` + ffmpeg matrix from this bakeoff |
| 7 | Optimization (sync cost, SIMD) | Sync is 15× the detect cost |

---

## 12. Verification plan

- **Round trip:** embed → detect → exact payload, across corpus
- **No-watermark:** unwatermarked audio must not pass at the calibrated FPR
- **Wrong key:** key A embed, key B detect → must fail (measured BER 0.482)
- **Payload variation:** multiple payloads on the same content must be distinguishable
  (measured 0.047 vs 0.953 BER)
- **Codec round-trip:** MP3 / AAC / Opus / Vorbis at the bitrate ladder
- **Transform matrix:** gain, filters, EQ, noise, requantisation, resampling
- **Temporal:** trims, offsets, mid-extract, TSM
- **Streaming equivalence:** 1 / 16 / 256 / 1 s / full-buffer chunking
- **Determinism:** same input + config + key + payload → identical output on every
  supported target, covered by a regression test
- **Quality:** SNR and LSD reported per test
- **Performance:** `benches/watermark.rs` at 5 s / 30 s / 5 min for embed and detect

---

## 13. Open questions

1. **Time-scale modification** — is ±2 % TSM a hard requirement? It is the only
   attack with a complete failure, and solving it well may need a different
   spreading geometry (e.g. logarithmic-frequency bands) rather than a parameter tweak.
2. **Minimum clip length** — should the API refuse short input, or embed a
   reduced/repeated payload and report lower confidence?
3. **Sync pattern design** — dedicated keyed sync block vs. the measured 64-phase
   search. The former is faster and more explicit; the latter needs no extra framing.
4. **Detection threshold ownership** — ship a fixed corpus-calibrated default, or
   require the caller to calibrate against their own null corpus?
5. **Multi-channel** — the bakeoff was mono only. Stereo needs a policy: embed in
   both channels, mid-only, or per-channel keys.
6. **Does `mag_dsss` need the `log_dsss` variant?** Log-magnitude is theoretically more
   gain-invariant; `mag_dsss` achieved gain invariance through relative scaling instead.
   Worth re-testing log-domain *with* host estimation, which the failed `log_dsss`
   candidate lacked.

---

## 14. Reproducing these results

The harness lives in `bakeoff/wm/` (committed; dev-only, not a workspace member, not
published — following the existing `bakeoff/` precedent for the chromaprint
comparison). `bakeoff/wm/ISSUE.md` is a verbatim copy of this issue body — **edit both
together**, or regenerate one from the other.

```bash
cd bakeoff/wm
cargo build --release

./target/release/audiofp-wm-bakeoff      # full attack matrix
./target/release/duration                # capacity vs clip length
./target/release/capacity                # frame counts for three probe tracks
./target/release/fp_study                # FP / wrong-key calibration
./target/release/zscore                  # detector statistic comparison
./target/release/speed                   # throughput
./target/release/quality                 # distortion vs alpha
./target/release/floor                   # 16-bit quantisation floor
./target/release/rt_verify               # inverse-STFT round-trip fidelity
./target/release/sync                    # temporal robustness of sync search
./target/release/gaps                    # targeted Opus / sync follow-ups
./target/release/opus_check              # Opus-only replication of the matrix path
```

Requires `ffmpeg` on `PATH` for the codec and filter attacks. Corpus files are
symlinked into `bakeoff/wm/assets/` from `tests/assets/` and
`tests/assets/catalog/`.

**Provenance**

- `audiofp` v0.4.4, commit `ced76c1`
- rustc 1.93.0 (254b59607 2026-01-19)
- Linux 7.0.0 x86_64, Intel i5-1135G7, 8 logical cores
- ffmpeg 8.0.1
- release build, `lto = "fat"`, `codegen-units = 1`, single-threaded

### Caveats

- **Corpus size.** Six tracks, 4 × 16 s + 2 × 30 s. Enough to separate the three
  candidates decisively and to establish the calibration methodology, but **not**
  enough to publish a final false-positive rate. The z-score threshold of 5.0 is a
  starting point that must be re-calibrated on a larger corpus.
- **Corpus content.** All six tracks are music/classical — **no speech**. The
  `tests/assets/` corpus includes `speech.ogg` and `piano.ogg`, and
  `tests/assets/catalog/` has nine more orchestral excerpts; none were used here.
  Speech (sparse, high dynamic range) is a meaningfully different host for a
  magnitude-domain watermark and should be in any re-calibration corpus.
- **Flat embedding strength.** No adaptive masking was used, so the reported quality
  is a *lower bound* on what the final design should achieve.
- **Single sample rate in the attack matrix.** Everything was embedded at 16 kHz.
  Real deployments may embed at 44.1/48 kHz, where the usable band and the
  codec-hostile regions differ.
- **ffmpeg codec defaults.** Codec attacks use libmp3lame / aac / libopus / libvorbis
  defaults. Other encoders may behave differently.
- **One machine.** Timing numbers are from a single laptop CPU; ratios (× realtime)
  should transfer, absolute milliseconds will not.
