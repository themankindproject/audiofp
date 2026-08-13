# Security Policy

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public GitHub issue for exploitable bugs.

- **Email:** kumarashutosh34169@gmail.com (subject prefix: `[audiofp security]`)
- **GitHub:** [Security advisories](https://github.com/themankindproject/audiofp/security/advisories/new) (preferred when available)

Include: crate version / git commit, feature flags, minimal repro (file or PCM description), and impact. We aim to acknowledge within **7 days** and coordinate a fix before public disclosure.

---

## Threat model (what `audiofp` is and is not)

`audiofp` extracts **perceptual** fingerprints and (optionally) runs **caller-supplied** ONNX models. It is a multimedia / DSP library, not a cryptographic toolkit.

| Asset | Trust assumption | Reality |
| --- | --- | --- |
| **Hash / embedding output** | Useful for identification & dedup | **Not** a MAC, signature, or password hash. Collision resistance is perceptual, not cryptographic. Do not use fingerprints as auth tokens or integrity proofs. |
| **Audio files** (`io`, Symphonia) | Often **untrusted** uploads | Parser / decoder surface. Prefer capped decode APIs. |
| **PCM `f32` buffers** | Caller-controlled | Extreme length → memory/CPU. Classical configs default to a 30‑minute sample cap; set `max_input_samples` / `DecodeLimits` for your SLA. |
| **ONNX models** (`neural`, `watermark`) | Must be **trusted** | Tract deserializes and executes the model. Treat model files like executable code. Do not load untrusted ONNX. |

```text
  Untrusted audio ──caps──▶ decode ──▶ classical / streaming ──▶ perceptual hashes
  Trusted PCM     ─────────▶ extract / push ──────────────────▶ (same)
  Trusted ONNX    ─────────▶ neural / watermark ──────────────▶ embeddings / detect
```

### In scope (library responsibilities)

- Fail with structured errors (`AfpError`) rather than panicking on normal bad input where APIs are fallible (`extract`, decode helpers).
- Provide **optional resource budgets** for untrusted paths:
  - `io::DecodeLimits` / `decode_to_mono_limited`
  - `max_input_samples` on classical (and neural) configs
  - Streaming: `max_push_samples` where implemented (Panako today; others tracked in issues)
- Document panic surfaces (e.g. some constructors still `assert!` on invalid config; neural streaming `push` may panic on inference failure — use `try_push` when you need `Result`).
- Keep dependency policy green (`cargo audit` / `cargo-deny` in CI).

### Out of scope / caller responsibilities

- Bounding wall-clock time for decode (timeout tracked separately).
- Guaranteeing complete fingerprints when the decoder skips recoverable packet errors (integrity mode tracked separately).
- Sandboxing Symphonia or Tract (run untrusted decode/inference in a separate process/container if required).
- Constant-time comparison of hashes for security-sensitive equality (not provided; perceptual hashes are not secrets).
- Protecting against intentional adversarial audio crafted to evade matching (robustness ≠ adversarial ML security).

---

## Recommended production defaults

For multi-tenant upload services:

1. Prefer **`decode_to_mono_limited`** with `DecodeLimits::both(max_bytes, max_samples)` — never rely on on-disk size alone for compressed formats.
2. Keep classical `max_input_samples` at the safe default (or tighter); set `None` only for trusted offline batch jobs.
3. Load **only pinned, reviewed** ONNX weights for `neural` / `watermark`; do not accept user-uploaded models.
4. Run the service with OS-level memory limits (cgroups) even when library caps are set.
5. Treat emitted hashes as **public-ish identifiers**, not credentials.

Example (trusted path vs upload path):

```rust
use audiofp::classical::Wang;
use audiofp::io::{decode_to_mono_at, decode_to_mono_at_limited, DecodeLimits};
use audiofp::prelude::*;

fn enroll_trusted(path: &str) -> audiofp::Result<WangFingerprint> {
    let samples = decode_to_mono_at(path, 8_000)?;
    let mut wang = Wang::default();
    wang.extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
}

fn enroll_upload(path: &str) -> audiofp::Result<WangFingerprint> {
    let limits = DecodeLimits::both(50 * 1024 * 1024, 30 * 60 * 8_000);
    let samples = decode_to_mono_at_limited(path, 8_000, limits)?;
    let mut wang = Wang::default();
    wang.extract(AudioBuffer::new(&samples, SampleRate::HZ_8000))
}
```

---

## Known limitations / hardening backlog

These are tracked as non-breaking issues; this document will be updated as they land:

| Topic | Issue |
| --- | --- |
| NaN/Inf PCM policy at extract/push | [#75](https://github.com/themankindproject/audiofp/issues/75) — done |
| Decoder integrity (fail on corrupt packets) | [#76](https://github.com/themankindproject/audiofp/issues/76) |
| Decode wall-clock timeout | [#77](https://github.com/themankindproject/audiofp/issues/77) |
| ONNX max size / content-hash pin | [#78](https://github.com/themankindproject/audiofp/issues/78) |
| Model path load-error mapping | [#79](https://github.com/themankindproject/audiofp/issues/79) — done |
| Finish `max_push_samples` on all streamers | [#80](https://github.com/themankindproject/audiofp/issues/80) — done |

---

## Supported versions

| Version | Supported |
| --- | --- |
| Latest `0.3.x` on crates.io / `main` | Yes |
| Older `0.3.x` / `0.2.x` | Best-effort; please upgrade |

Security fixes are released as patch or minor versions on the current major line (`0.x` pre-1.0 may include necessary API notes in the changelog).
