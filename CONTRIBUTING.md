# Contributing to audiofp

Thanks for your interest in contributing! This document describes how to set
up the project locally, the testing/lint expectations CI enforces, and the
versioning policy.

## Getting started

```bash
git clone https://github.com/themankindproject/audiofp
cd audiofp

# Run the full test suite (unit + doctest) with all features.
cargo test --all-features

# The same gates CI enforces:
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

The MSRV is **1.93.0** (pinned in `rust-toolchain.toml`).

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security reports go
through [SECURITY.md](SECURITY.md), not public issues.

## Project layout

```
audiofp/
├── Cargo.toml
├── rust-toolchain.toml      # MSRV 1.93.0
├── rustfmt.toml, clippy.toml, deny.toml
├── .github/workflows/ci.yml # parallel fmt + clippy + test jobs
├── .github/ISSUE_TEMPLATE/  # bug + feature forms
├── README.md, USAGE.md, CHANGELOG.md, SECURITY.md, CODE_OF_CONDUCT.md
└── src/
    ├── lib.rs
    ├── error.rs             # AfpError + Result
    ├── types.rs             # SampleRate, AudioBuffer, TimestampMs
    ├── fp.rs                # Fingerprinter, StreamingFingerprinter
    ├── dsp/                 # STFT, mel, peaks, resample, windows
    ├── classical/           # Wang, Panako, Haitsma + streaming
    ├── io/        [std]     # symphonia decoder
    └── watermark/ [feat]    # AudioSeal ONNX wrapper
```

## Pull request checklist

Before opening a PR, make sure:

1. `cargo test --all-features` passes (unit + doctests).
2. `cargo fmt --all -- --check` reports no diffs.
3. `cargo clippy --all-targets --all-features -- -D warnings` is clean.
4. `cargo clippy --all-targets --no-default-features -- -D warnings` is clean
   (catches accidental `std`-only code in the `no_std` path).
5. Public items have a doc comment with at least one example. `#![deny(missing_docs)]`
   is set on every crate root, so undocumented items will fail CI.
6. New algorithm code includes a streaming/offline equivalence test if it has
   both modes.
7. `CHANGELOG.md` entry under `[Unreleased]` describes the change.

## Coding standards

- **No `unwrap()` outside tests / examples.** Use `?` with `AfpError`.
- **No `expect()` without a message that helps an end user diagnose the issue.**
- **No `println!` in library code** — `tracing` is reserved for instrumentation.
- **No allocation in streaming `push` after warm-up.** Streaming impls allocate
  scratch at construction; the hot path must stay allocation-free.
- **`unsafe` is rare.** Allowed in: FFI boundaries, `bytemuck` casts checked by the
  trait, and audio-callback hot paths where bounds checks have been profiled.
  Every `unsafe` block carries a `// SAFETY:` comment.

## no_std status

`audiofp` is **host-only** today, despite the `no_std + alloc` API shape.
The `rustfft -> num-traits/std` dependency chain pulls in `std` unconditionally,
so `cargo check --no-default-features --target thumbv7em-none-eabihf` will
fail. `rust-toolchain.toml` therefore no longer lists a bare-metal target
(it was aspirational — nothing exercised the `rust-std` component for it).

True embedded support is tracked in `future.md` §2.1 (swap `rustfft` for
`microfft` behind a feature flag). Until that lands, please don't add code
that assumes bare-metal builds; assume `--no-default-features` runs on a
host target only.

## Versioning

The project follows [Semantic Versioning](https://semver.org/) once it ships
1.0.0. While we are pre-1.0:

- Patch versions (`0.1.x`) are bug fixes and additive APIs.
- Minor versions (`0.x.0`) may include breaking changes; the changelog will say so.
- Hash byte layouts are stable across patch and minor 0.x bumps. A future
  change to hash output will bump the algorithm's `name()` suffix
  (e.g. `"wang-v1"` → `"wang-v2"`) and roll the package major.

### API stability policy (0.x)

- **Breaking changes are batched** into `0.x.0` minor releases — never
  shipped piecemeal in a patch. Each breaking release must include a
  migration guide (`MIGRATING_*.md`) and a CHANGELOG entry listing every
  break.
- **Public API additions** (new types, methods, impls) are non-breaking
  and may land in any release.
- **Hash bytes** are part of the API contract: any change to produced
  hash bytes requires bumping the fingerprinter's `name()` suffix
  (`wang-v1` → `wang-v2`) in the same release, regenerating golden
  files, and (if the serialized representation changes) bumping the
  serialization `FORMAT_VERSION`.
- **Removing or renaming public items** is always a breaking change for
  `0.x.0`; prefer deprecation with a one-release grace period when
  practical.
- MSRV (currently 1.93.0, pinned in `rust-toolchain.toml`) is
  non-breaking to raise; bump it in a `0.x.0` release and note it in the
  changelog.

## Filing issues

When filing a bug, please include:

- Output of `rustc --version`.
- Audio format / sample rate / duration (a short clip is welcome if it's
  reproducible).
- The exact command or code that triggers the bug.
- For determinism issues: the hash sequence you got vs the one you expected.

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
