## Summary

<!-- What and why (1–3 bullets). Link related issues: Closes #N -->

## Semver

- [ ] **non-breaking** (additive API, docs, CI, internal fix)
- [ ] **breaking** (target 0.4.0 — update migration notes)

## Checklist

- [ ] `cargo test --all-features`
- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo clippy --all-targets --no-default-features -- -D warnings` (if touching shared code)
- [ ] `CHANGELOG.md` `[Unreleased]` updated
- [ ] Docs / examples updated when public API changes
