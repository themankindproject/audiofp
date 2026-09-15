#!/usr/bin/env python3
"""Compile- and runtime-check every runnable rust snippet in docs.

Extracts each ```rust fenced block from USAGE.md and README.md, wraps it in
a scratch crate that depends on the local audiofp checkout, and:

  * compiles every block (compile validity), and
  * executes every block that has no external dependency (files, ONNX
    models, tokio) as a test, so the snippets' own assertions run.

Blocks tagged ```rust,ignore are reference definitions (trait/struct
shapes mirrored from src/) and are skipped by design.

Usage:
    python3 scripts/check_usage_snippets.py [--keep]
    python3 scripts/check_usage_snippets.py --features std-wav,std-mp3 --docs README.md

    --keep       keep the scratch crate under target/usage-check for inspection
    --features   comma-separated Cargo features for the audiofp dependency
                 (default: all-codecs,neural,watermark,rayon)
    --docs       comma-separated doc basenames to check (default: all three)

Exit status 0 = all snippets compile and all runnable tests pass.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
ALL_DOC_FILES = (REPO / "USAGE.md", REPO / "README.md")
DEFAULT_FEATURES = ("all-codecs", "neural", "watermark", "rayon")

# Blocks whose code touches files / models / other crates are compile-only.
SKIP_RUN = (
    "decode_to_mono",
    ".onnx",
    "tokio",
    "song.",
    "clip.wav",
    "user_upload",
    "catalog_",
    "fingerprint_blocking",
    "enroll_batch",
    "suspect",
    "cache_to_file",
    "enroll_dir",
)

# Skip entire blocks when the selected feature set does not include a gate.
FEATURE_GATES: tuple[tuple[str, str], ...] = (
    ("neural::", "neural"),
    ("audiofp::neural", "neural"),
    ("watermark::", "watermark"),
    ("audiofp::watermark", "watermark"),
    ("fingerprint_batch_parallel", "rayon"),
    ("par_match_best", "rayon"),
    ("decode_to_mono", "std-wav"),
    ("audiofp::io", "std-wav"),
)


def cargo_toml(features: tuple[str, ...]) -> str:
    feature_list = ", ".join(f'"{f}"' for f in features)
    return f"""\
[package]
name = "usage_check"
version = "0.1.0"
edition = "2021"

[dependencies]
audiofp = {{ path = "{REPO.as_posix()}", features = [{feature_list}] }}
tokio = {{ version = "1", features = ["rt"] }}

[workspace]
"""


def extract_blocks(markdown: str) -> list[str]:
    # Runnable blocks: ```rust fences. Reference definitions use
    # ```rust,ignore and are skipped by the regex.
    return re.findall(r"```rust\n(.*?)```", markdown, re.S)


def block_needs_features(block: str, enabled: set[str]) -> bool:
    codec_ok = "all-codecs" in enabled or any(
        f.startswith("std-") for f in enabled
    )
    for needle, feature in FEATURE_GATES:
        if needle not in block:
            continue
        if feature.startswith("std-") and codec_ok:
            continue
        if feature not in enabled:
            return False
    return True


def main_invocation(block_idx: int) -> str:
    """Preserve Rust main's Termination semantics, including type aliases."""
    return (
        f"#[test]\nfn run_block_{block_idx:02d}() {{\n"
        "    use std::process::{ExitCode, Termination};\n"
        f"    assert_eq!(block_{block_idx:02d}::main().report(), ExitCode::SUCCESS);\n"
        "}"
    )


def body_invocation(block_idx: int) -> str:
    return f"#[test]\nfn run_block_{block_idx:02d}() {{ block_{block_idx:02d}::body(); }}"


def build_lib_rs(blocks: list[str], enabled: set[str]) -> tuple[str, int, int]:
    out = [
        "// Auto-generated from docs by scripts/check_usage_snippets.py.",
        "#![allow(dead_code, unused_variables, unused_imports, unused_mut, unused_parens)]",
        "",
    ]
    tests = ["#[cfg(test)]", "mod usage_tests {", "use super::*;"]
    ran = 0
    compiled = 0

    for i, block in enumerate(blocks):
        block = block.rstrip()
        if not block_needs_features(block, enabled):
            continue
        compiled += 1
        runnable = not any(k in block for k in SKIP_RUN)
        if "fn main" in block:
            block = block.replace("fn main", "pub fn main", 1)
            out.append(f"mod block_{i:02d} {{\n{block}\n}}")
            if runnable:
                tests.append(main_invocation(i))
                ran += 1
        else:
            out.append(
                f"mod block_{i:02d} {{\npub fn body() {{\n{block}\n}}\n}}"
            )
            if runnable:
                tests.append(body_invocation(i))
                ran += 1
        out.append("")

    tests.append("}")
    out.extend(["", *tests])
    return "\n".join(out), compiled, ran


def resolve_doc_files(names: str | None) -> list[pathlib.Path]:
    if not names:
        return [p for p in ALL_DOC_FILES if p.exists()]
    docs: list[pathlib.Path] = []
    for name in names.split(","):
        name = name.strip()
        path = REPO / name
        if not path.exists():
            print(f"missing doc file: {path}", file=sys.stderr)
            raise SystemExit(1)
        docs.append(path)
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument(
        "--features",
        default=",".join(DEFAULT_FEATURES),
        help="comma-separated audiofp Cargo features (default: all optional features on)",
    )
    ap.add_argument(
        "--docs",
        default=None,
        help="comma-separated doc basenames to check (default: USAGE.md, README.md)",
    )
    args = ap.parse_args()

    features = tuple(f.strip() for f in args.features.split(",") if f.strip())
    if not features:
        print("at least one --features entry is required", file=sys.stderr)
        return 1
    enabled = set(features)

    blocks: list[str] = []
    for doc in resolve_doc_files(args.docs):
        blocks.extend(extract_blocks(doc.read_text()))

    if not blocks:
        print("no rust blocks found in selected docs", file=sys.stderr)
        return 1

    lib_rs, compiled, ran = build_lib_rs(blocks, enabled)
    print(
        f"{len(blocks)} rust blocks in docs: "
        f"{len(blocks) - compiled} feature-gated out, "
        f"{compiled - ran} compile-only, {ran} compile+run"
    )
    if compiled == 0:
        print("no snippets selected after feature gating", file=sys.stderr)
        return 1
    if ran == 0:
        print(
            "zero runnable snippets — would pass vacuously; "
            "broaden --features or --docs",
            file=sys.stderr,
        )
        return 1

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="usage-check-"))
    try:
        (scratch / "src").mkdir()
        (scratch / "Cargo.toml").write_text(cargo_toml(features))
        (scratch / "src" / "lib.rs").write_text(lib_rs)

        proc = subprocess.run(
            ["cargo", "test", "--quiet"],
            cwd=scratch,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 1
        print("all doc rust snippets compile; runnable snippets pass")
        return 0
    finally:
        if args.keep:
            print(f"scratch crate kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
