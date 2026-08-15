#!/usr/bin/env python3
"""Compile- and runtime-check every runnable rust snippet in USAGE.md.

Extracts each ```rust fenced block, wraps it in a scratch crate that
depends on the local audiofp checkout (all optional features on), and:

  * compiles every block (compile validity), and
  * executes every block that has no external dependency (files, ONNX
    models, tokio) as a test, so the snippets' own assertions run.

Blocks tagged ```rust,ignore are reference definitions (trait/struct
shapes mirrored from src/) and are skipped by design.

Usage:
    python3 scripts/check_usage_snippets.py [--keep]

    --keep  keep the scratch crate under target/usage-check for inspection

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
USAGE = REPO / "USAGE.md"

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
)

CARGO_TOML = """\
[package]
name = "usage_check"
version = "0.1.0"
edition = "2021"

[dependencies]
audiofp = { path = "%s", features = ["all-codecs", "neural", "watermark", "rayon"] }
tokio = { version = "1", features = ["rt"] }

[workspace]
""" % REPO.as_posix()


def extract_blocks(markdown: str) -> list[str]:
    # Runnable blocks: ```rust fences. Reference definitions use
    # ```rust,ignore and are skipped by the regex.
    return re.findall(r"```rust\n(.*?)```", markdown, re.S)


def build_lib_rs(blocks: list[str]) -> str:
    out = [
        "// Auto-generated from USAGE.md by scripts/check_usage_snippets.py.",
        "#![allow(dead_code, unused_variables, unused_imports, unused_mut, unused_parens)]",
        "",
    ]
    tests = ["#[cfg(test)]", "mod usage_tests {", "use super::*;"]
    ran = 0

    for i, block in enumerate(blocks):
        block = block.rstrip()
        runnable = not any(k in block for k in SKIP_RUN)
        if "fn main" in block:
            block = block.replace("fn main", "pub fn main", 1)
            out.append(f"mod block_{i:02d} {{\n{block}\n}}")
            if runnable:
                tests.append(
                    f"#[test]\nfn run_block_{i:02d}() {{ block_{i:02d}::main(); }}"
                )
                ran += 1
        else:
            out.append(
                f"mod block_{i:02d} {{\npub fn body() {{\n{block}\n}}\n}}"
            )
            if runnable:
                tests.append(
                    f"#[test]\nfn run_block_{i:02d}() {{ block_{i:02d}::body(); }}"
                )
                ran += 1
        out.append("")

    tests.append("}")
    out.extend(["", *tests])
    print(f"{len(blocks)} rust blocks: {len(blocks) - ran} compile-only, {ran} compile+run")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    blocks = extract_blocks(USAGE.read_text())
    if not blocks:
        print("no rust blocks found in USAGE.md", file=sys.stderr)
        return 1

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="usage-check-"))
    try:
        (scratch / "src").mkdir()
        (scratch / "Cargo.toml").write_text(CARGO_TOML)
        (scratch / "src" / "lib.rs").write_text(build_lib_rs(blocks))

        # The no_std snippet uses a crate-root-only inner attribute; the
        # module wrapper unavoidably warns. Everything else must be clean.
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
        print("all USAGE.md rust snippets compile; runnable snippets pass")
        return 0
    finally:
        if args.keep:
            print(f"scratch crate kept at {scratch}")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
