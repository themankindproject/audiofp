#!/usr/bin/env python3
"""Regression tests for scripts/check_usage_snippets.py."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
CHECKER = REPO / "scripts" / "check_usage_snippets.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_usage_snippets", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def compile_and_run_test_harness(blocks: list[str]) -> subprocess.CompletedProcess[str]:
    checker = load_checker()
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="usage-check-test-"))
    try:
        source = scratch / "snippet.rs"
        lib_rs, _compiled, _ran = checker.build_lib_rs(blocks, set(checker.DEFAULT_FEATURES))
        source.write_text(lib_rs)
        binary = scratch / ("snippet.exe" if sys.platform == "win32" else "snippet")
        compile_result = subprocess.run(
            ["rustc", "--edition=2021", "--test", str(source), "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        if compile_result.returncode != 0:
            raise AssertionError(compile_result.stderr)
        return subprocess.run(
            [str(binary)],
            cwd=scratch,
            capture_output=True,
            text=True,
        )
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)


class CheckUsageSnippetsTests(unittest.TestCase):
    def test_main_returning_err_fails(self) -> None:
        blocks = [
            textwrap.dedent(
                """\
                fn main() -> Result<(), &'static str> {
                    Err("deliberate failure")
                }
                """
            )
        ]
        proc = compile_and_run_test_harness(blocks)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("test result: FAILED", proc.stdout)

    def test_result_type_alias_success(self) -> None:
        blocks = [
            textwrap.dedent(
                """\
                type MyResult<T> = Result<T, &'static str>;

                fn main() -> MyResult<()> {
                    Ok(())
                }
                """
            )
        ]
        proc = compile_and_run_test_harness(blocks)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("test result: ok", proc.stdout)

    def test_fully_qualified_result_success(self) -> None:
        blocks = [
            textwrap.dedent(
                """\
                fn main() -> std::result::Result<(), &'static str> {
                    Ok(())
                }
                """
            )
        ]
        proc = compile_and_run_test_harness(blocks)
        self.assertEqual(proc.returncode, 0)

    def test_plain_unit_main_success(self) -> None:
        blocks = [
            textwrap.dedent(
                """\
                fn main() {
                    assert_eq!(2 + 2, 4);
                }
                """
            )
        ]
        proc = compile_and_run_test_harness(blocks)
        self.assertEqual(proc.returncode, 0)

    def test_exit_code_failure_is_detected(self) -> None:
        blocks = [
            textwrap.dedent(
                """\
                use std::process::ExitCode;

                fn main() -> ExitCode {
                    ExitCode::FAILURE
                }
                """
            )
        ]
        proc = compile_and_run_test_harness(blocks)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("test result: FAILED", proc.stdout)

    def test_zero_runnable_snippets_is_rejected(self) -> None:
        checker = load_checker()
        blocks = [
            textwrap.dedent(
                """\
                fn main() {
                    let _ = audiofp::io::decode_to_mono_at("song.mp3", 8_000);
                }
                """
            )
        ]
        _lib, compiled, ran = checker.build_lib_rs(blocks, set(checker.DEFAULT_FEATURES))
        self.assertEqual(compiled, 1)
        self.assertEqual(ran, 0)

    def test_feature_gating_skips_neural_blocks(self) -> None:
        checker = load_checker()
        blocks = ["fn main() { let _ = audiofp::neural::NeuralEmbedderConfig::new(\"x.onnx\"); }"]
        _lib, compiled, ran = checker.build_lib_rs(blocks, {"std-wav", "std-mp3"})
        self.assertEqual(compiled, 0)
        self.assertEqual(ran, 0)


if __name__ == "__main__":
    unittest.main()
