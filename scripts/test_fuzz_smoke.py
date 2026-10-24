#!/usr/bin/env python3
"""Regression tests for scripts/fuzz_smoke.sh."""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fuzz_smoke.sh"
FUZZ_TOML = REPO / "fuzz" / "Cargo.toml"


class FuzzSmokeTests(unittest.TestCase):
    def test_script_has_no_hard_coded_target_count(self) -> None:
        text = SCRIPT.read_text()
        self.assertNotRegex(text, r"-ne\s+\d+")
        self.assertIn("fuzz/Cargo.toml", text)

    def test_script_targets_match_cargo_fuzz_bins(self) -> None:
        text = SCRIPT.read_text()
        start = text.index("TARGETS=(")
        end = text.index(")", start)
        script_targets = set(
            re.findall(r'^\s+([a-z0-9_]+)\s*$', text[start:end], re.M)
        )
        cargo_names = set(
            re.findall(
                r'\[\[bin\]\]\s*\nname = "([^"]+)"',
                FUZZ_TOML.read_text(),
                re.M,
            )
        )
        self.assertEqual(script_targets, cargo_names)

    def test_script_enforces_bounded_runs(self) -> None:
        text = SCRIPT.read_text()
        self.assertIn("-runs=", text)
        self.assertIn("-max_total_time=", text)
        self.assertIn("-max_len=", text)

    def test_script_fails_when_target_count_wrong(self) -> None:
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "scripts").mkdir()
            (root / "fuzz").mkdir()
            script = root / "scripts" / "fuzz_smoke.sh"
            shutil.copy(SCRIPT, script)
            (root / "fuzz" / "Cargo.toml").write_text(
                '[[bin]]\nname = "only_one"\n'
            )
            proc = subprocess.run(
                ["bash", str(script)], capture_output=True, text=True
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("expected 1 fuzz targets", proc.stderr)


if __name__ == "__main__":
    unittest.main()
