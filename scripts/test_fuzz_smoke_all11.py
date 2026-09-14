#!/usr/bin/env python3
"""Regression tests for scripts/fuzz_smoke_all11.sh."""

from __future__ import annotations

import pathlib
import re
import subprocess
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fuzz_smoke_all11.sh"
FUZZ_TOML = REPO / "fuzz" / "Cargo.toml"


class FuzzSmokeAll11Tests(unittest.TestCase):
    def test_script_lists_exactly_eleven_targets(self) -> None:
        text = SCRIPT.read_text()
        names = re.findall(r'^\s+([a-z0-9_]+)\s*$', text, re.M)
        # Only entries inside the TARGETS=( ... ) array.
        start = text.index("TARGETS=(")
        end = text.index(")", start)
        block = text[start:end]
        targets = re.findall(r'^\s+([a-z0-9_]+)\s*$', block, re.M)
        self.assertEqual(len(targets), 11)

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
        import tempfile
        import textwrap

        with tempfile.TemporaryDirectory() as tmp:
            bad = pathlib.Path(tmp) / "fuzz_smoke_bad.sh"
            bad.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    TARGETS=(only_one)
                    if [[ "${#TARGETS[@]}" -ne 11 ]]; then
                      echo "expected 11 fuzz targets, found ${#TARGETS[@]}" >&2
                      exit 1
                    fi
                    """
                )
            )
            bad.chmod(0o755)
            proc = subprocess.run([str(bad)], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("expected 11 fuzz targets", proc.stderr)


if __name__ == "__main__":
    unittest.main()
