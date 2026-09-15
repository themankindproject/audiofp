#!/usr/bin/env python3
"""Unit tests for repository shell helpers (exit status + stderr propagation)."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "codec_robustness.sh"


class CodecRobustnessShellTests(unittest.TestCase):
    def test_missing_corpus_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "empty-repo"
            (root / "scripts").mkdir(parents=True)
            (root / "scripts" / "codec_robustness.sh").write_text(
                SCRIPT.read_text()
            )
            proc = subprocess.run(
                ["bash", str(root / "scripts" / "codec_robustness.sh")],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Missing:", proc.stdout + proc.stderr)

    def test_failed_cargo_propagates_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "fake-repo"
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            (root / "tests" / "assets").mkdir(parents=True)
            for rel in (
                "galway.flac",
                "galway.mp3",
                "galway.ogg",
                "galway.m4a",
                "galway.wav",
                "galway.aiff",
                "galway_stereo.mp3",
                "galway_stereo.flac",
                "freak.flac",
                "freak.mp3",
                "freak.ogg",
                "freak.m4a",
                "freak_8000hz.mp3",
                "freak_11025hz.mp3",
                "freak_16000hz.mp3",
                "freak_22050hz.mp3",
                "freak_32000hz.mp3",
                "freak_44100hz.mp3",
            ):
                (root / "tests" / "assets" / rel).write_bytes(b"")

            stub = textwrap.dedent(
                """\
                #!/usr/bin/env bash
                echo "injected cargo failure on stderr" >&2
                exit 42
                """
            )
            cargo_bin = root / "bin"
            cargo_bin.mkdir()
            (cargo_bin / "cargo").write_text(stub)
            (cargo_bin / "cargo").chmod(0o755)

            (scripts / "codec_robustness.sh").write_text(
                SCRIPT.read_text()
            )

            env = {
                "PATH": f"{cargo_bin}:/usr/bin:/bin",
                "HOME": subprocess.os.environ.get("HOME", "/tmp"),
            }
            proc = subprocess.run(
                ["bash", str(scripts / "codec_robustness.sh")],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 42)
            self.assertIn("injected cargo failure", proc.stdout + proc.stderr)
            self.assertIn("FAIL", proc.stdout)


if __name__ == "__main__":
    unittest.main()
