"""Exercise launcher argument forwarding without installing CUDA or loading models."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SparkLaunchersTest(unittest.TestCase):
    def test_api_selects_spark_environment_without_forcing_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            uv = directory / "uv"
            uv.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\n')
            uv.chmod(0o755)
            result = subprocess.run(
                [
                    "sh",
                    str(ROOT / "start-api.sh"),
                    "--llama-checkpoint-path",
                    "checkpoints/drama 3",
                    "--listen",
                    "127.0.0.1:9090",
                ],
                cwd=directory,
                env={
                    **os.environ,
                    "PATH": f"{directory}:" + os.environ["PATH"],
                    "TORCHINDUCTOR_CACHE_DIR": str(directory / "cache"),
                },
                text=True,
                capture_output=True,
                check=True,
            )
            args = result.stdout.splitlines()
            self.assertEqual(args[0], str(ROOT))
            self.assertNotIn("--compile", args)
            self.assertIn("--locked", args)
            self.assertEqual(
                args[args.index("--project") + 1], "environments/dgx-spark"
            )
            self.assertEqual(args[args.index("--extra") + 1], "cu130")
            self.assertIn("tools/api_server.py", args)
            self.assertEqual(
                args[-4:],
                [
                    "--llama-checkpoint-path",
                    "checkpoints/drama 3",
                    "--listen",
                    "127.0.0.1:9090",
                ],
            )

    def test_webui_reports_missing_build_before_starting_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ("start-webui.sh", "start-api.sh"):
                (directory / name).write_bytes((ROOT / name).read_bytes())
                (directory / name).chmod(0o755)
            result = subprocess.run(
                ["sh", str(directory / "start-webui.sh")],
                cwd=directory,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("npm ci", result.stderr)
            self.assertIn("npm run build", result.stderr)


if __name__ == "__main__":
    unittest.main()
