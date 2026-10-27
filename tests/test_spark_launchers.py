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


class UnifiedLauncherTest(unittest.TestCase):
    def invoke(self, *args, env=None, local_weights=False):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "start.sh").write_bytes((ROOT / "start.sh").read_bytes())
            (root / "start-api.sh").write_text('#!/bin/sh\nprintf "ARG:%s\\n" "$@"\n')
            (root / "start-api.sh").chmod(0o755)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name, body in {
                "uname": 'case "$1" in -s) echo Linux;; -m) echo aarch64;; esac',
                "uv": "echo SYNC >&2",
                "nvidia-smi": "exit 0",
            }.items():
                path = bin_dir / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            # Existing environment must not trigger a second model download.
            venv = root / "environments/dgx-spark/.venv/bin"
            venv.mkdir(parents=True)
            for name in ("python", "hf"):
                p = venv / name
                p.write_text("#!/bin/sh\nexit 93\n")
                p.chmod(0o755)
            if local_weights:
                weights = root / "checkpoints/s2-pro"
                weights.mkdir(parents=True)
                (weights / "model.safetensors").write_bytes(b"weights")
            clean = {k: v for k, v in os.environ.items() if not k.startswith("FISH_")}
            result = subprocess.run(
                ["bash", str(root / "start.sh"), *args],
                cwd=root,
                env={
                    **clean,
                    "PATH": str(bin_dir) + ":" + os.environ["PATH"],
                    "TORCHINDUCTOR_CACHE_DIR": str(root / "cache"),
                    "FISH_SKIP_WEBUI": "1",
                    **(env or {}),
                },
                capture_output=True,
                text=True,
            )
            forwarded = [
                line[4:]
                for line in result.stdout.splitlines()
                if line.startswith("ARG:")
            ]
            return result, forwarded

    def test_cache_default_even_with_old_local_weights(self):
        result, args = self.invoke(local_weights=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model", args)
        self.assertNotIn("--llama-checkpoint-path", args)

    def test_cli_drama_selection_does_not_compile_or_predownload(self):
        result, args = self.invoke(
            "--model", "fishaudio/drama-3", "--revision", "commit", "--offline"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--compile", args)
        self.assertEqual(args.count("--model"), 1)
        self.assertEqual(args[args.index("--model") + 1], "fishaudio/drama-3")
        self.assertIn("--revision", args)

    def test_local_path_leaves_codec_fallback_to_resolver(self):
        result, args = self.invoke("--llama-checkpoint-path", "/models/custom model")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--model", args)
        self.assertNotIn("--decoder-checkpoint-path", args)
        self.assertNotIn("--compile", args)
        self.assertIn("/models/custom model", args)

    def test_conflicting_sources_fail_early(self):
        result, _ = self.invoke(
            "--model", "fishaudio/s2-pro", "--llama-checkpoint-path", "/local"
        )
        self.assertNotEqual(result.returncode, 0)

    def test_drama_rejects_explicit_compile(self):
        result, _ = self.invoke("--model=fishaudio/drama-3", "--compile")
        self.assertNotEqual(result.returncode, 0)

    def test_cli_selection_overrides_environment(self):
        result, args = self.invoke("--model=drama-3", env={"FISH_MODEL": "/old/local"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index("--model") + 1], "fishaudio/drama-3")
        self.assertNotIn("--compile", args)

    def test_environment_local_selection_and_compile_opt_out(self):
        result, args = self.invoke(env={"FISH_MODEL": "/local path"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/local path", args)
        self.assertNotIn("--model", args)
        result, args = self.invoke(env={"FISH_COMPILE": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--compile", args)

    def test_empty_selector_fails_before_setup(self):
        for flag in ("--model=", "--llama-checkpoint-path="):
            result, _ = self.invoke(flag)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not be empty", result.stderr)
            self.assertNotIn("SYNC", result.stderr)

    def test_explicit_codec_and_api_options_are_preserved(self):
        result, args = self.invoke(
            "--llama-checkpoint-path=/local",
            "--decoder-checkpoint-path",
            "/codec path",
            "--listen",
            "127.0.0.1:9090",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/codec path", args)
        self.assertIn("127.0.0.1:9090", args)


if __name__ == "__main__":
    unittest.main()
