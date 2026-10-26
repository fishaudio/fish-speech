"""Unit tests for the MCP wrapper; no Torch model or live API required."""

import base64
import io
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import tools.mcp_server as server
from fishapi import FishAPIError


def wav_bytes(seconds: float = 0.25, rate: int = 44100) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeAPI:
    base_url = "http://fake:8080"

    def __init__(self):
        self.calls = []

    def health(self):
        return {"status": "ok"}

    def tts_defaults(self):
        return {"model_family": "drama3", "temperature": 0.7}

    def list_voices(self):
        return ["alice", "bob"]

    def add_voice(self, voice_id, audio, text):
        self.calls.append(("add", voice_id, Path(audio).read_bytes(), text))
        return {"success": True, "message": "ok", "reference_id": voice_id}

    def delete_voice(self, voice_id):
        return {"success": True, "message": "ok", "reference_id": voice_id}

    def generate(self, text, *, voice=None, **options):
        self.calls.append(("generate", text, voice, options))
        return wav_bytes()


class MCPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"FISH_MCP_OUTPUT_DIR": self.temp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.api = FakeAPI()
        self.api_patch = mock.patch.object(server, "_api", return_value=self.api)
        self.api_patch.start()
        self.addCleanup(self.api_patch.stop)

    def test_health_and_voices(self):
        self.assertTrue(server.fish_health()["ok"])
        self.assertEqual(
            server.fish_voices_list(), {"count": 2, "voices": ["alice", "bob"]}
        )
        self.assertEqual(server.fish_defaults()["model_family"], "drama3")

    def test_tts_saves_wav_and_passes_options(self):
        result = server.fish_tts("Hello", voice="alice", seed=42, temperature=0.7)
        self.assertTrue(Path(result["path"]).is_file())
        self.assertEqual(result["duration_s"], 0.25)
        self.assertEqual(result["rtf"], result["elapsed_s"] / 0.25)
        kind, text, voice, options = self.api.calls[-1]
        self.assertEqual((kind, text, voice), ("generate", "Hello", "alice"))
        self.assertEqual(options["seed"], 42)
        self.assertEqual(options["temperature"], 0.7)

    def test_inline_reference_is_base64_encoded(self):
        audio = Path(self.temp.name) / "ref.wav"
        audio.write_bytes(wav_bytes(0.1))
        result = server.fish_tts(
            "Hello",
            reference_audio=str(audio),
            reference_text="Reference words",
            speaker=3,
        )
        self.assertTrue(Path(result["path"]).is_file())
        _, _, voice, options = self.api.calls[-1]
        self.assertIsNone(voice)
        reference = options["references"][0]
        self.assertEqual(base64.b64decode(reference["audio"]), audio.read_bytes())
        self.assertEqual(reference["text"], "Reference words")
        self.assertEqual(reference["speaker"], 3)

    def test_inline_reference_requires_transcript(self):
        audio = Path(self.temp.name) / "ref.wav"
        audio.write_bytes(wav_bytes(0.1))
        with self.assertRaises(ValueError):
            server.fish_tts("Hello", reference_audio=str(audio))

    def test_voice_and_reference_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            server.fish_tts(
                "Hello", voice="alice", reference_audio="x.wav", reference_text="x"
            )

    def test_output_name_cannot_escape_directory(self):
        result = server.fish_tts("Hello", output_name="../../escape.wav")
        self.assertEqual(Path(result["path"]).parent, Path(self.temp.name).resolve())
        self.assertNotIn("..", Path(result["path"]).name)

    def test_add_voice_reads_file(self):
        audio = Path(self.temp.name) / "ref.wav"
        audio.write_bytes(wav_bytes(0.1))
        result = server.fish_voice_add("alice", str(audio), "Hello")
        self.assertEqual(result["reference_id"], "alice")
        self.assertEqual(self.api.calls[-1][1], "alice")

    def test_api_failure_is_a_tool_error(self):
        with mock.patch.object(self.api, "health", side_effect=FishAPIError("down")):
            with self.assertRaises(RuntimeError):
                server.fish_health()

    def test_invalid_sampling_bounds(self):
        for kwargs in ({"temperature": 0.0}, {"top_p": 1.5}, {"speaker": 16}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    server.fish_tts("Hello", **kwargs)


if __name__ == "__main__":
    unittest.main()
