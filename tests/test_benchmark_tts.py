import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from tools import benchmark_tts as bench


class BenchmarkTest(unittest.TestCase):
    def test_header_is_not_audio(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b"\0\0" * 4)
        data = out.getvalue()
        self.assertFalse(bench.has_wav_sample(data[:44]))
        self.assertFalse(bench.has_wav_sample(data[:45]))
        self.assertTrue(bench.has_wav_sample(data[:46]))

    def test_stream_reader_measures_audio_separately(self):
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(44100)
            wav.writeframes(b"\0\0" * 4)
        data = out.getvalue()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read1(self, size):
                return next(self.chunks)

        response = Response()
        response.chunks = iter([data[:44], data[44:], b""])
        args = SimpleNamespace(
            streaming=True,
            seed=42,
            voice="narrator",
            temperature=None,
            top_p=None,
            chunk_length=100,
            single_chunk=False,
            timeout=1,
            save_dir=None,
        )
        with (
            patch.object(bench, "post_json", return_value=response),
            patch.object(bench.time, "perf_counter", side_effect=[0, 1, 2, 3]),
        ):
            result = bench.run_once("http://test", "hello", args)
        self.assertEqual(result["first_byte_s"], 1)
        self.assertEqual(result["first_audio_s"], 2)
        self.assertEqual(result["wall_s"], 3)
