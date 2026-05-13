"""
Regression tests for DEFEND-20260424T1530-fish-vqgan-encode-audio-memory-bomb
(secondary: fragile len>255 path/bytes discrimination heuristic in reference_loader.py).

Tests verify that ``ReferenceLoader.load_audio`` correctly routes:
  - raw bytes  -> BytesIO wrapper (raw audio payload)
  - valid path string -> passed directly to torchaudio as path
  - non-existent path / overlong string -> BytesIO wrapper + warning log

torchaudio.load is mocked so tests run without GPU/audio libraries.
"""

import io
import os
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loader():
    """Return a ReferenceLoader with torchaudio backend detection bypassed."""
    with patch("torchaudio.list_audio_backends", return_value=["soundfile"]):
        from fish_speech.inference_engine.reference_loader import ReferenceLoader
        loader = ReferenceLoader()
    return loader


def _fake_waveform():
    """Return (waveform_mock, sample_rate) compatible with torchaudio.load mock.

    We return a simple namespace with .shape and .squeeze() so that reference_loader's
    post-load waveform handling doesn't crash.
    """
    import types

    arr = np.zeros((1, 16000), dtype=np.float32)

    def squeeze():
        return types.SimpleNamespace(numpy=lambda: arr.squeeze())

    wf = types.SimpleNamespace(
        shape=(1, 16000),
        squeeze=squeeze,
    )
    return wf, 16000


# ---------------------------------------------------------------------------
# Bytes input -> always routed as raw audio (BytesIO)
# ---------------------------------------------------------------------------


class TestBytesAlwaysRawAudio:
    """bytes input must be treated as raw audio regardless of length."""

    def test_short_bytes_treated_as_raw_audio(self):
        """100 random bytes that cannot form a valid path -> routed to BytesIO."""
        loader = _make_loader()
        audio_bytes = os.urandom(100)

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(audio_bytes, sr=16000)

        assert isinstance(captured["src"], io.BytesIO), (
            "bytes input must be wrapped in BytesIO, not passed as a path"
        )
        assert captured["src"].read() == audio_bytes

    def test_long_bytes_treated_as_raw_audio(self):
        """512+ bytes -> still routed as BytesIO (not misidentified as a path)."""
        loader = _make_loader()
        audio_bytes = os.urandom(600)

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(audio_bytes, sr=16000)

        assert isinstance(captured["src"], io.BytesIO), (
            "Long bytes must still be wrapped in BytesIO"
        )

    def test_empty_bytes_treated_as_raw_audio(self):
        """Empty bytes -> BytesIO (degenerate but should not crash discrimination)."""
        loader = _make_loader()

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(b"", sr=16000)

        assert isinstance(captured["src"], io.BytesIO)


# ---------------------------------------------------------------------------
# Valid file path string -> passed directly to torchaudio
# ---------------------------------------------------------------------------


class TestValidFilePathString:
    """A str that points to an existing file must be passed as a path string."""

    def test_valid_file_path_passed_as_string(self, tmp_path):
        """A short str pointing to a real file -> torchaudio receives the path string."""
        loader = _make_loader()
        # Create a real file so Path.is_file() returns True.
        audio_file = tmp_path / "sample.wav"
        audio_file.write_bytes(b"\x52\x49\x46\x46" + b"\x00" * 100)

        path_str = str(audio_file)
        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(path_str, sr=16000)

        assert captured["src"] == path_str, (
            "Valid file path must be passed directly to torchaudio, not wrapped in BytesIO"
        )

    def test_path_under_512_chars_accepted(self, tmp_path):
        """Path length < 512 chars to a real file -> passes discrimination."""
        loader = _make_loader()
        audio_file = tmp_path / ("a" * 10 + ".wav")
        audio_file.write_bytes(b"\x00" * 64)
        path_str = str(audio_file)
        assert len(path_str) < 512

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(path_str, sr=16000)

        assert captured["src"] == path_str


# ---------------------------------------------------------------------------
# Non-existent / overlong string -> BytesIO fallback + warning log
# ---------------------------------------------------------------------------


class TestAmbiguousStringFallback:
    """A str that doesn't resolve to a file -> BytesIO + warning logged."""

    def test_nonexistent_path_falls_back_to_bytesio(self):
        """A short str that is NOT an existing file -> BytesIO + warning."""
        loader = _make_loader()
        bogus_path = "/no/such/file/ever.wav"

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load), \
             patch("fish_speech.inference_engine.reference_loader.logger") as mock_log:
            loader.load_audio(bogus_path, sr=16000)

        assert isinstance(captured["src"], io.BytesIO), (
            "Non-existent path must fall back to BytesIO"
        )
        mock_log.warning.assert_called()

    def test_overlong_string_falls_back_to_bytesio(self):
        """A str of length >= 512 -> BytesIO + warning (regardless of filesystem)."""
        loader = _make_loader()
        long_str = "x" * 512  # exactly at boundary

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load), \
             patch("fish_speech.inference_engine.reference_loader.logger") as mock_log:
            loader.load_audio(long_str, sr=16000)

        assert isinstance(captured["src"], io.BytesIO), (
            "Overlong string (>=512 chars) must be treated as inline data"
        )
        mock_log.warning.assert_called()
        warning_msg = str(mock_log.warning.call_args_list)
        assert "treating as inline data" in warning_msg

    def test_reference_name_string_falls_back_to_bytesio(self):
        """Short text reference name (valid chars, no path) -> BytesIO + warning."""
        loader = _make_loader()
        ref_name = "my-voice-ref-001"  # looks like a reference ID, not a file path

        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load), \
             patch("fish_speech.inference_engine.reference_loader.logger") as mock_log:
            loader.load_audio(ref_name, sr=16000)

        assert isinstance(captured["src"], io.BytesIO)
        mock_log.warning.assert_called()


# ---------------------------------------------------------------------------
# Old heuristic regression: the 256-byte boundary must no longer matter
# ---------------------------------------------------------------------------


class TestOldHeuristicRegression:
    """Bytes payloads of exactly 255 and 256 bytes must both go to BytesIO.

    The old heuristic (len > 255) would misroute 256-byte payloads for bytes
    input because Path(bytes) raises TypeError in Python 3.12+.  The new code
    uses isinstance(bytes) as the first branch, so all byte payloads go to
    BytesIO unconditionally.
    """

    def _run(self, payload):
        loader = _make_loader()
        captured = {}

        def fake_torchaudio_load(src, backend):
            captured["src"] = src
            return _fake_waveform()

        with patch("torchaudio.load", side_effect=fake_torchaudio_load):
            loader.load_audio(payload, sr=16000)

        return captured["src"]

    def test_255_bytes_to_bytesio(self):
        result = self._run(b"\x00" * 255)
        assert isinstance(result, io.BytesIO)

    def test_256_bytes_to_bytesio(self):
        result = self._run(b"\x00" * 256)
        assert isinstance(result, io.BytesIO)

    def test_512_bytes_to_bytesio(self):
        result = self._run(b"\x00" * 512)
        assert isinstance(result, io.BytesIO)
