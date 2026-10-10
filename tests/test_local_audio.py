"""Regression coverage for reference uploads from macOS Voice Memos."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from local_audio import decode_reference_audio


class ReferenceAudioTests(unittest.TestCase):
    def test_stereo_wav_is_downmixed_and_resampled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.wav"
            samples = np.column_stack((np.full(4800, 0.2), np.full(4800, 0.4)))
            sf.write(path, samples, 48000, subtype="FLOAT")
            audio = decode_reference_audio(path, 44100)
            self.assertEqual(audio.shape, (4410,))
            self.assertEqual(audio.dtype, np.float32)
            np.testing.assert_allclose(audio[100:-100], 0.3, atol=0.001)

    def test_m4a_upload_bytes_and_file_path_decode_identically(self):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            self.skipTest("FFmpeg is not installed")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "reference.m4a"
            subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=0.25",
                    "-ar",
                    "48000",
                    "-ac",
                    "2",
                    "-c:a",
                    "aac",
                    str(path),
                ],
                check=True,
            )
            data = path.read_bytes()
            # Exercise a seekable M4A with metadata stored after its media.
            self.assertGreater(data.index(b"moov"), data.index(b"mdat"))
            from_path = decode_reference_audio(str(path), 44100)
            from_upload = decode_reference_audio(data, 44100)
            np.testing.assert_array_equal(from_path, from_upload)
            self.assertEqual(from_upload.ndim, 1)
            self.assertTrue(np.isfinite(from_upload).all())
            self.assertGreater(from_upload.size, 44100 * 0.2)
            self.assertLess(from_upload.size, 44100 * 0.35)
            self.assertGreater(np.max(np.abs(from_upload)), 0.01)

    def test_empty_and_invalid_uploads_have_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            decode_reference_audio(b"", 44100)
        with self.assertRaisesRegex(ValueError, "could not be decoded"):
            decode_reference_audio(b"this is not an audio recording", 44100)


if __name__ == "__main__":
    unittest.main()
