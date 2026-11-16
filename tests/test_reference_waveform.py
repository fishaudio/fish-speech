from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.models.text2semantic.errors import InvalidInputError


@pytest.mark.parametrize("samples", [1, 8])
@pytest.mark.parametrize("channels", [1, 2])
def test_decoded_audio_remains_one_dimensional(tmp_path, samples, channels):
    audio = np.full((samples, channels), 0.25, dtype=np.float32)
    path = tmp_path / "sample.wav"
    sf.write(path, audio, 16000, subtype="FLOAT")
    loader = ReferenceLoader()
    loader.backend = "soundfile"
    result = loader.load_audio(str(path), 16000)
    assert result.shape == (samples,)
    np.testing.assert_allclose(result, 0.25)


def test_empty_decoded_audio_has_clear_error():
    loader = ReferenceLoader()
    with patch("torchaudio.load", return_value=(torch.empty(1, 0), 16000)):
        with pytest.raises(InvalidInputError, match="empty"):
            loader.load_audio(b"header-only", 16000)


def test_resampled_audio_remains_one_dimensional(tmp_path):
    path = tmp_path / "sample.wav"
    sf.write(path, np.full(32, 0.25, dtype=np.float32), 16000, subtype="FLOAT")
    loader = ReferenceLoader()
    loader.backend = "soundfile"
    assert loader.load_audio(str(path), 8000).shape == (16,)
