"""
Tests for inference_engine (get_audio_segment and related).
Run: pytest tests/test_inference_engine.py -v
"""
import unittest.mock as mock

import numpy as np
import pytest
import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import GenerateResponse


class TestGetAudioSegment:
    """get_audio_segment must use .detach() before .numpy() when inference_mode is not used."""

    def test_returns_numpy_when_decode_vq_returns_tensor_requiring_grad(self):
        """
        When decode_vq_tokens returns a tensor with requires_grad=True (e.g. without inference_mode),
        get_audio_segment must still return a numpy array without raising.
        Regression: previously .numpy() was called without .detach() and raised.
        """
        decoder_model = mock.Mock()
        decoder_model.device = torch.device("cpu")

        engine = TTSInferenceEngine(
            llama_queue=mock.Mock(),
            decoder_model=decoder_model,
            precision=torch.float32,
            compile=False,
        )
        # Simulate DAC output that might require grad when inference_mode is off
        fake_audio = torch.ones(100, requires_grad=True)
        engine.decode_vq_tokens = mock.Mock(return_value=fake_audio)

        result = GenerateResponse(
            action="sample",
            codes=torch.zeros(10, 5, dtype=torch.long),
        )

        out = engine.get_audio_segment(result)

        assert isinstance(out, np.ndarray), "get_audio_segment must return numpy array"
        assert out.shape == (100,)
        assert out.dtype == np.float32
        engine.decode_vq_tokens.assert_called_once()
