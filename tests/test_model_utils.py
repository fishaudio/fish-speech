import contextlib
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from tools.server.model_utils import _autocast_context, batch_vqgan_decode


class FakeCodec(torch.nn.Module):
    frame_length = 4
    sample_rate = 16

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def from_indices(self, codes: torch.Tensor) -> torch.Tensor:
        audio = codes[:, :1].to(torch.float32).repeat_interleave(self.frame_length, -1)
        return audio + 1


class ModelUtilsAutocastTest(unittest.TestCase):
    def test_cpu_device_returns_nullcontext(self):
        model = MagicMock()
        model.device = torch.device("cpu")
        ctx = _autocast_context(model)
        self.assertIsInstance(ctx, contextlib.AbstractContextManager)
        # Should execute cleanly without error
        with ctx:
            pass

    def test_missing_device_falls_back_to_nullcontext(self):
        model = MagicMock(spec=[])
        ctx = _autocast_context(model)
        self.assertIsInstance(ctx, contextlib.AbstractContextManager)
        with ctx:
            pass

    def test_batch_vqgan_decode_runs_on_cpu(self):
        codec = FakeCodec(device="cpu")
        codes = [
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 6, dtype=torch.long),
        ]
        audios = batch_vqgan_decode(codec, codes)
        self.assertEqual(len(audios), 2)
        self.assertEqual(audios[0].shape[-1], 12)
        self.assertEqual(audios[1].shape[-1], 24)
        self.assertTrue(all(a.dtype == np.float32 for a in audios))


if __name__ == "__main__":
    unittest.main()
