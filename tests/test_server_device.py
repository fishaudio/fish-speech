import unittest
from unittest.mock import patch

import torch

from tools.server.model_manager import resolve_model_device


class TestResolveModelDevice(unittest.TestCase):
    """An explicit --device must survive; only the historical ``cuda`` default is redirected."""

    @staticmethod
    def _resolve(device, cuda=False, mps=False):
        with (
            patch.object(torch.cuda, "is_available", return_value=cuda),
            patch.object(torch.backends.mps, "is_available", return_value=mps),
        ):
            return resolve_model_device(device)

    def test_keeps_an_explicit_non_cuda_device(self):
        for device in ("cpu", "npu", "npu:1", "xpu"):
            with self.subTest(device=device):
                self.assertEqual(self._resolve(device), device)

    def test_redirects_the_cuda_default_to_the_available_accelerator(self):
        self.assertEqual(self._resolve("cuda", cuda=True), "cuda")
        self.assertEqual(self._resolve("cuda", cuda=True, mps=True), "mps")
        self.assertEqual(self._resolve("cuda:1", cuda=True), "cuda:1")
        self.assertEqual(self._resolve("cuda"), "cpu")

    def test_unavailable_mps_falls_back_to_cpu(self):
        self.assertEqual(self._resolve("mps"), "cpu")
        self.assertEqual(self._resolve("mps", mps=True), "mps")

    def test_cpu_is_never_redirected(self):
        self.assertEqual(self._resolve("cpu", cuda=True, mps=True), "cpu")


if __name__ == "__main__":
    unittest.main()
