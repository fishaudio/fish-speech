import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.utils.schema import ServeTTSRequest
from tools.webui.inference import inference_wrapper


class FakeCodec(torch.nn.Module):
    def __init__(self, value=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.decoder = torch.nn.Identity()
        self.sample_rate = 44100
        self.value = value
        self.calls = 0

    @property
    def device(self):
        return self.weight.device

    def encode(self, audios, lengths):
        self.calls += 1
        return torch.full((1, 10, 3), self.value, dtype=torch.long), lengths


class RoutingTests(unittest.TestCase):
    def manager(self):
        manager = VQManager()
        manager.decoder_model = FakeCodec()
        manager.load_audio = lambda audio, sr: np.zeros(64, dtype=np.float32)
        manager._reference_codec_config = ("fixture-checkpoint", "fixture-config")
        return manager

    def test_default_gpu_route_uses_original_model_without_loading_or_moving_it(self):
        manager = self.manager()
        with (
            patch("fish_speech.inference_engine.vq_manager.DAC", FakeCodec),
            patch.object(
                manager.decoder_model, "to", wraps=manager.decoder_model.to
            ) as move,
            patch("fish_speech.models.dac.inference.load_model") as load,
        ):
            tokens = manager.encode_reference(b"fixture", True)
        self.assertEqual(tokens.shape, (10, 3))
        self.assertEqual(manager.decoder_model.calls, 1)
        move.assert_not_called()
        load.assert_not_called()
        self.assertEqual(ServeTTSRequest(text="test").reference_encoder_device, "gpu")

    def test_cpu_copy_is_loaded_once_and_original_decoder_is_untouched(self):
        manager = self.manager()
        cpu = FakeCodec(value=2)
        with (
            patch("fish_speech.inference_engine.vq_manager.DAC", FakeCodec),
            patch(
                "fish_speech.models.dac.inference.load_model", return_value=cpu
            ) as load,
            patch.object(
                manager.decoder_model, "to", wraps=manager.decoder_model.to
            ) as move,
        ):
            first = manager.encode_reference(b"fixture", True, encoder_device="cpu")
            second = manager.encode_reference(b"fixture", True, encoder_device="cpu")
        self.assertTrue(torch.equal(first, second))
        load.assert_called_once_with(
            "fixture-config", "fixture-checkpoint", device="cpu"
        )
        move.assert_not_called()
        self.assertEqual(manager.decoder_model.calls, 0)
        self.assertTrue(hasattr(manager.decoder_model, "decoder"))
        self.assertFalse(hasattr(cpu, "decoder"))
        self.assertEqual(cpu.calls, 2)

    def test_cpu_without_metadata_fails_clearly(self):
        manager = self.manager()
        manager._reference_codec_config = None
        with self.assertRaisesRegex(ValueError, "checkpoint metadata"):
            manager.encode_reference(b"fixture", True, encoder_device="cpu")

    def test_legacy_webui_positional_contract_and_new_keyword_options(self):
        engine = SimpleNamespace(
            inference=Mock(
                return_value=[SimpleNamespace(code="final", audio=(44100, np.ones(8)))]
            )
        )
        args = ["text", "", None, "", 128, 200, 0.8, 1.1, 0.8, 0, "off", engine]
        self.assertEqual(len(inference_wrapper(*args)), 2)
        self.assertEqual(
            engine.inference.call_args.args[0].reference_encoder_device, "gpu"
        )
        inference_wrapper(
            *args,
            reference_encoder_device="cpu",
            save_disk_cache=True,
            read_disk_cache=False,
            force_reencode=True,
        )
        req = engine.inference.call_args.args[0]
        self.assertEqual(req.reference_encoder_device, "cpu")
        self.assertTrue(req.save_disk_cache and req.force_reencode)
        self.assertFalse(req.read_disk_cache)

    def test_inference_entry_forwards_options_and_keeps_current_reference_text(self):
        engine = TTSInferenceEngine(queue.Queue(), FakeCodec(), torch.float32, False)
        engine.load_by_id = Mock(
            return_value=([torch.ones((10, 3), dtype=torch.long)], ["latest text"])
        )
        engine.send_Llama_request = Mock(
            side_effect=RuntimeError("stop after request dispatch")
        )
        req = ServeTTSRequest(
            text="test",
            reference_id="fixture",
            reference_encoder_device="cpu",
            read_disk_cache=False,
            save_disk_cache=True,
            force_reencode=True,
        )
        with self.assertRaisesRegex(RuntimeError, "request dispatch"):
            next(engine.inference(req))
        engine.load_by_id.assert_called_once_with(
            "fixture",
            "off",
            encoder_device="cpu",
            read_disk_cache=False,
            save_disk_cache=True,
            force_reencode=True,
        )
        self.assertEqual(engine.send_Llama_request.call_args.args[2], ["latest text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
