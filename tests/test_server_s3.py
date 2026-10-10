import queue
import unittest
from unittest.mock import patch

import numpy as np
import torch
from kui.asgi import HTTPException

from fish_speech.inference_engine import SAMPLING_DEFAULTS, TTSInferenceEngine
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.models.text2semantic import inference
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper
from tools.server.model_utils import batch_vqgan_decode


class FakeCodec(torch.nn.Module):
    """Stands in for the DAC: one code frame -> frame_length samples."""

    frame_length = 4
    sample_rate = 16

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

    @property
    def device(self) -> torch.device:
        return self.weight.device

    def from_indices(self, codes: torch.Tensor) -> torch.Tensor:
        audio = (
            codes[:, :1].to(self.weight.dtype).repeat_interleave(self.frame_length, -1)
        )
        return audio + 1


def make_engine(family: str) -> TTSInferenceEngine:
    return TTSInferenceEngine(
        llama_queue=queue.Queue(),
        decoder_model=FakeCodec(),
        precision=torch.float16,
        compile=False,
        model_family=family,
    )


class SamplingDefaultsTest(unittest.TestCase):
    def test_unset_fields_take_the_family_defaults(self):
        req = ServeTTSRequest(text="hi")
        for family in ("s2", "s3"):
            params = make_engine(family).resolve_sampling(req)
            self.assertEqual(params, SAMPLING_DEFAULTS[family])
        self.assertEqual(
            make_engine("s3").resolve_sampling(req),
            dict(
                max_new_tokens=0,
                top_p=0.9,
                top_k=30,
                repetition_penalty=1.0,
                temperature=1.0,
            ),
        )

    def test_explicit_fields_win(self):
        req = ServeTTSRequest(text="hi", max_new_tokens=64, top_p=0.5, temperature=0.3)
        params = make_engine("s3").resolve_sampling(req)
        self.assertEqual(
            (params["max_new_tokens"], params["top_p"], params["temperature"]),
            (64, 0.5, 0.3),
        )
        self.assertEqual(params["top_k"], 30)

    def test_s3_warns_about_repetition_penalty(self):
        with patch("fish_speech.inference_engine.logger") as log:
            make_engine("s3").resolve_sampling(
                ServeTTSRequest(text="hi", repetition_penalty=1.2)
            )
            make_engine("s2").resolve_sampling(
                ServeTTSRequest(text="hi", repetition_penalty=1.2)
            )
        self.assertEqual(log.warning.call_count, 1)

    def test_request_is_forwarded_with_resolved_sampling(self):
        engine = make_engine("s3")
        engine.send_Llama_request(ServeTTSRequest(text="hi"), [], [])
        sent = engine.llama_queue.get_nowait().request
        self.assertEqual(sent["temperature"], 1.0)
        self.assertEqual(sent["max_new_tokens"], 0)
        self.assertEqual(sent["text"], "hi")
        self.assertFalse(sent["single_chunk"])
        engine.send_Llama_request(ServeTTSRequest(text="hi", single_chunk=True), [], [])
        self.assertTrue(engine.llama_queue.get_nowait().request["single_chunk"])

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            make_engine("s1")


class LaunchQueueTest(unittest.TestCase):
    def test_model_load_failure_is_raised_not_hung(self):
        with patch.object(inference, "init_model", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                inference.launch_thread_safe_queue("ckpt", "cpu", torch.float32)


class ErrorMappingTest(unittest.TestCase):
    def status_code_for(self, error: Exception) -> int:
        class Engine:
            def inference(self, req):
                yield InferenceResult(code="error", audio=None, error=error)

        with self.assertRaises(HTTPException) as caught:
            list(inference_wrapper(ServeTTSRequest(text="hi"), Engine()))
        return caught.exception.status_code

    def test_invalid_input_is_400_and_failures_are_500(self):
        self.assertEqual(
            self.status_code_for(
                ValueError("speaker tags must be <|speaker:0|> to <|speaker:15|>")
            ),
            400,
        )
        self.assertEqual(self.status_code_for(RuntimeError("cuda")), 500)


class VQGANDecodeTest(unittest.TestCase):
    def test_batch_decode_trims_each_item(self):
        codec = FakeCodec()
        codes = [
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 5, dtype=torch.long),
        ]
        audios = batch_vqgan_decode(codec, codes)
        self.assertEqual([a.shape[-1] for a in audios], [12, 20])
        self.assertTrue(all(a.dtype == np.float32 for a in audios))
        np.testing.assert_allclose(audios[0], 1.0)


if __name__ == "__main__":
    unittest.main()
