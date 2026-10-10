import io
import queue
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from kui.asgi import HTTPException
from pydantic import ValidationError

from fish_speech.inference_engine import SAMPLING_DEFAULTS, TTSInferenceEngine
from fish_speech.inference_engine.utils import InferenceResult
from fish_speech.models.text2semantic import inference
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.generate import S3PromptTooLongError
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from tools.server.api_utils import encode_audio
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
        # The official Fish Audio SDK's defaults, for both families.
        sdk = dict(
            max_new_tokens=1024,
            top_p=0.7,
            top_k=30,
            repetition_penalty=1.2,
            temperature=0.7,
        )
        self.assertEqual(make_engine("s2").resolve_sampling(req), sdk)
        self.assertEqual(make_engine("s3").resolve_sampling(req), sdk)

    def test_defaults_for_clients(self):
        self.assertEqual(
            make_engine("s3").sampling_defaults(),
            {"model_family": "s3", "chunk_length": 200, **SAMPLING_DEFAULTS["s3"]},
        )
        self.assertEqual(make_engine("s2").sampling_defaults()["model_family"], "s2")

    def test_explicit_fields_win(self):
        req = ServeTTSRequest(text="hi", max_new_tokens=64, top_p=0.5, temperature=0.3)
        params = make_engine("s3").resolve_sampling(req)
        self.assertEqual(
            (params["max_new_tokens"], params["top_p"], params["temperature"]),
            (64, 0.5, 0.3),
        )
        self.assertEqual(params["top_k"], 30)

    def test_warns_that_repetition_penalty_is_ignored(self):
        with patch("fish_speech.inference_engine.logger") as log:
            for family in ("s3", "s2"):
                make_engine(family).resolve_sampling(
                    ServeTTSRequest(text="hi", repetition_penalty=1.3)
                )
            # The SDK sends its default, 1.2, with every request.
            make_engine("s3").resolve_sampling(
                ServeTTSRequest(text="hi", repetition_penalty=1.2)
            )
        self.assertEqual(log.warning.call_count, 2)

    def test_request_is_forwarded_with_resolved_sampling(self):
        engine = make_engine("s3")
        engine.send_Llama_request(ServeTTSRequest(text="hi"), [], [])
        sent = engine.llama_queue.get_nowait().request
        self.assertEqual(sent["temperature"], 0.7)
        self.assertEqual(sent["max_new_tokens"], 1024)
        self.assertEqual(sent["text"], "hi")
        self.assertFalse(sent["single_chunk"])
        engine.send_Llama_request(ServeTTSRequest(text="hi", single_chunk=True), [], [])
        self.assertTrue(engine.llama_queue.get_nowait().request["single_chunk"])

    def test_unknown_family(self):
        with self.assertRaises(ValueError):
            make_engine("s1")


class ReferenceSpeakerTest(unittest.TestCase):
    def references(self) -> list[ServeReferenceAudio]:
        return [
            ServeReferenceAudio(audio=b"a", text=" Hello.", speaker=2),
            ServeReferenceAudio(audio=b"b", text="<|speaker:5|> Hi.", speaker=1),
            ServeReferenceAudio(audio=b"c", text="Plain."),
        ]

    def load_texts(self, family: str, req: ServeTTSRequest) -> list[str]:
        engine = make_engine(family)
        with patch.object(engine, "encode_reference", return_value=torch.zeros(2, 3)):
            tokens, texts = engine.load_references(req)
        self.assertEqual(len(tokens), len(texts))
        return texts

    def test_speaker_tags_untagged_transcripts(self):
        req = ServeTTSRequest(text="hi", references=self.references())
        self.assertEqual(
            self.load_texts("s3", req),
            ["<|speaker:2|>  Hello.", "<|speaker:5|> Hi.", "Plain."],
        )
        self.assertEqual(
            self.load_texts("s2", req),
            ["<|speaker:2|> Hello.", "<|speaker:5|> Hi.", "Plain."],
        )
        self.assertEqual(self.load_texts("s3", ServeTTSRequest(text="hi")), [])

    def test_cached_audio_uses_the_current_transcript(self):
        engine = make_engine("s3")
        codes = torch.zeros(2, 3)
        with patch.object(engine, "encode_reference", return_value=codes) as encode:
            for text, speaker in (("Old.", None), ("New.", 3)):
                ref = ServeReferenceAudio(audio=b"a", text=text, speaker=speaker)
                req = ServeTTSRequest(
                    text="hi", references=[ref], use_memory_cache="on"
                )
                tokens, texts = engine.load_references(req)
        self.assertEqual(encode.call_count, 1)
        self.assertIs(tokens[0], codes)
        self.assertEqual(texts, ["<|speaker:3|> New."])

    def test_speaker_must_be_0_to_15(self):
        for speaker in (-1, 16, "1"):
            with self.assertRaises(ValidationError):
                ServeReferenceAudio(audio=b"a", text="Hello.", speaker=speaker)
        self.assertIsNone(ServeReferenceAudio(audio=b"a", text="Hello.").speaker)


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
                InvalidInputError(
                    "speaker tags must be <|speaker:0|> to <|speaker:15|>"
                )
            ),
            400,
        )
        self.assertEqual(self.status_code_for(S3PromptTooLongError("too long")), 400)
        # Other ValueErrors are internal failures, not the client's fault.
        self.assertEqual(
            self.status_code_for(ValueError("tokenizer/config mismatch")), 500
        )
        self.assertEqual(self.status_code_for(RuntimeError("cuda")), 500)


class EncodeAudioTest(unittest.TestCase):
    def test_every_request_format_encodes(self):
        audio = np.sin(np.linspace(0, 200, 4410)).astype(np.float32)
        pcm = encode_audio(audio, 44100, "pcm")
        decoded = np.frombuffer(pcm, np.int16) / 32768
        np.testing.assert_allclose(decoded, audio, atol=1e-4)
        wav, rate = sf.read(io.BytesIO(encode_audio(audio, 44100, "wav")))
        self.assertEqual((len(wav), rate), (4410, 44100))
        opus, rate = sf.read(io.BytesIO(encode_audio(audio, 44100, "opus")))
        self.assertEqual(rate, 48000)
        self.assertAlmostEqual(len(opus) / rate, 0.1, delta=0.02)
        self.assertGreater(len(encode_audio(audio, 44100, "mp3")), 0)


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
