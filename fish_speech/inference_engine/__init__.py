import gc
import queue
import unicodedata
from hashlib import sha256
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import (
    ServeReferenceCompatibility,
    ServeReferencePayload,
    ServeTTSRequest,
)


def _normalize_reference_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def _build_prompt_fingerprint(
    reference_text: str,
    prompt_tokens: list[list[int]],
    artifact_schema_version: int = 1,
) -> str:
    normalized_text = _normalize_reference_text(reference_text)
    token_tensor = torch.tensor(prompt_tokens, dtype=torch.int32)
    shape_ascii = f"{token_tensor.shape[0]},{token_tensor.shape[1]}".encode("ascii")
    preimage = (
        f"v{artifact_schema_version}\n".encode("ascii")
        + normalized_text.encode("utf-8")
        + b"\nint32\n"
        + shape_ascii
        + b"\n"
        + token_tensor.contiguous().numpy().tobytes()
    )
    return f"sha256:{sha256(preimage).hexdigest()}"


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
        reference_compatibility: ServeReferenceCompatibility | None = None,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        if reference_compatibility is None:
            sample_rate_hz = getattr(self.decoder_model, "sample_rate", 0)
            quantizer = getattr(self.decoder_model, "quantizer", None)
            num_codebooks = getattr(quantizer, "n_codebooks", 0) + 1
            reference_compatibility = ServeReferenceCompatibility(
                artifact_schema_version=1,
                codec_checkpoint_sha256="sha256:unknown",
                decoder_config_name=type(self.decoder_model).__name__,
                text2semantic_checkpoint_sha256="sha256:unknown",
                tokenizer_sha256="sha256:unknown",
                num_codebooks=num_codebooks,
                semantic_begin_id=0,
                sample_rate_hz=sample_rate_hz,
            )
        self.reference_compatibility = reference_compatibility

    def resolve_reference_payload(
        self, reference_payload: ServeReferencePayload
    ) -> tuple[list[torch.Tensor], list[str], str]:
        expected = self.reference_compatibility.model_dump(mode="python")
        actual = reference_payload.compatibility.model_dump(mode="python")
        for field_name, expected_value in expected.items():
            if actual.get(field_name) != expected_value:
                raise ValueError(
                    f"reference_payload compatibility mismatch for '{field_name}'"
                )

        reference_fingerprint = _build_prompt_fingerprint(
            reference_payload.reference_text,
            reference_payload.prompt_tokens,
            artifact_schema_version=actual["artifact_schema_version"],
        )
        if (
            reference_payload.reference_fingerprint is not None
            and reference_payload.reference_fingerprint != reference_fingerprint
        ):
            raise ValueError("reference_payload fingerprint mismatch")

        normalized_text = _normalize_reference_text(reference_payload.reference_text)
        prompt_tokens = [
            torch.tensor(reference_payload.prompt_tokens, dtype=torch.long)
        ]
        prompt_texts = [normalized_text]
        return prompt_tokens, prompt_texts, reference_fingerprint

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        reference_source = req.effective_reference_source()
        if reference_source == "reference_payload":
            if req.references:
                logger.warning(
                    "Ignoring inline references because reference_payload takes precedence"
                )
            if ref_id is not None:
                logger.warning(
                    "Ignoring reference_id because reference_payload takes precedence"
                )
            prompt_tokens, prompt_texts, _ = self.resolve_reference_payload(
                req.reference_payload
            )
        elif reference_source == "references":
            if ref_id is not None:
                logger.warning(
                    "Ignoring reference_id because inline references take precedence"
                )
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )
        elif ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming:
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments = []

        while True:
            # Get the response from the LLAMA model
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break

            # Check the response type
            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    f"Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                segment = self.get_audio_segment(result)

                if req.streaming:  # Used only by the API server
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )
                segments.append(segment)
            else:
                break

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
