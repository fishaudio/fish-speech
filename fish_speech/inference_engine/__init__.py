import gc
import queue
from typing import Generator, Literal

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.fish_omni.generate import (
    S3_DEFAULT_TEMPERATURE,
    S3_DEFAULT_TOP_K,
    S3_DEFAULT_TOP_P,
    SPEAKER_TAG_PATTERN,
    UNICODE_WHITE_SPACE,
)
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest

ModelFamily = Literal["s2", "s3"]

# Used for every sampling field a request leaves unset (None).
SAMPLING_DEFAULTS: dict[ModelFamily, dict[str, int | float]] = {
    "s2": dict(
        max_new_tokens=1024,
        top_p=0.8,
        top_k=30,
        repetition_penalty=1.1,
        temperature=0.8,
    ),
    # S3 chunks long text itself (max_new_tokens=0: 600 frames per chunk, 2048 with
    # single_chunk) and has built-in repetition-aware sampling instead of a
    # repetition penalty.
    "s3": dict(
        max_new_tokens=0,
        top_p=S3_DEFAULT_TOP_P,
        top_k=S3_DEFAULT_TOP_K,
        repetition_penalty=1.0,
        temperature=S3_DEFAULT_TEMPERATURE,
    ),
}


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
        model_family: ModelFamily = "s2",
    ) -> None:

        super().__init__()

        if model_family not in SAMPLING_DEFAULTS:
            raise ValueError(f"Unknown model family: {model_family!r}")

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile
        self.model_family = model_family

    def load_references(self, req: ServeTTSRequest) -> tuple[list, list[str]]:
        """Reference codes and transcripts, by reference_id or else by the uploaded
        references, whose speaker field tags their transcripts."""
        if req.reference_id is not None:
            return self.load_by_id(req.reference_id, req.use_memory_cache)
        if not req.references:
            return [], []
        prompt_tokens, prompt_texts = self.load_by_hash(
            req.references, req.use_memory_cache
        )
        prompt_texts = [
            self.tag_reference_text(text, ref.speaker)
            for ref, text in zip(req.references, prompt_texts)
        ]
        return prompt_tokens, prompt_texts

    def tag_reference_text(self, text: str, speaker: int | None) -> str:
        """Prefix <|speaker:N|> unless speaker is None or a tag already leads text."""
        if speaker is None or SPEAKER_TAG_PATTERN.match(
            text.lstrip(UNICODE_WHITE_SPACE)
        ):
            return text
        # The models' own formats: S3 puts a space after the tag, S2-Pro does not.
        separator = " " if self.model_family == "s3" else ""
        return f"<|speaker:{speaker}|>{separator}{text}"

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        prompt_tokens, prompt_texts = self.load_references(req)

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
            text=req.text,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            single_chunk=req.single_chunk,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            **self.resolve_sampling(req),
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

    def resolve_sampling(self, req: ServeTTSRequest) -> dict[str, int | float]:
        """Request sampling fields, falling back to the model family's defaults."""
        params = dict(SAMPLING_DEFAULTS[self.model_family])
        for name in ("max_new_tokens", "top_p", "repetition_penalty", "temperature"):
            value = getattr(req, name)
            if value is not None:
                params[name] = value

        if self.model_family == "s3" and req.repetition_penalty not in (None, 1.0):
            logger.warning(
                "S3 ignores repetition_penalty (repetition-aware sampling is built in)"
            )
        return params

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
