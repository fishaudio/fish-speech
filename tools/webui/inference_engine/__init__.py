import gc
import queue
from typing import Generator, Tuple, Union

import numpy as np
import torch
from loguru import logger

from fish_speech.i18n import i18n
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps, set_seed
from tools.api_server import decode_vq_tokens   # WTF
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from tools.schema import ServeTTSRequest
from tools.webui.inference_engine.reference_loader import ReferenceLoader
from tools.webui.inference_engine.utils import build_html_error_message, wav_chunk_header


class InferenceEngine(ReferenceLoader):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: torch.nn.Module,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Union[Generator, Tuple]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )

        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

        # If streaming, send the header
        if req.streaming:
            yield wav_chunk_header()

        segments = []

        while True:
            # Get the response from the LLAMA model
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                yield None, None, build_html_error_message(wrapped_result.response)
                break

            # Check the response type
            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                segment = self.get_audio_segment(result)

                if req.streaming:   # Used only by the API server
                    yield (segment * 32768).astype(np.int16).tobytes()
                else:
                    segments.append(segment)
            else:
                break
        
        # If streaming, we need to return the final audio
        if req.streaming:
            return

        # Edge case: no audio generated
        if len(segments) == 0:
            return (
                None,
                None,
                build_html_error_message(
                    i18n("No audio generated, please check the input text.")
                ),
            )

        # No matter streaming or not, we need to return the final audio
        audio = np.concatenate(segments, axis=0)
        yield None, (self.decoder_model.spec_transform.sample_rate, audio), None

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

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
            text=(
                req.text
                if not req.normalize
                else ChnNormedText(raw_text=req.text).normalize()
            ),
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            max_length=4096,
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
            fake_audios = decode_vq_tokens(
                decoder_model=self.decoder_model,
                codes=result.codes,
            )

        # Convert the audio to numpy
        return fake_audios.float().cpu().numpy()
