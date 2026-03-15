import gc
import os
import queue
import time
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
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):
    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:
        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile

    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        profile = os.getenv("FISH_PROFILE_INFERENCE", "0") in {
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        }
        req_tag = hex(id(req))[-6:]
        t_start = time.perf_counter()
        t_prev = t_start

        def _vram_gb() -> dict:
            if not torch.cuda.is_available():
                return {}
            return {
                "vram_alloc_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                "vram_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
                "vram_max_alloc_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
            }

        def _mark(event: str, **extra) -> None:
            nonlocal t_prev
            if not profile:
                return
            now = time.perf_counter()
            delta_ms = (now - t_prev) * 1000.0
            total_ms = (now - t_start) * 1000.0
            t_prev = now
            vram = _vram_gb()
            details = " ".join(f"{k}={v}" for k, v in (*vram.items(), *extra.items()))
            logger.info(
                "inference_timing req={} event={} delta_ms={:.1f} total_ms={:.1f} {}",
                req_tag,
                event,
                delta_ms,
                total_ms,
                details,
            )

        try:
            ref_id = req.reference_id
            prompt_tokens, prompt_texts = [], []
            if ref_id is not None:
                prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)
                _mark("ref_loaded", mode="id")
            elif req.references:
                prompt_tokens, prompt_texts = self.load_by_hash(
                    req.references, req.use_memory_cache
                )
                _mark("ref_loaded", mode="hash", refs=len(req.references))

            # Set the random seed if provided
            if req.seed is not None:
                set_seed(req.seed)
                logger.warning(f"set seed: {req.seed}")

            # Get the symbolic tokens from the LLAMA model
            stream_tokens = getattr(req, "stream_tokens", False) or req.streaming
            # Back-pressure: worker waits for ack after each chunk so DAC decode and LLM don't run on GPU concurrently (avoids OOM on 32 GB).
            ack_queue = queue.Queue() if stream_tokens else None
            response_queue = self.send_Llama_request(
                req, prompt_tokens, prompt_texts, req_tag=req_tag, ack_queue=ack_queue
            )
            _mark("llama_queued")
            if stream_tokens:
                logger.info("stream: inference started (token streaming), req={}", req_tag)

            # Get the sample rate from the decoder model
            if hasattr(self.decoder_model, "spec_transform"):
                sample_rate = self.decoder_model.spec_transform.sample_rate
            else:
                sample_rate = self.decoder_model.sample_rate

            # If streaming, send the header
            if req.streaming:
                _mark("yield_header")
                yield InferenceResult(
                    code="header",
                    audio=(
                        sample_rate,
                        np.array(wav_chunk_header(sample_rate=sample_rate)),
                    ),
                    error=None,
                )

            segments = []
            seg_idx = 0

            while True:
                # Get the response from the LLAMA model
                if stream_tokens:
                    logger.info("stream: waiting for next chunk from queue, req={}", req_tag)
                wrapped_result = response_queue.get()
                _mark("queue_get")
                if stream_tokens:
                    action = getattr(wrapped_result.response, "action", None) if hasattr(wrapped_result.response, "action") else None
                    logger.info("stream: queue_get status={} action={} req={}", wrapped_result.status, action, req_tag)
                if wrapped_result.status == "error":
                    logger.error("stream: got error from worker req={} err={}", req_tag, wrapped_result.response)
                    _mark("yield_error")
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

                if not isinstance(wrapped_result.response, GenerateResponse):
                    raise TypeError(
                        "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                    )

                result = wrapped_result.response
                if result.action != "next":
                    if stream_tokens:
                        logger.info(
                            "stream: decoding segment seg_idx={} codes_shape={} req={}",
                            seg_idx + 1,
                            result.codes.shape if result.codes is not None else None,
                            req_tag,
                        )
                    _mark("decode_vq_start", segment_idx=seg_idx + 1, codes_frames=result.codes.shape[1] if result.codes is not None else 0)
                    try:
                        if result.codes is not None and not result.codes.is_cuda:
                            result = GenerateResponse(
                                result.action,
                                result.codes.to(self.decoder_model.device),
                                getattr(result, "text", None),
                            )
                        segment = self.get_audio_segment(result)
                    except Exception as seg_err:
                        if stream_tokens:
                            logger.exception(
                                "stream: get_audio_segment FAILED seg_idx={} codes_shape={} req={}: {}",
                                seg_idx + 1,
                                result.codes.shape if result.codes is not None else None,
                                req_tag,
                                seg_err,
                            )
                        raise
                    seg_idx += 1
                    _mark("segment_decoded", segment_idx=seg_idx, samples=len(segment))
                    if stream_tokens:
                        logger.info("stream: segment_decoded seg_idx={} samples={} req={}", seg_idx, len(segment), req_tag)

                    if req.streaming:
                        _mark("yield_segment", segment_idx=seg_idx)
                        yield InferenceResult(
                            code="segment",
                            audio=(sample_rate, segment),
                            error=None,
                        )
                    if ack_queue is not None:
                        ack_queue.put(None)
                    segments.append(segment)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    if stream_tokens:
                        logger.info("stream: got_next (end of stream), total_segments={} req={}", seg_idx, req_tag)
                    if ack_queue is not None:
                        ack_queue.put(None)
                    _mark("got_next")
                    break

            # Clean up the memory so next request starts near baseline (~14 GB); otherwise second request OOMs on 32 GB.
            if torch.cuda.is_available():
                wrapped_result = None
                result = None
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()

            if len(segments) == 0:
                _mark("yield_error_empty")
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=RuntimeError("No audio generated, please check the input text."),
                )
            else:
                audio = np.concatenate(segments, axis=0)
                _mark("yield_final", total_samples=len(audio), segments=len(segments))
                yield InferenceResult(
                    code="final",
                    audio=(sample_rate, audio),
                    error=None,
                )

            return None
        finally:
            # When generator is closed (client done), try to free VRAM again so next request does not OOM.
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()

    def send_Llama_request(
        self,
        req: ServeTTSRequest,
        prompt_tokens: list,
        prompt_texts: list,
        req_tag: str,
        ack_queue: queue.Queue | None = None,
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """

        # Prepare the request
        # When streaming API is used, enable token-level streaming for low TTFA
        stream_tokens = getattr(req, "stream_tokens", False) or req.streaming
        request = dict(
            device=self.decoder_model.device,
            req_tag=req_tag,
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
            stream_tokens=stream_tokens,
            stream_chunk_size=getattr(req, "stream_chunk_size", 20),
            ack_queue=ack_queue,
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

        # Convert the audio to numpy (detach: inference path has no grad, but we no longer use inference_mode)
        return segment.float().detach().cpu().numpy()
