"""FastAPI app for the PyTorch backend (S2-Pro or Drama 3)."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import queue
import random
import struct
import wave
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from engine.torch.models.dac.inference import load_model as load_decoder_model
from engine.torch.models.dac.modded_dac import DAC
from engine.torch.models.text2semantic.errors import InvalidInputError
from engine.torch.models.text2semantic.fish_omni.generate import (
    SPEAKER_TAG_PATTERN,
    UNICODE_WHITE_SPACE,
    resolve_drama3_devices,
    resolve_drama3_dtype,
    resolve_epsilon_cutoff,
)
from engine.torch.models.text2semantic.fish_omni.model import is_fish_omni_checkpoint
from engine.torch.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    launch_thread_safe_queue,
    load_codec_model,
    resolve_codec_path,
)
from engine.torch.models.text2semantic.sampling import (
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TOP_K,
)
from engine.torch.serve.schema import (
    IGNORED_FIELDS,
    LOCAL_EXTENSIONS,
    SUPPORTED_FORMATS,
    ResolvedReference,
    TTSRequest,
)
from engine.torch.serve.speaker_cache import (
    cache_references,
    delete_entry,
    list_entries,
    resolve_references,
)

AMPLITUDE = 32768
_END = object()


@dataclass
class ServerConfig:
    """Startup options; ``from_env`` reads the ``FISH_*`` environment variables."""

    checkpoint_path: str = "checkpoints/s2-pro"
    codec_path: str | None = None
    decoder_config_name: str = "modded_dac_vq"
    device: str = "cuda"
    half: bool = False
    compile: bool = False
    reference_sampling: str | None = None
    host: str = "127.0.0.1"
    port: int = 8080

    @classmethod
    def from_env(cls) -> ServerConfig:
        def flag(name: str) -> bool:
            return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            checkpoint_path=os.getenv("FISH_CHECKPOINT_PATH", "checkpoints/s2-pro"),
            codec_path=os.getenv("FISH_CODEC_PATH") or None,
            decoder_config_name=os.getenv("FISH_DECODER_CONFIG", "modded_dac_vq"),
            device=os.getenv("FISH_DEVICE", "cuda"),
            half=flag("FISH_HALF"),
            compile=flag("FISH_COMPILE"),
            reference_sampling=os.getenv("FISH_REFERENCE_SAMPLING") or None,
            host=os.getenv("FISH_HOST", "127.0.0.1"),
            port=int(os.getenv("FISH_PORT", "8080")),
        )


@dataclass
class Runtime:
    """Everything the request handlers need, loaded once on startup."""

    llama_queue: queue.Queue[object]
    codec: DAC
    device: torch.device
    sample_rate: int
    precision: torch.dtype
    compile: bool
    model_family: Literal["s2", "drama3"]
    reference_sampling: str | None

    def close(self) -> None:
        # ``None`` is the worker thread's stop signal.
        self.llama_queue.put(None)


def load_runtime(config: ServerConfig) -> Runtime:
    precision = torch.half if config.half else torch.bfloat16
    model_family = (
        "drama3" if is_fish_omni_checkpoint(config.checkpoint_path) else "s2"
    )
    if config.reference_sampling is not None and model_family != "drama3":
        raise ValueError("FISH_REFERENCE_SAMPLING needs a Drama 3 checkpoint")

    codec_path = resolve_codec_path(config.checkpoint_path, config.codec_path)
    if model_family == "drama3":
        # The codec shares the tower's device (last one) and the model's dtype.
        devices, tower_device = resolve_drama3_devices(config.device)
        precision = resolve_drama3_dtype(precision, [*devices, tower_device])
        device, llama_device = tower_device, config.device
        codec = load_codec_model(codec_path, device, precision)
    else:
        device = config.device
        if torch.backends.mps.is_available():
            device = "mps"
        elif not torch.cuda.is_available():
            device = "cpu"
        llama_device = device
        codec = load_decoder_model(config.decoder_config_name, codec_path, device)

    llama_queue = launch_thread_safe_queue(
        checkpoint_path=config.checkpoint_path,
        device=llama_device,
        precision=precision,
        compile=config.compile,
        reference_sampling=config.reference_sampling,
    )
    sample_rate = (
        codec.spec_transform.sample_rate
        if hasattr(codec, "spec_transform")
        else codec.sample_rate
    )
    logger.info(f"Loaded {model_family} LLM + DAC codec on {device} ({precision})")
    return Runtime(
        llama_queue=llama_queue,
        codec=codec,
        device=next(codec.parameters()).device,
        sample_rate=sample_rate,
        precision=precision,
        compile=config.compile,
        model_family=model_family,
        reference_sampling=config.reference_sampling,
    )


def pcm16(waveform: np.ndarray) -> bytes:
    # Clip so that 1.0 does not wrap around to -32768.
    samples = np.clip(waveform * AMPLITUDE, -AMPLITUDE, AMPLITUDE - 1)
    return samples.astype("<i2").tobytes()


def wav_bytes(sample_rate: int, waveform: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16(waveform))
    return buffer.getvalue()


def wav_stream_header(sample_rate: int) -> bytes:
    """A 44-byte WAV header whose length is unknown until the stream ends."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
    header = buffer.getvalue()
    unknown = struct.pack("<I", 0xFFFFFFFF)
    return header[:4] + unknown + header[8:40] + unknown


def tag_reference_text(runtime: Runtime, text: str, speaker: int | None) -> str:
    if speaker is None or SPEAKER_TAG_PATTERN.match(
        text.lstrip(UNICODE_WHITE_SPACE)
    ):
        return text
    # Drama 3 puts a space after the tag, S2-Pro does not.
    separator = " " if runtime.model_family == "drama3" else ""
    return f"<|speaker:{speaker}|>{separator}{text}"


@torch.inference_mode()
def encode_references(
    runtime: Runtime, references: list[ResolvedReference]
) -> tuple[list[torch.Tensor], list[str]]:
    """Encode inline reference audio into VQ prompt codes (no caches or files)."""
    if not references:
        return [], []

    dtype = next(runtime.codec.parameters()).dtype
    prompt_tokens: list[torch.Tensor] = []
    prompt_text: list[str] = []
    for reference in references:
        try:
            audio_bytes = base64.b64decode(reference.audio)
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=True
            )
        except (binascii.Error, ValueError, RuntimeError) as exc:
            raise InvalidInputError(f"invalid reference audio: {exc}") from exc

        wav = torch.from_numpy(waveform.mean(axis=1))
        if sample_rate != runtime.sample_rate:
            wav = torchaudio.functional.resample(wav, sample_rate, runtime.sample_rate)
        audios = wav.to(device=runtime.device, dtype=dtype)[None, None, :]
        audio_lengths = torch.tensor(
            [audios.shape[-1]], device=runtime.device, dtype=torch.long
        )
        prompt_tokens.append(runtime.codec.encode(audios, audio_lengths)[0][0])
        prompt_text.append(
            tag_reference_text(runtime, reference.text, reference.speaker)
        )
    return prompt_tokens, prompt_text


def build_generate_request(
    runtime: Runtime,
    req: TTSRequest,
    prompt_tokens: list[torch.Tensor],
    prompt_text: list[str],
) -> dict[str, object]:
    params: dict[str, object] = {
        "max_new_tokens": req.max_new_tokens,
        "top_p": req.top_p,
        "top_k": DEFAULT_TOP_K if req.top_k is None else req.top_k,
        "repetition_penalty": DEFAULT_REPETITION_PENALTY,
        "temperature": req.temperature,
    }

    if runtime.model_family == "drama3":
        # Validation only: generate_drama3 resolves the value itself.
        resolve_epsilon_cutoff(runtime.reference_sampling, req.epsilon_cutoff)
        params["epsilon_cutoff"] = req.epsilon_cutoff
    elif req.epsilon_cutoff:
        raise InvalidInputError(
            "epsilon_cutoff applies to Drama 3 reference sampling only"
        )
    if params["top_k"] == 0 and runtime.reference_sampling is None:
        raise InvalidInputError(
            "top_k=0 requires a Drama 3 checkpoint with FISH_REFERENCE_SAMPLING"
        )

    if req.seed is not None:
        seed = min(abs(req.seed), 1 << 31)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Set seed: {req.seed}")

    return dict(
        device=runtime.device,
        text=req.text,
        compile=runtime.compile,
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        single_chunk=req.single_chunk,
        prompt_tokens=prompt_tokens,
        prompt_text=prompt_text,
        **params,
    )


@torch.inference_mode()
def iter_segments(runtime: Runtime, generate_request: dict[str, object]) -> Iterator[np.ndarray]:
    """Submit one generation request and yield decoded audio chunks in order."""
    response_queue: queue.Queue[object] = queue.Queue()
    runtime.llama_queue.put(
        GenerateRequest(request=generate_request, response_queue=response_queue)
    )

    while True:
        wrapped = response_queue.get()
        if wrapped.status == "error":
            error = wrapped.response
            raise (
                error
                if isinstance(error, Exception)
                else RuntimeError("Generation failed")
            )
        result = wrapped.response
        if not isinstance(result, GenerateResponse):
            raise TypeError(f"Expected GenerateResponse, got {type(result).__name__}")
        if result.action == "next":
            break

        codes = result.codes.to(runtime.device)
        logger.info(f"VQ features: {codes.shape}")
        # torch.autocast is unsupported on MPS.
        with (
            nullcontext()
            if torch.backends.mps.is_available()
            else torch.autocast(runtime.device.type, runtime.precision)
        ):
            segment = runtime.codec.from_indices(codes[None])[0].squeeze()
        yield segment.float().cpu().numpy()


async def stream_wav(
    sample_rate: int, segments: Iterator[np.ndarray], include_header: bool = True
) -> AsyncIterator[bytes]:
    if include_header:
        yield wav_stream_header(sample_rate)
    while True:
        try:
            segment = await asyncio.to_thread(next, segments, _END)
        except Exception:  # noqa: BLE001 - headers are sent; only logging is possible
            logger.exception("Generation failed while streaming")
            return
        if segment is _END:
            return
        yield pcm16(segment)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    cfg = config or ServerConfig.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = load_runtime(cfg)
        try:
            yield
        finally:
            app.state.runtime.close()

    app = FastAPI(
        title="fish-speech torch engine", version="3.0.0", lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = cfg

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        runtime: Runtime | None = getattr(request.app.state, "runtime", None)
        return {
            "status": "ok" if runtime else "loading",
            "model_family": runtime.model_family if runtime else None,
            "sample_rate": runtime.sample_rate if runtime else None,
        }

    @app.get("/v1/tts/defaults")
    def defaults(request: Request) -> dict[str, object]:
        runtime: Runtime | None = getattr(request.app.state, "runtime", None)
        return {
            "model_family": runtime.model_family if runtime else None,
            "defaults": TTSRequest.model_construct().model_dump(
                exclude={"text"}, exclude_none=True
            ),
            "formats": list(SUPPORTED_FORMATS),
            "ignored_fields": list(IGNORED_FIELDS),
            "local_extensions": list(LOCAL_EXTENSIONS),
            "reference_id_source": ".app_cache",
        }

    @app.get("/v1/speaker-cache")
    def speaker_cache() -> dict[str, object]:
        return {"entries": list_entries()}

    @app.delete("/v1/speaker-cache/{entry_id}")
    def delete_speaker_cache(entry_id: str) -> dict[str, str]:
        if not delete_entry(entry_id):
            raise HTTPException(404, "cache entry not found")
        return {"status": "ok"}

    @app.post("/v1/tts")
    def tts(req: TTSRequest, request: Request) -> Response:
        runtime: Runtime = request.app.state.runtime
        if req.format not in SUPPORTED_FORMATS:
            raise HTTPException(
                400, f"local server cannot encode {req.format}; use wav or pcm"
            )

        try:
            references = resolve_references(req)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        # Cache failures must not break synthesis.
        try:
            cache_references(references)
        except OSError:
            logger.exception("Could not cache speaker references")

        try:
            prompt_tokens, prompt_text = encode_references(runtime, references)
            generate_request = build_generate_request(
                runtime, req, prompt_tokens, prompt_text
            )
        except InvalidInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Invalid TTS request")
            raise HTTPException(400, str(exc)) from exc

        segments = iter_segments(runtime, generate_request)
        raw_pcm = req.format == "pcm"
        media_type = "audio/pcm" if raw_pcm else "audio/wav"
        if req.streaming:
            return StreamingResponse(
                stream_wav(
                    runtime.sample_rate, segments, include_header=not raw_pcm
                ),
                media_type=media_type,
            )

        try:
            parts = list(segments)
        except InvalidInputError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(500, str(exc)) from exc
        if not parts:
            raise HTTPException(500, "No audio generated, please check the input text.")
        audio = np.concatenate(parts)
        content = pcm16(audio) if raw_pcm else wav_bytes(runtime.sample_rate, audio)
        return Response(content=content, media_type=media_type)

    return app


app = create_app()
