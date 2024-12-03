import io
import json
import os
import queue
import re
import time
import traceback
import wave
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any

import librosa
import numpy as np
import ormsgpack
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from baize.datastructures import ContentType
from kui.asgi import (
    Body,
    FactoryClass,
    HTTPException,
    HttpRequest,
    HttpView,
    JSONResponse,
    Kui,
    OpenAPI,
    StreamResponse,
    request,
)
from kui.asgi.routing import MultimethodRoutes
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import struct
from threading import Lock

import httpx
from cachetools import LRUCache, cached
from funasr import AutoModel
from silero_vad import get_speech_timestamps, load_silero_vad

# from fish_speech.conversation import IM_END_TOKEN, SEMANTIC_TOKEN
from fish_speech.tokenizer import FishTokenizer, IM_END_TOKEN
from fish_speech.models.text2semantic.llama import BaseModelArgs

# from fish_speech.models.vqgan.lit_module import VQGAN
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps, set_seed
from tools.file import AUDIO_EXTENSIONS, audio_to_bytes, list_files, read_ref_text
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
    launch_thread_safe_queue_agent,
)
from tools.schema import (
    GLOBAL_NUM_SAMPLES,
    ASRPackRequest,
    ServeASRRequest,
    ServeASRResponse,
    ServeASRSegment,
    ServeAudioPart,
    ServeForwardMessage,
    ServeMessage,
    ServeRequest,
    ServeResponse,
    ServeStreamDelta,
    ServeStreamResponse,
    ServeTextPart,
    ServeTimedASRResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
    ServeVQPart,
)
from tools.vqgan.inference import load_model as load_decoder_model

global_lock = Lock()

# Whether to disable keepalive (which is helpful if the server is in the same cluster)
DISABLE_KEEPALIVE = os.getenv("DISABLE_KEEPALIVE", "false").lower() == "true"
async_client = httpx.AsyncClient(
    timeout=120, limits=httpx.Limits(keepalive_expiry=0 if DISABLE_KEEPALIVE else None)
)
backends = torchaudio.list_audio_backends()

if "ffmpeg" in backends:
    backend = "ffmpeg"
else:
    backend = "soundfile"


def wav_chunk_header(sample_rate=44100, bit_depth=16, channels=1):
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(bit_depth // 8)
        wav_file.setframerate(sample_rate)

    wav_header_bytes = buffer.getvalue()
    buffer.close()
    return wav_header_bytes


# Define utils for web server
async def http_execption_handler(exc: HTTPException):
    return JSONResponse(
        dict(
            statusCode=exc.status_code,
            message=exc.content,
            error=HTTPStatus(exc.status_code).phrase,
        ),
        exc.status_code,
        exc.headers,
    )


async def other_exception_handler(exc: "Exception"):
    traceback.print_exc()

    status = HTTPStatus.INTERNAL_SERVER_ERROR
    return JSONResponse(
        dict(statusCode=status, message=str(exc), error=status.phrase),
        status,
    )


def load_audio(reference_audio, sr):
    if len(reference_audio) > 255 or not Path(reference_audio).exists():
        audio_data = reference_audio
        reference_audio = io.BytesIO(audio_data)

    waveform, original_sr = torchaudio.load(reference_audio, backend=backend)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if original_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=sr)
        waveform = resampler(waveform)

    audio = waveform.squeeze().numpy()
    return audio


def encode_reference(*, decoder_model, reference_audio, enable_reference_audio):
    if enable_reference_audio and reference_audio is not None:
        # Load audios, and prepare basic info here
        reference_audio_content = load_audio(
            reference_audio, decoder_model.spec_transform.sample_rate
        )

        audios = torch.from_numpy(reference_audio_content).to(decoder_model.device)[
            None, None, :
        ]
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=decoder_model.device, dtype=torch.long
        )
        logger.info(
            f"Loaded audio with {audios.shape[2] / decoder_model.spec_transform.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        if isinstance(decoder_model, FireflyArchitecture):
            prompt_tokens = decoder_model.encode(audios, audio_lengths)[0][0]

        logger.info(f"Encoded prompt: {prompt_tokens.shape}")
    else:
        prompt_tokens = None
        logger.info("No reference audio provided")

    return prompt_tokens


def decode_vq_tokens(
    *,
    decoder_model,
    codes,
):
    feature_lengths = torch.tensor([codes.shape[1]], device=decoder_model.device)
    logger.info(f"VQ features: {codes.shape}")

    if isinstance(decoder_model, FireflyArchitecture):
        # VQGAN Inference
        return decoder_model.decode(
            indices=codes[None],
            feature_lengths=feature_lengths,
        )[0].squeeze()

    raise ValueError(f"Unknown model type: {type(decoder_model)}")


routes = MultimethodRoutes(base_class=HttpView)


def get_content_type(audio_format):
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "flac":
        return "audio/flac"
    elif audio_format == "mp3":
        return "audio/mpeg"
    else:
        return "application/octet-stream"


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def batch_encode(model, audios: list[bytes | torch.Tensor]):
    audios = [
        (
            torch.from_numpy(
                librosa.load(io.BytesIO(audio), sr=model.spec_transform.sample_rate)[0]
            )[None]
            if isinstance(audio, bytes)
            else audio
        )
        for audio in audios
    ]

    # if any(audio.shape[-1] > model.spec_transform.sample_rate * 120 for audio in audios):
    #     raise ValueError("Single audio length is too long (>120s)")

    max_length = max(audio.shape[-1] for audio in audios)
    print(f"Encode max length: {max_length / model.spec_transform.sample_rate:.2f}s")

    lengths = torch.tensor([audio.shape[-1] for audio in audios], device=model.device)
    max_length = lengths.max().item()
    padded = torch.stack(
        [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[-1]))
            for audio in audios
        ]
    ).to(model.device)

    features, feature_lengths = model.encode(padded, audio_lengths=lengths)
    features, feature_lengths = features.cpu(), feature_lengths.cpu()

    return [feature[..., :length] for feature, length in zip(features, feature_lengths)]


@cached(
    cache=LRUCache(maxsize=10000),
    key=lambda model, audios: (model.device, tuple(audios)),
)
def cached_vqgan_batch_encode(model, audios: list[bytes]):
    return batch_encode(model, audios)


@routes.http.post("/v1/vqgan/encode")
def api_vqgan_encode(payload: Annotated[ServeVQGANEncodeRequest, Body(exclusive=True)]):

    start_time = time.time()
    tokens = cached_vqgan_batch_encode(decoder_model, payload.audios)
    logger.info(f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms")

    return ormsgpack.packb(
        ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


@torch.no_grad()
@torch.autocast(device_type="cuda", dtype=torch.half)
def vqgan_decode(model, features):
    lengths = torch.tensor(
        [feature.shape[-1] for feature in features], device=model.device
    )
    max_length = lengths.max().item()
    padded = torch.stack(
        [
            torch.nn.functional.pad(feature, (0, max_length - feature.shape[-1]))
            for feature in features
        ]
    ).to(model.device)

    # If bs too large, we do micro batch decode
    audios, audio_lengths = [], []
    for i in range(0, padded.shape[0], 8):
        audio, audio_length = model.decode(
            padded[i : i + 8], feature_lengths=lengths[i : i + 8]
        )
        audios.append(audio)
        audio_lengths.append(audio_length)
    audios = torch.cat(audios, dim=0)
    audio_lengths = torch.cat(audio_lengths, dim=0)
    audios, audio_lengths = audios.cpu(), audio_lengths.cpu()

    return [audio[..., :length].numpy() for audio, length in zip(audios, audio_lengths)]


@routes.http.post("/v1/vqgan/decode")
def api_vqgan_decode(payload: Annotated[ServeVQGANDecodeRequest, Body(exclusive=True)]):
    tokens = [torch.tensor(token, dtype=torch.int) for token in payload.tokens]
    start_time = time.time()
    audios = vqgan_decode(decoder_model, tokens)
    logger.info(f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms")
    audios = [audio.astype(np.float16).tobytes() for audio in audios]
    return ormsgpack.packb(
        ServeVQGANDecodeResponse(audios=audios), option=ormsgpack.OPT_SERIALIZE_PYDANTIC
    )


@torch.no_grad()
def batch_asr(model, audios, sr, language="auto"):
    resampled_audios = []
    for audio in audios:
        audio = torchaudio.functional.resample(audio, sr, 16000)
        assert audio.ndim == 1
        resampled_audios.append(audio)

    with global_lock:
        res = model.generate(
            input=resampled_audios,
            batch_size=len(resampled_audios),
            language=language,
            use_itn=True,
        )

    results = []
    for r, audio in zip(res, audios):
        text = r["text"]
        text = re.sub(r"<\|.*?\|>", "", text)
        duration = len(audio) / sr * 1000
        huge_gap = False

        if "timestamp" in r and len(r["timestamp"]) > 2:
            for timestamp_a, timestamp_b in zip(
                r["timestamp"][:-1], r["timestamp"][1:]
            ):
                # If there is a gap of more than 5 seconds, we consider it as a huge gap
                if timestamp_b[0] - timestamp_a[1] > 5000:
                    huge_gap = True
                    break

            # Doesn't make sense to have a huge gap at the end
            if duration - r["timestamp"][-1][1] > 3000:
                huge_gap = True

        results.append(
            {
                "text": text,
                "duration": duration,
                "huge_gap": huge_gap,
            }
        )

    return results


@routes.http.post("/v1/asr")
def api_invoke_asr(payload: Annotated[ServeASRRequest, Body(exclusive=True)]):
    start_time = time.time()
    audios = [np.frombuffer(audio, dtype=np.float16) for audio in payload.audios]
    audios = [torch.from_numpy(audio).float() for audio in audios]

    if any(audios.shape[-1] >= 30 * payload.sample_rate for audios in audios):
        raise HTTPException(status_code=400, detail="Audio length is too long")

    transcriptions = batch_asr(
        asr_model, audios=audios, sr=payload.sample_rate, language=payload.language
    )
    logger.info(f"[EXEC] ASR time: {(time.time() - start_time) * 1000:.2f}ms")

    return ormsgpack.packb(
        ServeASRResponse(transcriptions=transcriptions),
        option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
    )


from fish_speech.conversation import Conversation, Message


def execute_request(
    input_queue: queue.Queue,
    tokenizer: FishTokenizer,
    config: BaseModelArgs,
    request: ServeRequest,
    device: str = "cuda:0",
):

    im_end_id = tokenizer.get_token_id(IM_END_TOKEN)
    messages = []
    for message in request.messages:
        messages.append(message.to_conversation_message())

    assert len(messages) >= 1, "At least one message is required"
    # assert messages[-1].role == "user", "The last message must be from the user"

    if messages[-1].role == "user":
        messages.append(
            Message(role="assistant", parts=[], add_im_end=False, modality="voice")
        )
    elif messages[-1].role == "raw":
        messages[-1].add_im_start = False
        messages[-1].add_im_end = False
        messages[-1].modality = "voice"
    else:
        assert (
            messages[-1].role == "assistant"
        ), "The last message must be from the assistant"
        messages[-1].add_im_end = False

    conv = Conversation(messages=messages)

    conv.visualize(tokenizer)
    prompt = conv.encode_for_inference(
        tokenizer=tokenizer, num_codebooks=config.num_codebooks
    ).to(device)

    if request.streaming:
        for i in range(request.num_samples):
            yield ServeStreamResponse(
                sample_id=i,
                delta=ServeStreamDelta(
                    role="assistant",
                ),
            )

    req = {
        "prompt": prompt,
        "max_new_tokens": request.max_new_tokens,
        "im_end_id": im_end_id,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": request.repetition_penalty,
        "num_samples": request.num_samples,
        "early_stop_threshold": request.early_stop_threshold,
    }

    start = time.time()
    response_queue = queue.Queue()
    input_queue.put(GenerateRequest(req, response_queue))

    # Decoding
    decode_buffer = [[] for _ in range(request.num_samples)]
    parts = [[] for _ in range(request.num_samples)]

    def send_reset_buffer(sample_id):
        nonlocal decode_buffer
        if len(decode_buffer[sample_id]) == 0:
            return

        decoded = tokenizer.decode(decode_buffer[sample_id])
        part = ServeTextPart(text=decoded)

        if request.streaming:
            yield ServeStreamResponse(delta=ServeStreamDelta(part=part))
        else:
            parts[sample_id].append(part)

        decode_buffer[sample_id] = []

    # Decode process
    finished = [False for _ in range(request.num_samples)]
    stats = {}
    idx = 0
    while True:
        response = response_queue.get()

        if response in ["stop", "error"]:
            break

        for sample_id, tokens in enumerate(response):
            if finished[sample_id]:
                continue

            if tokens[0] == im_end_id:
                finished[sample_id] = True
                if request.streaming:
                    yield from send_reset_buffer(sample_id)
                    yield ServeStreamResponse(
                        sample_id=sample_id,
                        finish_reason="stop",
                        stats=stats,
                    )
                continue

            is_semantic = (
                tokenizer.semantic_begin_id <= tokens[0] <= tokenizer.semantic_end_id
            )
            if is_semantic and request.streaming:
                yield from send_reset_buffer(sample_id)
                # Streaming vq
                _tokens = tokens[1:].clone()

                if config.share_codebook_embeddings is False:
                    for i in range(len(_tokens)):
                        _tokens[i] -= config.codebook_size * i

                yield ServeStreamResponse(
                    sample_id=sample_id,
                    delta=ServeStreamDelta(part=ServeVQPart(codes=_tokens.tolist())),
                )
                continue

            # Not streaming vq
            if is_semantic:
                yield from send_reset_buffer(sample_id)
                # None streaming vq
                if len(parts[sample_id]) == 0 or not isinstance(
                    parts[sample_id][-1], ServeVQPart
                ):
                    _tokens = tokens[1:].clone()

                    if config.share_codebook_embeddings is False:
                        for i in range(len(_tokens)):
                            _tokens[i] -= config.codebook_size * i

                    parts[sample_id].append(ServeVQPart(codes=_tokens.tolist()))
                else:
                    for codebook_id, value in enumerate(tokens[1:, :]):
                        val = value.item()
                        if config.share_codebook_embeddings is False:
                            val -= config.codebook_size * codebook_id

                        parts[sample_id][-1].codes[codebook_id].append(val)
                continue

            if not is_semantic:
                # Stream text decode is not supported now
                decode_buffer[sample_id].append(tokens[0, 0])

        if idx == 0:
            stats["time_to_first_token"] = (time.time() - start) * 1000

        idx += 1

    for sample_id in range(request.num_samples):
        yield from send_reset_buffer(sample_id)

    stats["total_time"] = (time.time() - start) * 1000
    stats["total_tokens"] = idx

    if request.streaming:
        for sample_id in range(request.num_samples):
            if finished[sample_id]:
                continue
            yield ServeStreamResponse(
                finish_reason=response, stats=stats, sample_id=sample_id
            )
        return

    yield ServeResponse(
        messages=[
            ServeMessage(role="assistant", parts=parts[i])
            for i in range(request.num_samples)
        ],
        finish_reason=response,
        stats=stats,
    )


@routes.http.post("/v1/chat")
def api_invoke_chat(
    req: Annotated[ServeRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """

    # This makes torch compile happy
    assert (
        req.num_samples == GLOBAL_NUM_SAMPLES
    ), f"num_samples must be {GLOBAL_NUM_SAMPLES}"

    content_type = request.headers.get("Content-Type", "application/json")
    json_mode = "application/json" in content_type

    async def wrapped_generator():
        generator = execute_request(llama_queue, tokenizer, config, req, args.device)

        for i in generator:
            if json_mode:
                body = i.model_dump_json().encode("utf-8")
                yield b"data: " + body + b"\n\n"
            else:
                body = ormsgpack.packb(i, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)
                yield struct.pack("I", len(body)) + body

    # Naive mode
    if req.streaming is False:
        result = next(execute_request(llama_queue, tokenizer, config, req, args.device))

        if json_mode:
            return JSONResponse(result.model_dump())
        else:
            return ormsgpack.packb(result, option=ormsgpack.OPT_SERIALIZE_PYDANTIC)

    return StreamResponse(
        iterable=wrapped_generator(), content_type="text/event-stream"
    )


@torch.inference_mode()
def inference(req: ServeTTSRequest):

    global prompt_tokens, prompt_texts

    idstr: str | None = req.reference_id
    if idstr is not None:
        ref_folder = Path("references") / idstr
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )

        if req.use_memory_cache == "never" or (
            req.use_memory_cache == "on-demand" and len(prompt_tokens) == 0
        ):
            prompt_tokens = [
                encode_reference(
                    decoder_model=decoder_model,
                    reference_audio=audio_to_bytes(str(ref_audio)),
                    enable_reference_audio=True,
                )
                for ref_audio in ref_audios
            ]
            prompt_texts = [
                read_ref_text(str(ref_audio.with_suffix(".lab")))
                for ref_audio in ref_audios
            ]
        else:
            logger.info("Use same references")

    else:
        # Parse reference audio aka prompt
        refs = req.references

        if req.use_memory_cache == "never" or (
            req.use_memory_cache == "on-demand" and len(prompt_tokens) == 0
        ):
            prompt_tokens = [
                encode_reference(
                    decoder_model=decoder_model,
                    reference_audio=ref.audio,
                    enable_reference_audio=True,
                )
                for ref in refs
            ]
            prompt_texts = [ref.text for ref in refs]
        else:
            logger.info("Use same references")

    if req.seed is not None:
        set_seed(req.seed)
        logger.warning(f"set seed: {req.seed}")

    # LLAMA Inference
    request = dict(
        device=decoder_model.device,
        max_new_tokens=req.max_new_tokens,
        text=(
            req.text
            if not req.normalize
            else ChnNormedText(raw_text=req.text).normalize()
        ),
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        compile=args.compile,
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        max_length=4096,
        prompt_tokens=prompt_tokens,
        prompt_text=prompt_texts,
    )

    response_queue = queue.Queue()
    llama_queue.put(
        GenerateRequest(
            request=request,
            response_queue=response_queue,
        )
    )

    if req.streaming:
        yield wav_chunk_header()

    segments = []
    while True:
        result: WrappedGenerateResponse = response_queue.get()
        if result.status == "error":
            raise result.response
            break

        result: GenerateResponse = result.response
        if result.action == "next":
            break

        with autocast_exclude_mps(
            device_type=decoder_model.device.type, dtype=args.precision
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
            )

        fake_audios = fake_audios.float().cpu().numpy()

        if req.streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes()
        else:
            segments.append(fake_audios)

    if req.streaming:
        return

    if len(segments) == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )

    fake_audios = np.concatenate(segments, axis=0)
    yield fake_audios


async def inference_async(req: ServeTTSRequest):
    for chunk in inference(req):
        yield chunk


async def buffer_to_async_generator(buffer):
    yield buffer


@routes.http.post("/v1/tts")
async def api_invoke_model(
    req: Annotated[ServeTTSRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """

    if args.max_text_length > 0 and len(req.text) > args.max_text_length:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content=f"Text is too long, max length is {args.max_text_length}",
        )

    if req.streaming and req.format != "wav":
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            content="Streaming only supports WAV format",
        )

    if req.streaming:
        return StreamResponse(
            iterable=inference_async(req),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )
    else:
        fake_audios = next(inference(req))
        buffer = io.BytesIO()
        sf.write(
            buffer,
            fake_audios,
            decoder_model.spec_transform.sample_rate,
            format=req.format,
        )

        return StreamResponse(
            iterable=buffer_to_async_generator(buffer.getvalue()),
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type=get_content_type(req.format),
        )


@routes.http.post("/v1/health")
async def api_health():
    """
    Health check
    """
    for num_samples in [GLOBAL_NUM_SAMPLES, 1]:
        test_request = ServeRequest(
            messages=[
                ServeMessage(
                    role="raw",
                    parts=[
                        ServeTextPart(text="Speak out the provided text."),
                    ],
                ),
            ],
            streaming=True,
            num_samples=num_samples,
        )
        for value in execute_request(
            llama_queue, tokenizer, config, test_request, args.device
        ):
            if (
                isinstance(value, (ServeStreamResponse, ServeResponse))
                and value.finish_reason == "error"
            ):
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    detail="Model is not loaded correctly",
                )
    return JSONResponse({"status": "ok"})


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["agent", "tts"], default="tts")
    parser.add_argument("--load-asr-model", action="store_true")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/fish-speech-1.4",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/fish-speech-1.4/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1)

    return parser.parse_args()


# Define Kui app
openapi = OpenAPI(
    {
        "title": "Fish Speech API",
        "version": "1.4.2",
    },
).routes


class MsgPackRequest(HttpRequest):
    async def data(
        self,
    ) -> Annotated[
        Any, ContentType("application/msgpack"), ContentType("application/json")
    ]:
        if self.content_type == "application/msgpack":
            return ormsgpack.unpackb(await self.body)

        elif self.content_type == "application/json":
            return await self.json

        raise HTTPException(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            headers={"Accept": "application/msgpack, application/json"},
        )


app = Kui(
    routes=routes + openapi[1:],  # Remove the default route
    exception_handlers={
        HTTPException: http_execption_handler,
        Exception: other_exception_handler,
    },
    factory_class=FactoryClass(http=MsgPackRequest),
    cors_config={},
)


def load_asr_model(*, device="cuda", hub="ms"):
    return AutoModel(
        model="iic/SenseVoiceSmall",
        device=device,
        disable_pbar=True,
        hub=hub,
    )


# Each worker process created by Uvicorn has its own memory space,
# meaning that models and variables are not shared between processes.
# Therefore, any global variables (like `llama_queue` or `decoder_model`)
# will not be shared across workers.


# Multi-threading for deep learning can cause issues, such as inconsistent
# outputs if multiple threads access the same buffers simultaneously.
# Instead, it's better to use multiprocessing or independent models per thread.
@app.on_startup
def initialize_app(app: Kui):

    global args, llama_queue, tokenizer, config, decoder_model, vad_model, asr_model, prompt_tokens, prompt_texts

    prompt_tokens, prompt_texts = [], []

    args = parse_args()  # args same as ones in other processes
    args.precision = torch.half if args.half else torch.bfloat16

    if args.load_asr_model:
        logger.info(f"Loading ASR model...")
        asr_model = load_asr_model(device=args.device)

    logger.info("Loading Llama model...")

    if args.mode == "tts":
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=args.llama_checkpoint_path,
            device=args.device,
            precision=args.precision,
            compile=args.compile,
        )
    else:
        llama_queue, tokenizer, config = launch_thread_safe_queue_agent(
            checkpoint_path=args.llama_checkpoint_path,
            device=args.device,
            precision=args.precision,
            compile=args.compile,
        )

    logger.info("Llama model loaded, loading VQ-GAN model...")

    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    logger.info("VQ-GAN model loaded, warming up...")

    vad_model = load_silero_vad()

    logger.info("VAD model loaded, warming up...")

    if args.mode == "tts":
        # Dry run to ensure models work and avoid first-time latency
        list(
            inference(
                ServeTTSRequest(
                    text="Hello world.",
                    references=[],
                    reference_id=None,
                    max_new_tokens=0,
                    chunk_length=200,
                    top_p=0.7,
                    repetition_penalty=1.5,
                    temperature=0.7,
                    emotion=None,
                    format="wav",
                )
            )
        )

    logger.info(f"Warming up done, starting server at http://{args.listen}")


if __name__ == "__main__":

    import uvicorn

    args = parse_args()
    host, port = args.listen.split(":")
    uvicorn.run(
        "tools.api:app",
        host=host,
        port=int(port),
        workers=args.workers,
        log_level="info",
    )
