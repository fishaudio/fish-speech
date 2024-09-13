import base64
import io
import json
import queue
import random
import sys
import traceback
import wave
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

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
)
from kui.asgi.routing import MultimethodRoutes
from loguru import logger
from pydantic import BaseModel, Field, conint

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from fish_speech.models.vqgan.lit_module import VQGAN
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
from fish_speech.utils import autocast_exclude_mps
from tools.commons import ServeReferenceAudio, ServeTTSRequest
from tools.file import AUDIO_EXTENSIONS, audio_to_bytes, list_files, read_ref_text
from tools.llama.generate import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
    launch_thread_safe_queue,
)
from tools.vqgan.inference import load_model as load_decoder_model


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

    waveform, original_sr = torchaudio.load(
        reference_audio, backend="ffmpeg" if sys.platform == "linux" else "soundfile"
    )

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


@torch.inference_mode()
def inference(req: ServeTTSRequest):

    idstr: str | None = req.reference_id
    if idstr is not None:
        ref_folder = Path("references") / idstr
        ref_folder.mkdir(parents=True, exist_ok=True)
        ref_audios = list_files(
            ref_folder, AUDIO_EXTENSIONS, recursive=True, sort=False
        )
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
        # Parse reference audio aka prompt
        refs = req.references
        if refs is None:
            refs = []
        prompt_tokens = [
            encode_reference(
                decoder_model=decoder_model,
                reference_audio=ref.audio,
                enable_reference_audio=True,
            )
            for ref in refs
        ]
        prompt_texts = [ref.text for ref in refs]

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
        max_length=2048,
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

    return JSONResponse({"status": "ok"})


def parse_args():
    parser = ArgumentParser()
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
    },
).routes


class MsgPackRequest(HttpRequest):
    async def data(self) -> Annotated[Any, ContentType("application/msgpack")]:
        if self.content_type == "application/msgpack":
            return ormsgpack.unpackb(await self.body)

        raise HTTPException(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            headers={"Accept": "application/msgpack"},
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


if __name__ == "__main__":

    import uvicorn

    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
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

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                emotion=None,
                format="wav",
            )
        )
    )

    logger.info(f"Warming up done, starting server at http://{args.listen}")
    host, port = args.listen.split(":")
    uvicorn.run(app, host=host, port=int(port), workers=args.workers, log_level="info")
