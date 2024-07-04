import base64
import io
import json
import queue
import random
import traceback
import wave
from argparse import ArgumentParser
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Literal, Optional

import librosa
import numpy as np
import pyrootutils
import soundfile as sf
import torch
from kui.asgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Kui,
    OpenAPI,
    StreamResponse,
)
from kui.asgi.routing import MultimethodRoutes
from loguru import logger
from pydantic import BaseModel, Field

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# from fish_speech.models.vqgan.lit_module import VQGAN
from fish_speech.models.vqgan.modules.firefly import FireflyArchitecture
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


def encode_reference(*, decoder_model, reference_audio, enable_reference_audio):
    if enable_reference_audio and reference_audio is not None:
        # Load audios, and prepare basic info here
        reference_audio_content, _ = librosa.load(
            reference_audio, sr=decoder_model.spec_transform.sample_rate, mono=True
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
        ).squeeze()

    raise ValueError(f"Unknown model type: {type(decoder_model)}")


routes = MultimethodRoutes(base_class=HttpView)


def get_random_paths(base_path, data, speaker, emotion):
    if base_path and data and speaker and emotion and (Path(base_path).exists()):
        if speaker in data and emotion in data[speaker]:
            files = data[speaker][emotion]
            lab_files = [f for f in files if f.endswith(".lab")]
            wav_files = [f for f in files if f.endswith(".wav")]

            if lab_files and wav_files:
                selected_lab = random.choice(lab_files)
                selected_wav = random.choice(wav_files)

                lab_path = Path(base_path) / speaker / emotion / selected_lab
                wav_path = Path(base_path) / speaker / emotion / selected_wav
                if lab_path.exists() and wav_path.exists():
                    return lab_path, wav_path

    return None, None


def load_json(json_file):
    if not json_file:
        logger.info("Not using a json file")
        return None
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        logger.warning(f"ref json not found: {json_file}")
        data = None
    except Exception as e:
        logger.warning(f"Loading json failed: {e}")
        data = None
    return data


class InvokeRequest(BaseModel):
    text: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    reference_text: Optional[str] = None
    reference_audio: Optional[str] = None
    max_new_tokens: int = 1024
    chunk_length: Annotated[int, Field(ge=0, le=500, strict=True)] = 100
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    emotion: Optional[str] = None
    format: Literal["wav", "mp3", "flac"] = "wav"
    streaming: bool = False
    ref_json: Optional[str] = "ref_data.json"
    ref_base: Optional[str] = "ref_data"
    speaker: Optional[str] = None


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
def inference(req: InvokeRequest):
    # Parse reference audio aka prompt
    prompt_tokens = None

    ref_data = load_json(req.ref_json)
    ref_base = req.ref_base

    lab_path, wav_path = get_random_paths(ref_base, ref_data, req.speaker, req.emotion)

    if lab_path and wav_path:
        with open(lab_path, "r", encoding="utf-8") as lab_file:
            ref_text = lab_file.read()
        req.reference_audio = wav_path
        req.reference_text = ref_text
        logger.info("ref_path: " + str(wav_path))
        logger.info("ref_text: " + ref_text)

    # Parse reference audio aka prompt
    prompt_tokens = encode_reference(
        decoder_model=decoder_model,
        reference_audio=req.reference_audio,
        enable_reference_audio=req.reference_audio is not None,
    )

    # LLAMA Inference
    request = dict(
        device=decoder_model.device,
        max_new_tokens=req.max_new_tokens,
        text=req.text,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        compile=args.compile,
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        max_length=2048,
        prompt_tokens=prompt_tokens,
        prompt_text=req.reference_text,
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

        with torch.autocast(
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


async def inference_async(req: InvokeRequest):
    for chunk in inference(req):
        yield chunk


async def buffer_to_async_generator(buffer):
    yield buffer


@routes.http.post("/v1/invoke")
async def api_invoke_model(
    req: Annotated[InvokeRequest, Body(exclusive=True)],
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
        default="checkpoints/fish-speech-1.2",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/fish-speech-1.2/firefly-gan-vq-fsq-4x1024-42hz-generator.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="firefly_gan_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=1)

    return parser.parse_args()


# Define Kui app
openapi = OpenAPI(
    {
        "title": "Fish Speech API",
    },
).routes

app = Kui(
    routes=routes + openapi[1:],  # Remove the default route
    exception_handlers={
        HTTPException: http_execption_handler,
        Exception: other_exception_handler,
    },
    cors_config={},
)


if __name__ == "__main__":
    import threading

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
            InvokeRequest(
                text="Hello world.",
                reference_text=None,
                reference_audio=None,
                max_new_tokens=0,
                top_p=0.7,
                repetition_penalty=1.2,
                temperature=0.7,
                emotion=None,
                format="wav",
                ref_base=None,
                ref_json=None,
            )
        )
    )

    logger.info(f"Warming up done, starting server at http://{args.listen}")
    host, port = args.listen.split(":")
    uvicorn.run(app, host=host, port=int(port), workers=args.workers, log_level="info")
