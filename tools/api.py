import base64
import io
import queue
import threading
import traceback
import wave
from argparse import ArgumentParser
from http import HTTPStatus
from typing import Annotated, Literal, Optional

import librosa
import numpy as np
import pyrootutils
import soundfile as sf
import torch
from kui.wsgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Kui,
    OpenAPI,
    StreamResponse,
)
from kui.wsgi.routing import MultimethodRoutes
from loguru import logger
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from tools.llama.generate import launch_thread_safe_queue
from tools.vqgan.inference import load_model as load_vqgan_model
from tools.webui import inference


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
def http_execption_handler(exc: HTTPException):
    return JSONResponse(
        dict(
            statusCode=exc.status_code,
            message=exc.content,
            error=HTTPStatus(exc.status_code).phrase,
        ),
        exc.status_code,
        exc.headers,
    )


def other_exception_handler(exc: "Exception"):
    traceback.print_exc()

    status = HTTPStatus.INTERNAL_SERVER_ERROR
    return JSONResponse(
        dict(statusCode=status, message=str(exc), error=status.phrase),
        status,
    )


routes = MultimethodRoutes(base_class=HttpView)


class InvokeRequest(BaseModel):
    text: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    reference_text: Optional[str] = None
    reference_audio: Optional[str] = None
    max_new_tokens: int = 0
    chunk_length: Annotated[int, Field(ge=0, le=200, strict=True)] = 30
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.5
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    speaker: Optional[str] = None
    format: Literal["wav", "mp3", "flac"] = "wav"
    streaming: bool = False


@torch.inference_mode()
def inference(req: InvokeRequest):
    # Parse reference audio aka prompt
    prompt_tokens = None
    if req.reference_audio is not None:
        buffer = io.BytesIO(base64.b64decode(req.reference_audio))
        reference_audio_content, _ = librosa.load(
            buffer, sr=vqgan_model.sampling_rate, mono=True
        )
        audios = torch.from_numpy(reference_audio_content).to(vqgan_model.device)[
            None, None, :
        ]

        logger.info(
            f"Loaded audio with {audios.shape[2] / vqgan_model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=vqgan_model.device, dtype=torch.long
        )
        prompt_tokens = vqgan_model.encode(audios, audio_lengths)[0][0]

    # LLAMA Inference
    request = dict(
        tokenizer=llama_tokenizer,
        device=vqgan_model.device,
        max_new_tokens=req.max_new_tokens,
        text=req.text,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        temperature=req.temperature,
        compile=args.compile,
        iterative_prompt=req.chunk_length > 0,
        chunk_length=req.chunk_length,
        max_length=args.max_length,
        speaker=req.speaker,
        prompt_tokens=prompt_tokens,
        prompt_text=req.reference_text,
        is_streaming=True,
    )

    payload = dict(
        response_queue=queue.Queue(),
        request=request,
    )
    llama_queue.put(payload)

    if req.streaming:
        yield wav_chunk_header()

    segments = []
    while True:
        result = payload["response_queue"].get()
        if result == "next":
            # TODO: handle next sentence
            continue

        if result == "done":
            if payload["success"] is False:
                raise payload["response"]
            break

        # VQGAN Inference
        feature_lengths = torch.tensor([result.shape[1]], device=vqgan_model.device)
        fake_audios = vqgan_model.decode(
            indices=result[None], feature_lengths=feature_lengths, return_audios=True
        )[0, 0]
        fake_audios = fake_audios.float().cpu().numpy()
        fake_audios = np.concatenate([fake_audios, np.zeros((11025,))], axis=0)

        if req.streaming:
            yield (fake_audios * 32768).astype(np.int16).tobytes()
        else:
            segments.append(fake_audios)

    if req.streaming is False:
        fake_audios = np.concatenate(segments, axis=0)
        yield fake_audios


@routes.http.post("/v1/invoke")
def api_invoke_model(
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

    generator = inference(req)
    if req.streaming:
        return StreamResponse(
            iterable=generator,
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type="application/octet-stream",
        )
    else:
        fake_audios = next(generator)
        buffer = io.BytesIO()
        sf.write(buffer, fake_audios, vqgan_model.sampling_rate, format=req.format)

        return StreamResponse(
            iterable=[buffer.getvalue()],
            headers={
                "Content-Disposition": f"attachment; filename=audio.{req.format}",
            },
            content_type="application/octet-stream",
        )


@routes.http.post("/v1/health")
def api_health():
    """
    Health check
    """

    return JSONResponse({"status": "ok"})


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/text2semantic-sft-medium-v1-4k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_large"
    )
    parser.add_argument(
        "--vqgan-checkpoint-path",
        type=str,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--vqgan-config-name", type=str, default="vqgan_pretrain")
    parser.add_argument("--tokenizer", type=str, default="fishaudio/fish-speech-1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8000")

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

    from zibai import create_bind_socket, serve

    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        config_name=args.llama_config_name,
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        max_length=args.max_length,
        compile=args.compile,
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info("Llama model loaded, loading VQ-GAN model...")

    vqgan_model = load_vqgan_model(
        config_name=args.vqgan_config_name,
        checkpoint_path=args.vqgan_checkpoint_path,
        device=args.device,
    )

    logger.info("VQ-GAN model loaded, warming up...")

    # Dry run to check if the model is loaded correctly and avoid the first-time latency
    list(
        inference(
            InvokeRequest(
                text="A warm-up sentence.",
                reference_text=None,
                reference_audio=None,
                max_new_tokens=0,
                chunk_length=30,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                speaker=None,
                format="wav",
            )
        )
    )

    logger.info(f"Warming up done, starting server at http://{args.listen}")
    sock = create_bind_socket(args.listen)
    sock.listen()

    # Start server
    serve(
        app=app,
        bind_sockets=[sock],
        max_workers=10,
        graceful_exit=threading.Event(),
    )
