import base64
import io
import traceback
from argparse import ArgumentParser
from http import HTTPStatus
from threading import Lock
from typing import Annotated, Literal, Optional

import librosa
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
    allow_cors,
)
from kui.wsgi.routing import MultimethodRoutes
from loguru import logger
from pydantic import BaseModel
from transformers import AutoTokenizer

from tools.llama.generate import generate_long
from tools.llama.generate import load_model as load_llama_model
from tools.vqgan.inference import load_model as load_vqgan_model
from tools.webui import inference

lock = Lock()


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
    chunk_length: int = 30
    top_k: int = 0
    top_p: float = 0.7
    repetition_penalty: float = 1.5
    temperature: float = 0.7
    speaker: Optional[str] = None
    format: Literal["wav", "mp3", "flac"] = "wav"


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
    result = generate_long(
        model=llama_model,
        tokenizer=llama_tokenizer,
        device=vqgan_model.device,
        decode_one_token=decode_one_token,
        max_new_tokens=req.max_new_tokens,
        text=req.text,
        top_k=int(req.top_k) if req.top_k > 0 else None,
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
    )

    codes = next(result)

    # VQGAN Inference
    feature_lengths = torch.tensor([codes.shape[1]], device=vqgan_model.device)
    fake_audios = vqgan_model.decode(
        indices=codes[None], feature_lengths=feature_lengths, return_audios=True
    )[0, 0]

    fake_audios = fake_audios.float().cpu().numpy()

    return fake_audios


@routes.http.post("/invoke")
def api_invoke_model(
    req: Annotated[InvokeRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """

    if args.max_gradio_length > 0 and len(req.text) > args.max_gradio_length:
        raise HTTPException(
            HTTPStatus.BAD_REQUEST,
            f"Text is too long, max length is {args.max_gradio_length}",
        )

    try:
        # Lock, avoid interrupting the inference process
        lock.acquire()
        fake_audios = inference(req)
    except Exception as e:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
    finally:
        # Release lock
        lock.release()

    buffer = io.BytesIO()
    sf.write(buffer, fake_audios, vqgan_model.sampling_rate, format=req.format)

    return StreamResponse(
        iterable=[buffer.getvalue()],
        headers={
            "Content-Disposition": f"attachment; filename=audio.{req.format}",
            "Content-Type": "application/octet-stream",
        },
    )


@routes.http.post("/health")
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
        default="checkpoints/text2semantic-medium-v1-2k.pth",
    )
    parser.add_argument(
        "--llama-config-name", type=str, default="dual_ar_2_codebook_medium"
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
    parser.add_argument("--max-gradio-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8000")

    return parser.parse_args()


# Define Kui app
app = Kui(
    exception_handlers={
        HTTPException: http_execption_handler,
        Exception: other_exception_handler,
    },
    cors_config={},
)

# Swagger UI & routes
app.router << ("/v1" // routes) << ("/docs" // OpenAPI().routes)


if __name__ == "__main__":
    import threading

    from zibai import create_bind_socket, serve

    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    logger.info("Loading Llama model...")
    llama_model, decode_one_token = load_llama_model(
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
    inference(
        InvokeRequest(
            text="A warm-up sentence.",
            reference_text=None,
            reference_audio=None,
            max_new_tokens=0,
            chunk_length=30,
            top_k=0,
            top_p=0.7,
            repetition_penalty=1.5,
            temperature=0.7,
            speaker=None,
            format="wav",
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
