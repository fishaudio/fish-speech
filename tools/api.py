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

from fish_speech.models.vits_decoder.lit_module import VITSDecoder
from fish_speech.models.vqgan.lit_module import VQGAN
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


def encode_reference(*, decoder_model, reference_audio, enable_reference_audio):
    if enable_reference_audio and reference_audio is not None:
        # Load audios, and prepare basic info here
        reference_audio_content, _ = librosa.load(
            reference_audio, sr=decoder_model.sampling_rate, mono=True
        )
        audios = torch.from_numpy(reference_audio_content).to(decoder_model.device)[
            None, None, :
        ]
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=decoder_model.device, dtype=torch.long
        )
        logger.info(
            f"Loaded audio with {audios.shape[2] / decoder_model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        if isinstance(decoder_model, VQGAN):
            prompt_tokens = decoder_model.encode(audios, audio_lengths)[0][0]
            reference_embedding = None  # VQGAN does not have reference embedding
        elif isinstance(decoder_model, VITSDecoder):
            reference_spec = decoder_model.spec_transform(audios[0])
            reference_embedding = decoder_model.generator.encode_ref(
                reference_spec,
                torch.tensor([reference_spec.shape[-1]], device=decoder_model.device),
            )
            logger.info(f"Loaded reference audio from {reference_audio}")
            prompt_tokens = decoder_model.generator.vq.encode(audios, audio_lengths)[0][
                0
            ]
        else:
            raise ValueError(f"Unknown model type: {type(decoder_model)}")

        logger.info(f"Encoded prompt: {prompt_tokens.shape}")
    elif isinstance(decoder_model, VITSDecoder):
        prompt_tokens = None
        reference_embedding = torch.zeros(
            1, decoder_model.generator.gin_channels, 1, device=decoder_model.device
        )
        logger.info("No reference audio provided, use zero embedding")
    else:
        prompt_tokens = None
        reference_embedding = None
        logger.info("No reference audio provided")

    return prompt_tokens, reference_embedding


def decode_vq_tokens(
    *,
    decoder_model,
    codes,
    text_tokens: torch.Tensor | None = None,
    reference_embedding: torch.Tensor | None = None,
):
    feature_lengths = torch.tensor([codes.shape[1]], device=decoder_model.device)
    logger.info(f"VQ features: {codes.shape}")

    if isinstance(decoder_model, VQGAN):
        # VQGAN Inference
        return decoder_model.decode(
            indices=codes[None],
            feature_lengths=feature_lengths,
            return_audios=True,
        ).squeeze()

    if isinstance(decoder_model, VITSDecoder):
        # VITS Inference
        quantized = decoder_model.generator.vq.indicies_to_vq_features(
            indices=codes[None], feature_lengths=feature_lengths
        )
        logger.info(f"Restored VQ features: {quantized.shape}")

        return decoder_model.generator.decode(
            quantized,
            torch.tensor([quantized.shape[-1]], device=decoder_model.device),
            text_tokens,
            torch.tensor([text_tokens.shape[-1]], device=decoder_model.device),
            ge=reference_embedding,
        ).squeeze()

    raise ValueError(f"Unknown model type: {type(decoder_model)}")


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

    # Parse reference audio aka prompt
    prompt_tokens, reference_embedding = encode_reference(
        decoder_model=decoder_model,
        reference_audio=(
            io.BytesIO(base64.b64decode(req.reference_audio))
            if req.reference_audio is not None
            else None
        ),
        enable_reference_audio=req.reference_audio is not None,
    )

    # LLAMA Inference
    request = dict(
        tokenizer=llama_tokenizer,
        device=decoder_model.device,
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

        text_tokens = llama_tokenizer.encode(result.text, return_tensors="pt").to(
            decoder_model.device
        )

        with torch.autocast(
            device_type=decoder_model.device.type, dtype=args.precision
        ):
            fake_audios = decode_vq_tokens(
                decoder_model=decoder_model,
                codes=result.codes,
                text_tokens=text_tokens,
                reference_embedding=reference_embedding,
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
        sf.write(buffer, fake_audios, decoder_model.sampling_rate, format=req.format)

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
        "--llama-config-name", type=str, default="dual_ar_2_codebook_medium"
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/vq-gan-group-fsq-2x1024.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="vqgan_pretrain")
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
