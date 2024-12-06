import io
import os
import re
import time
import traceback
from argparse import ArgumentParser
from http import HTTPStatus
from typing import Annotated, Any

import librosa
import numpy as np
import ormsgpack
import pyrootutils
import soundfile as sf
import torch
import torchaudio
import uvicorn
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

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from threading import Lock

import httpx
from cachetools import LRUCache, cached
from funasr import AutoModel
from silero_vad import load_silero_vad

from tools.inference_engine import TTSInferenceEngine
from tools.llama.generate import (
    launch_thread_safe_queue,
    launch_thread_safe_queue_agent,
)
from tools.schema import (
    GLOBAL_NUM_SAMPLES,
    ServeASRRequest,
    ServeASRResponse,
    ServeTTSRequest,
    ServeVQGANDecodeRequest,
    ServeVQGANDecodeResponse,
    ServeVQGANEncodeRequest,
    ServeVQGANEncodeResponse,
)
from tools.server.inference import inference_wrapper as inference
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

async def inference_async(req: ServeTTSRequest, engine: TTSInferenceEngine):
    for chunk in inference(req, engine):
        if isinstance(chunk, bytes):
            yield chunk

async def buffer_to_async_generator(buffer):
    yield buffer

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

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA is not available, running on CPU.")
        args.device = "cpu"

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
                    format="wav",
                )
            )
        )

    logger.info(f"Warming up done, starting server at http://{args.listen}")


if __name__ == "__main__":

    args = parse_args()
    host, port = args.listen.split(":")

    uvicorn.run(
        "tools.api_server:app",
        host=host,
        port=int(port),
        workers=args.workers,
        log_level="info",
    )


# Construction Area Ahead

class HealthView(HttpView):
    """
    Return the health status of the server.
    """
    async def post(self, request):
        return JSONResponse({"status": "ok"})


class VQGANEncodeView(HttpView):
    """
    Encode the audio into symbolic tokens.
    """
    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeVQGANEncodeRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Encode the audio
        start_time = time.time()
        tokens = cached_vqgan_batch_encode(decoder_model, req.audios)
        logger.info(f"[EXEC] VQGAN encode time: {(time.time() - start_time) * 1000:.2f}ms")

        # Return the response
        return ormsgpack.packb(
            ServeVQGANEncodeResponse(tokens=[i.tolist() for i in tokens]),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )


class VQGANDecodeView(HttpView):
    """
    Decode the symbolic tokens into audio.
    """
    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeVQGANDecodeRequest(**payload)
        
        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        decoder_model = model_manager.decoder_model

        # Decode the audio
        tokens = [torch.tensor(token, dtype=torch.int) for token in req.tokens]
        start_time = time.time()
        audios = vqgan_decode(decoder_model, tokens)
        logger.info(f"[EXEC] VQGAN decode time: {(time.time() - start_time) * 1000:.2f}ms")
        audios = [audio.astype(np.float16).tobytes() for audio in audios]

        # Return the response
        return ormsgpack.packb(
            ServeVQGANDecodeResponse(audios=audios),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC
        )


class ASRView(HttpView):
    """
    Perform automatic speech recognition on the audio.
    """
    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeASRRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        asr_model = model_manager.asr_model

        # Perform ASR
        start_time = time.time()
        audios = [np.frombuffer(audio, dtype=np.float16) for audio in req.audios]
        audios = [torch.from_numpy(audio).float() for audio in audios]

        if any(audios.shape[-1] >= 30 * req.sample_rate for audios in audios):
            raise HTTPException(status_code=400, content="Audio length is too long")

        transcriptions = batch_asr(
            asr_model, audios=audios, sr=req.sample_rate, language=req.language
        )
        logger.info(f"[EXEC] ASR time: {(time.time() - start_time) * 1000:.2f}ms")

        # Return the response
        return ormsgpack.packb(
            ServeASRResponse(transcriptions=transcriptions),
            option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
        )


class TTSView(HttpView):
    """
    Perform text-to-speech on the input text.
    """
    async def post(self, request):
        # Decode the request
        payload = await request.data()
        req = ServeTTSRequest(**payload)

        # Get the model from the app
        model_manager: ModelManager = request.app.state.model_manager
        engine = model_manager.tts_inference_engine

        # Check if the text is too long
        if args.max_text_length > 0 and len(req.text) > args.max_text_length:
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content=f"Text is too long, max length is {args.max_text_length}",
            )

        # Check if streaming is enabled
        if req.streaming and req.format != "wav":
            raise HTTPException(
                HTTPStatus.BAD_REQUEST,
                content="Streaming only supports WAV format",
            )

        # Perform TTS
        if req.streaming:
            return StreamResponse(
                iterable=inference_async(req, engine),
                headers={
                    "Content-Disposition": f"attachment; filename=audio.{req.format}",
                },
                content_type=get_content_type(req.format),
            )
        else:
            fake_audios = next(inference(req, engine))
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

class API(ExceptionHandler)
    def __init__(self):
        self.args = parse_args()
        self.routes = [
            ("/v1/health", HealthView),
            ("/v1/vqgan/encode", VQGANEncodeView),
            ("/v1/vqgan/decode", VQGANDecodeView),
            ("/v1/asr", ASRView),
            ("/v1/tts", TTSView),
        ]

        self.openapi = OpenAPI(
            {
                "title": "Fish Speech API",
                "version": "1.5.0",
            },
        ).routes

        # Initialize the app
        self.app = Kui(
            routes=self.routes + self.openapi[1:],  # Remove the default route
            exception_handlers={
                HTTPException: self.http_execption_handler,
                Exception: self.other_exception_handler,
            },
            factory_class=FactoryClass(http=MsgPackRequest),
            cors_config={},
        )

        # Associate the app with the model manager
        self.app.on_startup(self.initialize_app)

    async def initialize_app(self, app: Kui):
        # Make the ModelManager available to the views
        app.state.model_manager = ModelManager(
            device=self.args.device,
            half=self.args.half,
            compile=self.args.compile,
            asr_enabled=self.args.load_asr_model,
            llama_checkpoint_path=self.args.llama_checkpoint_path,
            decoder_checkpoint_path=self.args.decoder_checkpoint_path,
            decoder_config_name=self.args.decoder_config_name,
        )

        logger.info(f"Warming up done, starting server at http://{self.args.listen}")


if __name__ == "__main__":
    api = API()
    host, port = api.args.listen.split(":")

    uvicorn.run(
        api.app,
        host=host,
        port=int(port),
        workers=api.args.workers,
        log_level="info",
    )
