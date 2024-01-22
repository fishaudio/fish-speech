import gc
import io
import time
import traceback
from http import HTTPStatus
from threading import Lock
from typing import Annotated, Literal, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from hydra import compose, initialize
from hydra.utils import instantiate
from kui.wsgi import (
    Body,
    HTTPException,
    HttpView,
    JSONResponse,
    Kui,
    OpenAPI,
    Path,
    StreamResponse,
    allow_cors,
)
from kui.wsgi.routing import MultimethodRoutes, Router
from loguru import logger
from pydantic import BaseModel
from transformers import AutoTokenizer

import tools.llama.generate
from fish_speech.models.vqgan.utils import sequence_mask
from tools.llama.generate import encode_tokens, generate, load_model


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

# Define models
MODELS = {}


class LlamaModel:
    def __init__(
        self,
        config_name: str,
        checkpoint_path: str,
        device,
        precision: str,
        tokenizer_path: str,
        compile: bool,
    ):
        self.device = device
        self.compile = compile

        self.t0 = time.time()
        self.precision = torch.bfloat16 if precision == "bfloat16" else torch.float16
        self.model = load_model(config_name, checkpoint_path, device, self.precision)
        self.model_size = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        torch.cuda.synchronize()
        logger.info(f"Time to load model: {time.time() - self.t0:.02f} seconds")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if self.compile:
            logger.info("Compiling model ...")
            tools.llama.generate.decode_one_token = torch.compile(
                tools.llama.generate.decode_one_token,
                mode="reduce-overhead",
                fullgraph=True,
            )

    def __del__(self):
        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("The llama is removed from memory.")


class VQGANModel:
    def __init__(self, config_name: str, checkpoint_path: str, device: str):
        with initialize(version_base="1.3", config_path="../fish_speech/configs"):
            self.cfg = compose(config_name=config_name)

        self.model = instantiate(self.cfg.model)
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.model.device,
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(device)

        logger.info("Restored VQGAN model from checkpoint")

    def __del__(self):
        self.cfg = None
        self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("The vqgan model is removed from memory.")

    @torch.no_grad()
    def sematic_to_wav(self, indices):
        model = self.model
        indices = indices.to(model.device).long()
        feature_lengths = torch.tensor([indices.shape[1]], device=model.device)
        decoded = model.decode(indices=indices[None], feature_lengths=feature_lengths)

        # Save audio
        fake_audio = decoded.audios[0, 0].cpu().numpy().astype(np.float32)

        return fake_audio, model.sampling_rate

    @torch.no_grad()
    def wav_to_semantic(self, audio):
        model = self.model
        # Load audio
        audio, _ = librosa.load(
            audio,
            sr=model.sampling_rate,
            mono=True,
        )
        audios = torch.from_numpy(audio).to(model.device)[None, None, :]
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sampling_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor(
            [audios.shape[2]], device=model.device, dtype=torch.long
        )
        encoded = model.encode(audios, audio_lengths)
        indices = encoded.indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")
        return indices


class LoadLlamaModelRequest(BaseModel):
    config_name: str = "text2semantic_finetune"
    checkpoint_path: str = "checkpoints/text2semantic-400m-v0.2-4k.pth"
    precision: Literal["float16", "bfloat16"] = "bfloat16"
    tokenizer: str = "fishaudio/speech-lm-v1"
    compile: bool = True


class LoadVQGANModelRequest(BaseModel):
    config_name: str = "vqgan_pretrain"
    checkpoint_path: str = "checkpoints/vqgan-v1.pth"


class LoadModelRequest(BaseModel):
    device: str = "cuda"
    llama: LoadLlamaModelRequest
    vqgan: LoadVQGANModelRequest


class LoadModelResponse(BaseModel):
    name: str


@routes.http.put("/models/{name}")
def api_load_model(
    name: Annotated[str, Path("default")],
    req: Annotated[LoadModelRequest, Body(exclusive=True)],
) -> Annotated[LoadModelResponse, JSONResponse[200, {}, LoadModelResponse]]:
    """
    Load model
    """

    if name in MODELS:
        del MODELS[name]

    llama = req.llama
    vqgan = req.vqgan

    logger.info("Loading model ...")
    new_model = {
        "llama": LlamaModel(
            config_name=llama.config_name,
            checkpoint_path=llama.checkpoint_path,
            device=req.device,
            precision=llama.precision,
            tokenizer_path=llama.tokenizer,
            compile=llama.compile,
        ),
        "vqgan": VQGANModel(
            config_name=vqgan.config_name,
            checkpoint_path=vqgan.checkpoint_path,
            device=req.device,
        ),
        "lock": Lock(),
    }

    MODELS[name] = new_model

    return LoadModelResponse(name=name)


@routes.http.delete("/models/{name}")
def api_delete_model(
    name: Annotated[str, Path("default")],
) -> JSONResponse[200, {}, dict]:
    """
    Delete model
    """

    if name not in MODELS:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            content="Model not found.",
        )

    del MODELS[name]

    return JSONResponse(
        dict(message="Model deleted."),
        200,
    )


@routes.http.get("/models")
def api_list_models() -> JSONResponse[200, {}, dict]:
    """
    List models
    """

    return JSONResponse(
        dict(models=list(MODELS.keys())),
        200,
    )


class InvokeRequest(BaseModel):
    text: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    prompt_text: Optional[str] = None
    prompt_tokens: Optional[str] = None
    max_new_tokens: int = 0
    top_k: Optional[int] = None
    top_p: float = 0.5
    repetition_penalty: float = 1.5
    temperature: float = 0.7
    order: str = "zh,jp,en"
    use_g2p: bool = True
    seed: Optional[int] = None
    speaker: Optional[str] = None


@routes.http.post("/models/{name}/invoke")
def api_invoke_model(
    name: Annotated[str, Path("default")],
    req: Annotated[InvokeRequest, Body(exclusive=True)],
):
    """
    Invoke model and generate audio
    """

    if name not in MODELS:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            content="Cannot find model.",
        )

    model = MODELS[name]
    llama_model_manager = model["llama"]
    vqgan_model_manager = model["vqgan"]

    device = llama_model_manager.device
    seed = req.seed
    prompt_tokens = req.prompt_tokens
    logger.info(f"Device: {device}")

    if prompt_tokens is not None and prompt_tokens.endswith(".npy"):
        prompt_tokens = torch.from_numpy(np.load(prompt_tokens)).to(device)
    elif prompt_tokens is not None and prompt_tokens.endswith(".wav"):
        prompt_tokens = vqgan_model_manager.wav_to_semantic(prompt_tokens)
    elif prompt_tokens is not None:
        logger.error(f"Unknown prompt tokens: {prompt_tokens}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            content="Unknown prompt tokens, it should be either .npy or .wav file.",
        )
    else:
        prompt_tokens = None

    # Lock
    model["lock"].acquire()

    encoded = encode_tokens(
        llama_model_manager.tokenizer,
        req.text,
        prompt_text=req.prompt_text,
        prompt_tokens=prompt_tokens,
        bos=True,
        device=device,
        use_g2p=req.use_g2p,
        speaker=req.speaker,
        order=req.order,
    )
    prompt_length = encoded.size(1)
    logger.info(f"Encoded prompt shape: {encoded.shape}")

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    torch.cuda.synchronize()

    t0 = time.perf_counter()
    y = generate(
        model=llama_model_manager.model,
        prompt=encoded,
        max_new_tokens=req.max_new_tokens,
        eos_token_id=llama_model_manager.tokenizer.eos_token_id,
        precision=llama_model_manager.precision,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
    )

    torch.cuda.synchronize()
    t = time.perf_counter() - t0

    tokens_generated = y.size(1) - prompt_length
    tokens_sec = tokens_generated / t
    logger.info(
        f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
    )
    logger.info(
        f"Bandwidth achieved: {llama_model_manager.model_size * tokens_sec / 1e9:.02f} GB/s"
    )
    logger.info(f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    codes = y[1:, prompt_length:-1]
    codes = codes - 2
    assert (codes >= 0).all(), "Codes should be >= 0"

    # Release lock
    model["lock"].release()

    # --------------- llama end ------------
    audio, sr = vqgan_model_manager.sematic_to_wav(codes)
    # --------------- vqgan end ------------

    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="wav")

    return StreamResponse(
        iterable=[buffer.getvalue()],
        headers={
            "Content-Disposition": "attachment; filename=audio.wav",
            "Content-Type": "application/octet-stream",
        },
    )


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
