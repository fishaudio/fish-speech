import gc
import time
from typing import Optional
from urllib.parse import unquote

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request, HTTPException
from hydra import compose, initialize
from hydra.utils import instantiate
from pydantic import BaseModel
from transformers import AutoTokenizer

import tools.llama.generate
from fish_speech.models.vqgan.utils import sequence_mask
from tools.llama.generate import load_model, encode_tokens, generate
from tools.log import logger


class LLamaModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_size = None
        self.t0 = None
        self.device = None
        self.precision = None
        self.compilation = None
        self.decode_one_token = None

    def load_model(self, config_name: str, checkpoint_path: str,
                   device, precision: bool, tokenizer_path: str, compilation: bool):
        self.device = device

        self.compilation = compilation
        if self.model is None:
            self.t0 = time.time()
            self.precision = torch.bfloat16 if precision else torch.half
            self.model = load_model(config_name, checkpoint_path, device, self.precision)
            self.model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            torch.cuda.synchronize()
            logger.info(f"Time to load model: {time.time() - self.t0:.02f} seconds")
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            if self.compilation:
                tools.llama.generate.decode_one_token = torch.compile(
                    tools.llama.generate.decode_one_token, mode="reduce-overhead", fullgraph=True
                )


    def get_model(self):
        if self.model is None:
            raise ValueError("Model is not loaded.")
        return self.model

    def get_model_size(self):
        if self.model_size is None:
            raise ValueError("Model is not loaded. Thus no size info.")
        return self.model_size

    def get_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self.tokenizer

    def get_device(self):
        return "cuda" if self.device is None else self.device

    def get_precision(self):
        return torch.half if self.precision is None else self.precision

    def del_model(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "The llama model is deleted!"


class VQGANModelManager:
    def __init__(self):
        self.model = None
        self.cfg = None

    def load_model(self, config_name: str, checkpoint_path: str):
        if self.cfg is None:
            with initialize(version_base="1.3", config_path="./fish_speech/configs"):
                self.cfg = compose(config_name=config_name)
        if self.model is None:
            self.model = instantiate(self.cfg.model)
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.model.device,
            )
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self.model.load_state_dict(state_dict, strict=True)
            self.model.eval()
            self.model.cuda()
        logger.info("Restored model from checkpoint")
        return self.model

    def del_model(self):
        self.cfg = None
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "The vqgan model is deleted!"

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=True)
    def sematic_to_wav(self, input_path, output_path):
        model = self.model
        if input_path.endswith(".wav"):
            logger.info(f"Processing in-place reconstruction of {input_path}")
            # Load audio
            audio, _ = librosa.load(
                input_path,
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

            features = gt_mels = model.mel_transform(
                audios, sample_rate=model.sampling_rate
            )

            if model.downsample is not None:
                features = model.downsample(features)

            mel_lengths = audio_lengths // model.hop_length
            feature_lengths = (
                    audio_lengths
                    / model.hop_length
                    / (model.downsample.total_strides if model.downsample is not None else 1)
            ).long()

            feature_masks = torch.unsqueeze(
                sequence_mask(feature_lengths, features.shape[2]), 1
            ).to(gt_mels.dtype)
            mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
                gt_mels.dtype
            )

            # vq_features is 50 hz, need to convert to true mel size
            text_features = model.mel_encoder(features, feature_masks)
            _, indices, _ = model.vq_encoder(text_features, feature_masks)

            if indices.ndim == 4 and indices.shape[1] == 1 and indices.shape[3] == 1:
                indices = indices[:, 0, :, 0]
            else:
                logger.error(f"Unknown indices shape: {indices.shape}")
                return

            logger.info(f"Generated indices of shape {indices.shape}")

            # Save indices
            np.save(output_path.replace(".wav", ".npy"), indices.cpu().numpy())
        elif input_path.endswith(".npy"):
            logger.info(f"Processing precomputed indices from {input_path}")
            indices = np.load(input_path)
            indices = torch.from_numpy(indices).to(model.device).long()
            assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        else:
            raise ValueError(f"Unknown input type: {input_path}")

            # Restore
        indices = indices.unsqueeze(1).unsqueeze(-1)
        mel_lengths = indices.shape[2] * (
            model.downsample.total_strides if model.downsample is not None else 1
        )
        mel_lengths = torch.tensor([mel_lengths], device=model.device, dtype=torch.long)
        mel_masks = torch.ones(
            (1, 1, mel_lengths), device=model.device, dtype=torch.float32
        )

        text_features = model.vq_encoder.decode(indices)

        logger.info(
            f"VQ Encoded, indices: {indices.shape} equivalent to "
            + f"{1 / (mel_lengths[0] * model.hop_length / model.sampling_rate / indices.shape[2]):.2f} Hz"
        )

        text_features = F.interpolate(text_features, size=mel_lengths[0], mode="nearest")

        # Sample mels
        decoded_mels = model.decoder(text_features, mel_masks)
        fake_audios = model.generator(decoded_mels)
        logger.info(
            f"Generated audio of shape {fake_audios.shape}, equivalent to {fake_audios.shape[-1] / model.sampling_rate:.2f} seconds"
        )

        # Save audio
        fake_audio = fake_audios[0, 0].cpu().numpy().astype(np.float32)
        sf.write(f"{output_path}", fake_audio, model.sampling_rate)
        logger.info(f"Saved audio to {output_path}")
        return f"Saved audio to {output_path}"


app = FastAPI()
app.logger = logger

llama_model_manager = LLamaModelManager()
vqgan_model_manager = VQGANModelManager()
logger.info("FastAPI, 启动!")


@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI application!"}


class LoadLLamaModelRequest(BaseModel):
    config_name: str
    checkpoint_path: str
    device: str
    precision: bool
    tokenizer: str
    compilation: bool


@app.post("/load-llama-model")
async def load_llama_model_api(
        request: Request,
        req: LoadLLamaModelRequest,
):
    """用post请求"""
    logger.info(
        f"{request.client.host}:{request.client.port}/load-model  {unquote(str(request.query_params))} "
        f"config_name={req.config_name}"
    )
    logger.info("Loading model ...")
    try:

        llama_model_manager.load_model(req.config_name, req.checkpoint_path, req.device, req.precision, req.tokenizer,
                                       req.compilation)
        return {"message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LoadVQGANModelRequest(BaseModel):
    config_name: str
    checkpoint_path: str


@app.post("/load-vqgan-model")
async def load_vqgan_model_api(
        request: Request,
        req: LoadVQGANModelRequest
):
    """用post请求"""
    logger.info(
        f"{request.client.host}:{request.client.port}/load-model  {unquote(str(request.query_params))} "
        f"config_name={req.config_name}"
    )
    try:
        vqgan_model_manager.load_model(req.config_name, req.checkpoint_path)
        return {"message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UseModelRequest(BaseModel):
    text: str
    prompt_text: Optional[str] = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游."
    prompt_tokens: Optional[str] = None
    num_samples: int = 1
    max_new_tokens: int = 0
    top_k: Optional[int] = None
    top_p: float = 0.5
    repetition_penalty: float = 1.5
    temperature: float = 0.7
    use_g2p: bool = True
    seed: int = 42
    speaker: Optional[str] = None


@app.post("/use-model")
async def use_model(
        request: Request,
        req: UseModelRequest
):
    # 这里添加使用模型处理文本的逻辑

    """用post请求"""
    logger.info(
        f"{request.client.host}:{request.client.port}/use-model  {unquote(str(request.query_params))} "
    )
    device = llama_model_manager.get_device()
    model = llama_model_manager.get_model()
    tokenizer = llama_model_manager.get_tokenizer()
    precision = llama_model_manager.get_precision()
    model_size = llama_model_manager.get_model_size()
    text = req.text
    prompt_text = req.prompt_text
    prompt_tokens = req.prompt_tokens
    num_samples = req.num_samples
    max_new_tokens = req.max_new_tokens
    top_k = req.top_k
    top_p = req.top_p
    repetition_penalty = req.repetition_penalty
    temperature = req.temperature
    use_g2p = req.use_g2p
    seed = req.seed
    speaker = req.speaker

    prompt_tokens = (
        torch.from_numpy(np.load(prompt_tokens)).to(device)
        if prompt_tokens is not None
        else None
    )
    encoded = encode_tokens(
        tokenizer,
        text,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
        bos=True,
        device=device,
        use_g2p=use_g2p,
        speaker=speaker,
    )
    prompt_length = encoded.size(1)
    logger.info(f"Encoded prompt shape: {encoded.shape}")
    torch.manual_seed(seed)

    for i in range(num_samples):
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        y = generate(
            model=model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            precision=precision,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if i == 0 and compile:
            logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / t
        logger.info(
            f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
        logger.info(
            f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
        )

        codes = y[1:, prompt_length:-1]
        codes = codes - 2
        assert (codes >= 0).all(), "Codes should be >= 0"

        np.save(f"codes_{i}.npy", codes.cpu().numpy())
        logger.info(f"Saved codes to codes_{i}.npy")
        # --------------- llama end ------------
        vqgan_model_manager.sematic_to_wav(f"codes_{i}.npy", f"fake_{i}.wav")
        # --------------- vqgan end ------------

    # all end
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"message": "Model used successfully", "input": text}


class DelModelRequest(BaseModel):
    text: Optional[str] = None


@app.post("/del-model")
async def del_model(
        request: Request,
        req: DelModelRequest
):
    """用post请求"""
    logger.info(
        f"{request.client.host}:{request.client.port}/del-model  {unquote(str(request.query_params))} "
    )
    try:
        msg_1 = llama_model_manager.del_model()
        msg_2 = vqgan_model_manager.del_model()
        # 添加使用模型的逻辑
        return {"message": msg_1 + '\n' + msg_2}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
