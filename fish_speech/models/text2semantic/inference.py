import inspect
import os
import queue
import re
import threading
import time
import traceback
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import click
import numpy as np
import torch
import torch._inductor.config
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from tqdm import tqdm

from fish_speech.content_sequence import (
    TextPart,
    VQPart,
)
from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.chunking import (
    DRAMA3_CHUNK_MAX_NEW_TOKENS,
    Drama3History,
    split_drama3_text,
)
from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_DEFAULT_MAX_NEW_TOKENS,
    Drama3DecodeGraphs,
    Drama3PromptTooLongError,
    build_drama3_prompt,
    generate_drama3,
    resolve_drama3_devices,
    resolve_drama3_dtype,
)
from fish_speech.models.text2semantic.fish_omni.model import (
    FishOmniModel,
    is_fish_omni_checkpoint,
)
from fish_speech.models.text2semantic.fish_omni.tower import (
    REFERENCE_SAMPLING_FALLBACKS,
    ReferenceSamplingFallback,
)
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_CHUNK_LENGTH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from fish_speech.tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.rand_like(probs_sort)
    q = -torch.log(q)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


RAS_WIN_SIZE = 10  # window for Repetition Aware Sampling
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,  # 注意: 我看到你传进来的是 int，这很关键
) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    top_k_mask = indices >= top_k
    sorted_indices_to_remove = (cum_probs > top_p) | top_k_mask
    sorted_indices_to_remove[..., 0] = False  # keep the top-1 token, per row

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = torch.where(
        indices_to_remove, float("-Inf"), logits
    )  # 同样替换 masked_fill_ 为 torch.where
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: torch.Tensor | None = None,
    kv_len: int | None = None,
) -> torch.Tensor:
    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        kv_len=kv_len,
    )
    logits = forward_result.logits  # (1, 1, vocab_size)
    hidden_states = forward_result.hidden_states

    # Apply constrained decoding: only allow semantic tokens + im_end
    biased_logits = logits + semantic_logit_bias

    # Normal sample
    main_token_normal = sample(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )[0]

    # RAS: also sample with high temp to use as fallback if token repeats
    high_temp = torch.tensor(
        RAS_HIGH_TEMP, device=temperature.device, dtype=temperature.dtype
    )
    high_top_p = torch.tensor(RAS_HIGH_TOP_P, device=top_p.device, dtype=top_p.dtype)
    main_token_high = sample(
        biased_logits, temperature=high_temp, top_p=high_top_p, top_k=top_k
    )[0]

    # Use high-temp sample if: token is semantic AND token is in previous window
    if previous_tokens is not None:
        in_window = (previous_tokens[0] == main_token_normal).any()
        # Use tensor ops (&, torch.where) instead of Python (and, if) — torch.compile requires no data-dependent branching
        is_semantic = (main_token_normal >= model.config.semantic_begin_id) & (
            main_token_normal <= model.config.semantic_end_id
        )
        should_use_high = in_window & is_semantic
        main_token_normal = torch.where(
            should_use_high, main_token_high, main_token_normal
        )

    codebooks = [main_token_normal]

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)

    a = codebooks[0] - model.config.semantic_begin_id
    a = torch.clamp(a, min=0, max=model.config.codebook_size - 1)

    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits  # DualAR predicts config.codebook_size number of tokens

        # Convert logits to probs (no constrain for fast codebooks)
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    # Only delete references, let Python GC handle cleanup
    del logits, hidden_states, forward_result

    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: int,
    semantic_logit_bias: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    kv_start_pos: int | None = None,
):
    if kv_start_pos is None:
        # Compatibility fallback for direct callers. The production generation
        # path passes the Python position explicitly to avoid a device sync.
        kv_start_pos = int(input_pos[0].item())

    # Rolling window for RAS (Repetition Aware Sampling)
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, RAS_WIN_SIZE),
        dtype=torch.int,
        device=cur_token.device,
    )
    # Accumulate all generated tokens (the actual output)
    new_tokens = []

    # [MODIFIED] Pre-fetch ID for efficiency loop
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)

    callback_parameters = inspect.signature(decode_one_token).parameters
    supports_kv_len = "kv_len" in callback_parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in callback_parameters.values()
    )

    for i in tqdm(range(num_new_tokens)):
        decode_kwargs = {
            "model": model,
            "x": cur_token,
            "input_pos": input_pos,
            "previous_tokens": previous_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "semantic_logit_bias": semantic_logit_bias,
            "audio_masks": audio_masks,
            "audio_parts": audio_parts,
        }
        if supports_kv_len:
            decode_kwargs["kv_len"] = kv_start_pos + i + 1

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(**decode_kwargs).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        # Roll RAS window left and insert new token at end
        previous_tokens = previous_tokens.roll(-1, dims=1)
        previous_tokens[:, -1] = next_token.view(model.config.num_codebooks + 1, -1)[
            :, 0
        ]
        new_tokens.append(next_token)

        if cur_token[0, 0, -1] == im_end_id:
            break

    if not new_tokens:  # num_new_tokens <= 0
        return cur_token.new_empty((model.config.num_codebooks + 1, 0))

    del cur_token

    return torch.cat(new_tokens, dim=1)


def generated_codes(
    y: torch.Tensor, prompt_length: int, im_end_id: int
) -> torch.Tensor:
    """The codebook rows after the prompt, without the closing <|im_end|> frame;
    a batch that ran out of max_new_tokens has none, so its last frame is audio."""
    end = y.size(1) - int(y[0, -1].item() == im_end_id)
    return y[1:, prompt_length:end].clone()


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: DualARTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise InvalidInputError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    # max_new_tokens <= 0 means up to the context length.
    if max_new_tokens > 0:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device = prompt.device
    dtype = next(
        model.parameters()
    ).dtype  # model weight dtype (bfloat16), NOT prompt dtype (int32)

    # Critical fix: Only set up cache on first run or when necessary
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,  # Fixed to 1, avoid dynamic changes
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks

    # Create new tensor each time, but try to reuse memory
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt.dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temp_val = sampling_kwargs.get("temperature", DEFAULT_TEMPERATURE)
    top_p_val = sampling_kwargs.get("top_p", DEFAULT_TOP_P)
    top_k_val = sampling_kwargs.get("top_k", DEFAULT_TOP_K)

    temperature = torch.tensor(temp_val, device=device, dtype=dtype)
    top_p = torch.tensor(top_p_val, device=device, dtype=dtype)

    # Build semantic logit bias: 0 for semantic tokens + im_end, -inf for all others
    vocab_size = model.config.vocab_size
    semantic_logit_bias = torch.full(
        (1, 1, vocab_size), float("-inf"), device=device, dtype=dtype
    )

    # [MODIFIED] Use config for semantic range
    semantic_logit_bias[
        0, 0, model.config.semantic_begin_id : model.config.semantic_end_id + 1
    ] = 0.0

    # [MODIFIED] Use tokenizer.get_token_id (Wrapper method)
    semantic_logit_bias[0, 0, model.tokenizer.get_token_id(IM_END_TOKEN)] = 0.0

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        top_k_val,
        semantic_logit_bias,
        audio_masks,
        audio_parts,
        kv_len=T,
    ).clone()
    seq[:, T : T + 1] = first_token

    # Recreate input_pos
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k_val,
        semantic_logit_bias=semantic_logit_bias,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
        kv_start_pos=T,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    # Clean up temporary variables
    del first_token, x, prompt, empty, input_pos

    return seq


def init_model(
    checkpoint_path,
    device,
    precision,
    compile=False,
    reference_sampling: ReferenceSamplingFallback | None = None,
):
    """reference_sampling, Drama 3 only: the repetition fallback of Fish Audio's
    reference sampling, which then replaces the default sampler; None keeps it."""
    if reference_sampling not in (None, *REFERENCE_SAMPLING_FALLBACKS):
        raise ValueError(
            f"reference_sampling must be one of {REFERENCE_SAMPLING_FALLBACKS} or "
            f"None; got {reference_sampling!r}"
        )
    if is_fish_omni_checkpoint(checkpoint_path):
        if compile:
            raise ValueError("--compile is not supported for Drama 3 checkpoints")
        devices, tower_device = resolve_drama3_devices(device)
        dtype = resolve_drama3_dtype(precision, [*devices, tower_device])
        logger.info(
            f"Using FishOmniModel (Drama 3): backbone on {devices}, "
            f"tower on {tower_device}, {dtype}"
        )
        model = FishOmniModel.from_pretrained(
            checkpoint_path, devices=devices, tower_device=tower_device, dtype=dtype
        )
        if reference_sampling is not None:
            model.audio_decoder.reference_sampling = reference_sampling
            logger.info(
                f"Drama 3 sampling: reference sampling, {reference_sampling} fallback"
            )
        return model, None
    if reference_sampling is not None:
        raise ValueError("reference sampling is supported for Drama 3 checkpoints only")

    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        # prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Mark whether cache has been initialized
    model._cache_setup_done = False

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="default" if torch.cuda.is_available() else None,
            fullgraph=True,
            dynamic=True,
        )

    return model.eval(), decode_one_token


def resolve_codec_path(
    checkpoint_path: str | Path, codec_path: str | Path | None = None
) -> Path:
    """codec_path, else <checkpoint>/codec.pth, else the sibling s2-pro/codec.pth.

    Drama 3 uses the S2-Pro codec, and fine-tuned S2-Pro directories may not carry one.
    """
    if codec_path is not None:
        codec_path = Path(codec_path)
        if not codec_path.is_file():
            raise FileNotFoundError(f"codec checkpoint does not exist: {codec_path}")
        return codec_path
    checkpoint_path = Path(checkpoint_path)
    candidate = checkpoint_path / "codec.pth"
    if candidate.is_file():
        return candidate
    fallback = checkpoint_path.parent / "s2-pro" / "codec.pth"
    if fallback.is_file():
        logger.info(
            f"{checkpoint_path} has no codec.pth; using the S2-Pro codec {fallback}"
        )
        return fallback
    raise FileNotFoundError(
        f"no codec.pth in {checkpoint_path} or {fallback.parent}; pass the codec "
        "path (--codec-path for the CLI, --decoder-checkpoint-path for the server)"
    )


@torch.inference_mode()
def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
    """Load the DAC codec model for audio encoding/decoding.

    Weight norm is folded in fp32 before the cast, which keeps FP16 exact enough.
    """
    config_path = Path(__file__).parent.parent.parent / "configs" / "modded_dac_vq.yaml"
    cfg = OmegaConf.load(str(config_path))
    codec = instantiate(cfg)

    state_dict = torch.load(
        codec_checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    rebuilt_buffers = {
        f"{name}.{buffer}" if name else buffer
        for name, module in codec.named_modules()
        for buffer in module._non_persistent_buffers_set
    }
    state_dict = {k: v for k, v in state_dict.items() if k not in rebuilt_buffers}
    codec.load_state_dict(state_dict, strict=True)
    codec.eval()
    for module in list(codec.modules()):
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    codec.to(device=device, dtype=precision)
    logger.info(f"Loaded codec {codec_checkpoint_path} on {device} ({precision})")
    return codec


@torch.inference_mode()
def encode_audio(audio_path, codec, device):
    """Encode an audio file to VQ codes."""
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = torchaudio.functional.resample(wav.to(device), sr, codec.sample_rate)[0]

    # Match codec model dtype (e.g. bfloat16)
    model_dtype = next(codec.parameters()).dtype
    audios = wav[None, None].to(dtype=model_dtype)  # (1, 1, T)
    audio_lengths = torch.tensor([len(wav)], device=device, dtype=torch.long)

    indices, feature_lengths = codec.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]  # (num_codebooks, T)


@torch.inference_mode()
def decode_to_audio(codes, codec):
    """Decode VQ codes to audio waveform."""
    # codes: (num_codebooks, T) -> (1, num_codebooks, T)
    audio = codec.from_indices(codes[None])
    return audio[0, 0]  # (T,) mono waveform


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: torch.Tensor | None = None
    text: str | None = None


def split_text_by_speaker(text: str) -> list[str]:
    """
    Split text into turns based on <|speaker:X|> tags.

    Args:
        text: The full text with speaker tags

    Returns:
        List of speaker turns, each starting with <|speaker:X|>
    """
    pattern = r"(<\|speaker:\d+\|>)"
    parts = re.split(pattern, text)

    turns = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if re.match(pattern, part):
            if i + 1 < len(parts):
                turn = part + parts[i + 1]
                turns.append(turn.strip())
                i += 2
            else:
                turns.append(part)
                i += 1
        elif part:
            # Text before the first speaker tag (or any untagged fragment) was being
            # dropped. Assign it to speaker 0 so it is still synthesized, matching
            # parse_speaker_segments() in fish_omni/chunking.py.
            turns.append(f"<|speaker:0|>{part}")
            i += 1
        else:
            i += 1

    return turns


def group_turns_into_batches(
    turns: list[str], max_speakers: int = 3, max_bytes: int = 300
) -> list[str]:
    """
    Group turns into batches based on speaker count or byte limit.

    Args:
        turns: List of speaker turns
        max_speakers: Maximum number of speakers per batch (default 3)
        max_bytes: Maximum UTF-8 bytes per batch (default 300)

    Returns:
        List of batched text strings
    """

    def _speaker_of(turn: str) -> str | None:
        match = re.match(r"<\|speaker:\d+\|>", turn.strip())
        return match.group(0) if match else None

    batches = []
    current_batch = []
    current_speakers = set()
    current_bytes = 0

    for turn in turns:
        turn_bytes = len(turn.encode("utf-8"))
        speaker = _speaker_of(turn)

        # max_speakers limits DISTINCT speakers per batch (per the docstring), not the
        # number of turns. Counting len(current_batch) split monologues / repeated
        # speakers into needless batches; only a new, unseen speaker counts here.
        adds_speaker = speaker is not None and speaker not in current_speakers
        would_exceed_speakers = adds_speaker and len(current_speakers) >= max_speakers
        would_exceed_bytes = current_bytes + turn_bytes > max_bytes and current_batch

        if would_exceed_speakers or would_exceed_bytes:
            batches.append("\n".join(current_batch))
            current_batch = [turn]
            current_bytes = turn_bytes
            current_speakers = {speaker} if speaker is not None else set()
        else:
            current_batch.append(turn)
            current_bytes += turn_bytes
            if speaker is not None:
                current_speakers.add(speaker)

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    # Accepted for API compatibility and ignored: decoding uses repetition-aware
    # sampling (RAS) instead.
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    temperature: float = DEFAULT_TEMPERATURE,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = DEFAULT_CHUNK_LENGTH,
    prompt_text: str | list[str] | None = None,
    prompt_tokens: torch.Tensor | list[torch.Tensor] | None = None,
    single_chunk: bool = False,
    # Drama 3 reference sampling only; None leaves it to the sampler.
    epsilon_cutoff: float | None = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    if isinstance(model, FishOmniModel):
        yield from generate_long_drama3(
            model=model,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
            single_chunk=single_chunk,
            epsilon_cutoff=epsilon_cutoff,
        )
        return
    if epsilon_cutoff:
        raise InvalidInputError(
            "epsilon_cutoff applies to Drama 3 reference sampling only"
        )
    if top_k < 1:
        raise InvalidInputError(f"top_k must be at least 1 for S2-Pro; got {top_k}")

    use_prompt = bool(prompt_text) and bool(prompt_tokens)
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    if use_prompt:
        assert len(prompt_text) == len(
            prompt_tokens
        ), "Prompt text and tokens must have the same length"

    if prompt_tokens:
        prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    max_length = model.config.max_seq_len

    # Build base conversation with system message
    base_conversation = Conversation()

    if use_prompt:
        # Auto-add speaker tags to prompt texts that don't have them
        tagged_prompt_text = []
        for i, t in enumerate(prompt_text):
            if not re.search(r"<\|speaker:\d+\|>", t):
                tagged_prompt_text.append(f"<|speaker:{i}|>{t}")
            else:
                tagged_prompt_text.append(t)

        system_parts = [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
        ]
        reference_text = "\n".join(tagged_prompt_text)
        system_parts.append(TextPart(text=reference_text, cal_loss=False))
        system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
        all_codes = torch.cat([c for c in prompt_tokens], dim=1)
        system_parts.append(VQPart(codes=all_codes, cal_loss=False))
        # torch.save(all_codes, "debug_vq_codes.pt")
    else:
        system_parts = [
            TextPart(text="convert the provided text to speech", cal_loss=False)
        ]

    base_conversation.append(
        Message(
            role="system",
            parts=system_parts,
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )
    )

    # Split text by speaker and group into batches
    turns = split_text_by_speaker(text)
    if turns:
        batches = group_turns_into_batches(
            turns, max_speakers=5, max_bytes=chunk_length
        )
    else:
        batches = [text]

    logger.info(f"Split into {len(turns)} turns, grouped into {len(batches)} batches")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        # Deep copy base conversation for this sample
        conversation = deepcopy(base_conversation)

        for batch_idx, batch_text in enumerate(batches):
            logger.info(
                f"--- Sample {sample_idx}, Batch {batch_idx} "
                f"({len(batch_text.encode('utf-8'))} bytes) ---"
            )
            logger.info(f"Batch text: {batch_text}")

            # Add user message
            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(text=batch_text, cal_loss=False)],
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            # Deep copy for generation (don't pollute original conversation)
            conversation_gen = deepcopy(conversation)
            conversation_gen.append(
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_start=True,
                    add_im_end=False,
                )
            )

            logger.info("Visualizing prompt structure:")
            conversation_gen.visualize(
                tokenizer,
                merge_audio_tokens=True,
                merge_semantic_tokens=True,
            )

            encoded, audio_masks, audio_parts = conversation_gen.encode_for_inference(
                tokenizer, num_codebooks=model.config.num_codebooks
            )

            logger.info(f"Encoded prompt shape: {encoded.shape}")
            if audio_parts is not None:
                logger.info(f"Audio parts shape: {audio_parts.shape}")
            if audio_masks is not None:
                logger.info(
                    f"Audio masks non-zero count: {torch.count_nonzero(audio_masks)}"
                )

            if encoded.size(1) > max_length - 2048:
                raise InvalidInputError(
                    f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}"
                )

            encoded = encoded.to(device=device)
            prompt_length = encoded.size(1)

            y = generate(
                model=model,
                prompt=encoded,
                max_new_tokens=max_new_tokens,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
                decode_one_token=decode_one_token,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if sample_idx == 0 and batch_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_batch = time.perf_counter() - t0
            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t_batch if t_batch > 0 else 0
            logger.info(
                f"Batch {batch_idx}: Generated {tokens_generated} tokens in "
                f"{t_batch:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )

            codes = generated_codes(
                y, prompt_length, tokenizer.get_token_id(IM_END_TOKEN)
            )
            assert (codes >= 0).all(), f"Negative code found: {codes}"
            if codes.size(1) == y.size(1) - prompt_length:
                logger.warning(
                    "Reached max_new_tokens before EOS; audio may be truncated"
                )

            if codes.size(1) == 0:
                logger.warning("The model emitted EOS before any audio frame")
                conversation.messages.pop()  # this batch's unanswered user turn
            else:
                # Add assistant message with generated codes back to conversation
                conversation.append(
                    Message(
                        role="assistant",
                        parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
                        cal_loss=False,
                        modality="voice",
                        add_im_start=True,
                        add_im_end=True,
                    )
                )
                yield GenerateResponse(action="sample", codes=codes, text=batch_text)

            # Cleanup
            del y, encoded

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        yield GenerateResponse(action="next")


def generate_long_drama3(
    *,
    model: FishOmniModel,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    prompt_text: str | list[str] | None = None,
    prompt_tokens: torch.Tensor | list[torch.Tensor] | None = None,
    single_chunk: bool = False,
    epsilon_cutoff: float | None = None,
) -> Iterator[GenerateResponse]:
    """Drama 3 counterpart of generate_long, chunked like the hosted server.

    The text is split by split_drama3_text and the chunks are generated one after
    another, each yielded as soon as it is done; the first chunk and the latest one
    go back into the next prompt as history. single_chunk generates the whole text
    in one pass instead. max_new_tokens <= 0 means 600 frames per chunk (2048 with
    single_chunk); every request is clamped so prompt rows plus max_new_tokens stay
    within the CUDA-graph capacity.
    """
    if isinstance(prompt_text, str):
        prompt_text, prompt_tokens = [prompt_text], [prompt_tokens]
    prompt_text = list(prompt_text or [])
    prompt_tokens = [codes.cpu() for codes in (prompt_tokens or [])]
    chunks = [text] if single_chunk else split_drama3_text(text) or [text]
    # Reject bad text in any chunk before generating: failing at a later chunk
    # would waste the earlier ones, and cut a stream short after it started.
    capacity = Drama3DecodeGraphs.graph_capacity(model)
    for chunk in chunks:
        rows = len(
            build_drama3_prompt(
                chunk, model.tokenizer, model.config, prompt_text, prompt_tokens
            )
        )
        if rows >= capacity:
            raise InvalidInputError(f"prompt has {rows} rows; no room left to generate")
    requested = max_new_tokens
    if requested <= 0:
        requested = (
            DRAMA3_DEFAULT_MAX_NEW_TOKENS
            if single_chunk
            else DRAMA3_CHUNK_MAX_NEW_TOKENS
        )
    logger.info(
        f"Drama 3: {len(chunks)} chunk(s), {len(prompt_tokens)} reference(s), "
        f"max_new_tokens={requested} per chunk"
    )
    devices = {p.device for p in model.backbone.parameters()} | {model.device}
    devices = sorted((d for d in devices if d.type == "cuda"), key=str)

    for sample_idx in range(num_samples):
        history = Drama3History()
        for chunk_idx, chunk in enumerate(chunks):
            prompt = build_drama3_chunk_prompt(
                model, chunk, prompt_text, prompt_tokens, history
            )
            # The caches grow to fit a request and never shrink, and growing them
            # drops the captured CUDA graphs, so no request may exceed this.
            limit = capacity - len(prompt)
            if limit < 1:
                raise InvalidInputError(
                    f"prompt has {len(prompt)} rows; no room left to generate"
                )
            if max_new_tokens > limit:
                logger.warning(
                    f"max_new_tokens={max_new_tokens} exceeds the {limit} frames left "
                    f"after the {len(prompt)}-row prompt; clamping"
                )
            budget = min(requested, limit)

            for device in devices:
                torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            frames, finish_reason = generate_drama3(
                model,
                prompt,
                max_new_tokens=budget,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                epsilon_cutoff=epsilon_cutoff,
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Sample {sample_idx}, chunk {chunk_idx + 1}/{len(chunks)}: "
                f"{len(prompt)}-row prompt, {len(frames)} frames in {elapsed:.2f}s "
                f"({len(frames) / elapsed:.2f} frames/s), finish={finish_reason}"
            )
            peaks = (
                f"{d}: {torch.cuda.max_memory_allocated(d) / 2**30:.1f} GiB"
                for d in devices
            )
            logger.info(f"Peak GPU memory: {', '.join(peaks)}")
            if finish_reason != "eos":
                logger.warning(
                    "Reached max_new_tokens before EOS; audio may be truncated"
                )
            if len(frames):
                codes = frames[:, 1:].T.contiguous()
                yield GenerateResponse(action="sample", codes=codes, text=chunk)
                history.push(chunk, codes.cpu())
            else:
                logger.warning("The model emitted EOS before any audio frame")
        yield GenerateResponse(action="next")


def build_drama3_chunk_prompt(
    model: FishOmniModel,
    text: str,
    prompt_text: list[str],
    prompt_tokens: list[torch.Tensor],
    history: Drama3History,
) -> torch.Tensor:
    """The prompt for one chunk with as much history as fits DRAMA3_MAX_PROMPT_ROWS."""
    error = None
    for turns in history.candidates():
        try:
            return build_drama3_prompt(
                text,
                model.tokenizer,
                model.config,
                prompt_text,
                prompt_tokens,
                history=turns,
            )
        except Drama3PromptTooLongError as too_long:
            error = too_long
    raise error


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: GenerateResponse | Exception | None = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
    reference_sampling: ReferenceSamplingFallback | None = None,
):
    input_queue = queue.Queue()
    init_event = threading.Event()
    init_errors: list[Exception] = []

    def worker():
        try:
            model, decode_one_token = init_model(
                checkpoint_path,
                device,
                precision,
                compile=compile,
                reference_sampling=reference_sampling,
            )
            # Drama 3 sizes its caches per request (prompt rows + max_new_tokens).
            if not isinstance(model, FishOmniModel):
                with torch.device(device):
                    model.setup_caches(
                        max_batch_size=1,
                        max_seq_len=model.config.max_seq_len,
                        dtype=next(model.parameters()).dtype,
                    )
        except Exception as e:
            # Re-raised by the caller, which would otherwise wait forever.
            init_errors.append(e)
            return
        finally:
            init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )

                # Only clear cache after complete request batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))
                # Clear cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()
    if init_errors:
        raise init_errors[0]

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="<|speaker:0|>你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-samples", type=int, default=1)
@click.option(
    "--max-new-tokens",
    type=int,
    default=DEFAULT_MAX_NEW_TOKENS,
    help="Per text chunk; 0 means the model's own cap (S2-Pro: the context "
    "length; Drama 3: 600 per chunk, 2048 with --single-chunk).",
)
@click.option("--top-p", type=float, default=DEFAULT_TOP_P)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    help="Under --reference-sampling, 0 removes the cap.",
)
@click.option("--temperature", type=float, default=DEFAULT_TEMPERATURE)
@click.option(
    "--reference-sampling/--no-reference-sampling",
    default=False,
    help="Drama 3: sample with Fish Audio's reference sampling (epsilon cutoff on the "
    "main token, every repeated main token replaced, residual codes kept inside the "
    "codec's codebooks). Recommended: --temperature 1.0 --top-p 1.0 --top-k 0.",
)
@click.option(
    "--reference-sampling-fallback",
    type=click.Choice(REFERENCE_SAMPLING_FALLBACKS),
    default="argmax",
    help="What replaces a main token found among the last 16: the most probable "
    "candidate (argmax) or a draw at temperature 1.0, top-p 0.95, top-k 30 (ras).",
)
@click.option(
    "--epsilon-cutoff",
    type=float,
    default=None,
    help="Reference sampling's main-token epsilon cutoff in [0, 1); default 1e-3, 0 "
    "turns it off.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/s2-pro",
)
@click.option(
    "--device",
    type=str,
    default="cuda",
    help="Drama 3 accepts a comma-separated list; the last device holds tower + codec.",
)
@click.option(
    "--codec-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Defaults to <checkpoint>/codec.pth, else the sibling s2-pro/codec.pth.",
)
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option(
    "--half/--no-half",
    default=False,
    help="float16 instead of bfloat16; Drama 3 switches to float16 by itself before SM80.",
)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=DEFAULT_CHUNK_LENGTH)
@click.option(
    "--single-chunk/--no-single-chunk",
    default=False,
    help="Drama 3: generate the whole text in one pass instead of chunks.",
)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: tuple[str, ...] | None,
    prompt_tokens: tuple[Path, ...] | None,
    prompt_audio: tuple[Path, ...] | None,
    output: Path | None,
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    reference_sampling: bool,
    reference_sampling_fallback: ReferenceSamplingFallback,
    epsilon_cutoff: float | None,
    checkpoint_path: Path,
    device: str,
    codec_path: Path | None,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    single_chunk: bool,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if prompt_text and not prompt_audio and not prompt_tokens:
        raise ValueError(
            "--prompt-text requires either --prompt-audio or --prompt-tokens"
        )
    if prompt_text and prompt_tokens and len(prompt_text) != len(prompt_tokens):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )
    if prompt_text and prompt_audio and len(prompt_text) != len(prompt_audio):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt audio ({len(prompt_audio)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path,
        device,
        precision,
        compile=compile,
        reference_sampling=reference_sampling_fallback if reference_sampling else None,
    )
    if isinstance(model, FishOmniModel):
        # The codec shares the tower's device and runs in the model's dtype.
        codec_device, codec_precision = str(model.device), model.dtype
    else:
        codec_device, codec_precision = device, precision
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    codec = None
    # Resolved before generating, so a missing codec fails fast.
    codec_checkpoint = (
        resolve_codec_path(checkpoint_path, codec_path)
        if prompt_audio or output
        else None
    )

    # Handle prompt: --prompt-audio takes priority over --prompt-tokens
    prompt_tokens_list = None
    if prompt_audio:
        logger.info("Loading codec model for audio encoding...")
        codec = load_codec_model(codec_checkpoint, codec_device, codec_precision)
        prompt_tokens_list = [
            encode_audio(p, codec, codec_device).cpu() for p in prompt_audio
        ]
        logger.info(f"Encoded {len(prompt_audio)} audio file(s) to VQ codes")
    elif prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(p)) for p in prompt_tokens]

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        single_chunk=single_chunk,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
        epsilon_cutoff=epsilon_cutoff,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                merged_codes = torch.cat(codes, dim=1)
                codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                np.save(codes_npy_path, merged_codes.cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")

                # Decode to wav if --output is specified
                if output:
                    if codec is None:
                        logger.info("Loading codec model for audio decoding...")
                        codec = load_codec_model(
                            codec_checkpoint, codec_device, codec_precision
                        )
                    audio = decode_to_audio(merged_codes.to(codec_device), codec)
                    import soundfile as sf

                    out_path = (
                        str(output)
                        if num_samples == 1
                        else str(output.with_stem(f"{output.stem}_{idx}"))
                    )
                    sf.write(out_path, audio.cpu().float().numpy(), codec.sample_rate)
                    logger.info(f"Saved audio to {out_path}")

            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
