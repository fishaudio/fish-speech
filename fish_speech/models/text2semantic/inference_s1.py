import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
from loguru import logger
from tqdm import tqdm

from fish_speech.content_sequence import ContentSequence, TextPart, VQPart
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

S1_REPETITION_WINDOW = 16


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None and previous_tokens.numel() > 0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    forward_result = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = forward_result.logits
    hidden_states = forward_result.hidden_states

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[0]
                if previous_tokens is not None and previous_tokens.size(1) > 0
                else None
            ),
        )[0]
    ]

    # Only clear cache for fast_layers, avoid clearing main model cache
    for layer in model.fast_layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    a[a >= model.config.codebook_size] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        # S1 fast codebooks are generated from the first 1024 logits.
        short_logits = logits[:, :, :1024]

        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None and previous_tokens.size(1) > 0
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)

    del logits, hidden_states, forward_result
    return codebooks.T


def decode_n_tokens(
    model: DualARTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    generated = 0
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    for i in tqdm(range(num_new_tokens)):
        window_start = max(0, i - S1_REPETITION_WINDOW)
        window = previous_tokens[:, window_start:i]

        with sdpa_kernel(SDPBackend.MATH):
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )
        generated = i + 1

        if cur_token[0, 0, -1] == im_end_id:
            break

    del cur_token
    if generated == 0:
        return torch.empty(
            (model.config.num_codebooks + 1, 0),
            dtype=torch.int,
            device=previous_tokens.device,
        )

    return previous_tokens[:, :generated]


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

    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

    else:
        max_new_tokens = model.config.max_seq_len - T

    device = prompt.device
    prompt_dtype = prompt.dtype

    # Critical fix: Only set up cache on first run or when necessary
    if not hasattr(model, "_cache_setup_done") or not model._cache_setup_done:
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=prompt_dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temperature = getattr(
        model, "fixed_temperature", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    top_p = getattr(
        model, "fixed_top_p", torch.tensor(0.8, device=device, dtype=torch.float)
    )
    repetition_penalty = getattr(
        model,
        "fixed_repetition_penalty",
        torch.tensor(1.1, device=device, dtype=torch.float),
    )

    temp_val = sampling_kwargs.get("temperature", 0.8)
    top_p_val = sampling_kwargs.get("top_p", 0.8)
    rep_val = sampling_kwargs.get("repetition_penalty", 1.1)

    if abs(temperature.item() - temp_val) > 1e-6:
        temperature.fill_(temp_val)
    if abs(top_p.item() - top_p_val) > 1e-6:
        top_p.fill_(top_p_val)
    if abs(repetition_penalty.item() - rep_val) > 1e-6:
        repetition_penalty.fill_(rep_val)

    first_token = decode_one_token_ar(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    del first_token, x, prompt, empty, input_pos
    return seq


def init_model(checkpoint_path, device, precision, compile=False):
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info("Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        logger.info("Using DualARTransformer (S1 pipeline)")
    else:
        raise ValueError("Unsupported model type")

    model.fixed_temperature = torch.tensor(0.8, device=device, dtype=torch.float)
    model.fixed_top_p = torch.tensor(0.8, device=device, dtype=torch.float)
    model.fixed_repetition_penalty = torch.tensor(1.1, device=device, dtype=torch.float)

    model._cache_setup_done = False

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token


def generate_long(
    *,
    model,
    device: Union[str, torch.device],
    decode_one_token: Callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[Union[str, list[str]]] = None,
    prompt_tokens: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    # Keep args for compatibility with S2 request payload.
    _ = iterative_prompt
    _ = chunk_length

    use_prompt = prompt_text is not None and prompt_tokens is not None
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
    base_content_sequence = ContentSequence(modality="interleave")
    max_length = model.config.max_seq_len

    if use_prompt:
        for prompt_text_item, prompt_codes in zip(prompt_text, prompt_tokens):
            base_content_sequence.append(
                [
                    TextPart(text=prompt_text_item),
                    VQPart(codes=prompt_codes),
                ],
                add_end=True,
                speaker=0,
            )

    base_content_sequence.append(
        [
            TextPart(text=text),
        ],
        add_end=False,
        speaker=0,
    )

    encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
        tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded.size(1) > max_length - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}")

    encoded = encoded.to(device=device)
    logger.info(f"Encoded text: {text}")

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prompt_length = encoded.size(1)
        t0 = time.perf_counter()

        y = generate(
            model=model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            decode_one_token=decode_one_token,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if sample_idx == 0 and compile:
            logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0
        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"Generated {tokens_generated} tokens in {elapsed:.02f} seconds, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        codes = y[1:, prompt_length:-1].clone()
        assert (codes >= 0).all(), "Negative code found"

        yield GenerateResponse(action="sample", codes=codes, text=text)
        del y, codes

        yield GenerateResponse(action="next")


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
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

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=exc))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


@click.command()
@click.option(
    "--text",
    type=str,
    default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
)
@click.option("--prompt-text", type=str, default=None, multiple=True)
@click.option(
    "--prompt-tokens",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    multiple=True,
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.8)
@click.option("--repetition-penalty", type=float, default=1.1)
@click.option("--temperature", type=float, default=0.8)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="checkpoints/openaudio-s1-mini",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="output")
def main(
    text: str,
    prompt_text: Optional[tuple[str, ...]],
    prompt_tokens: Optional[tuple[Path, ...]],
    num_samples: int,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    if (
        prompt_text is not None
        and prompt_tokens is not None
        and len(prompt_text) != len(prompt_tokens)
    ):
        raise ValueError(
            f"Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same"
        )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    prompt_tokens_list = None
    if prompt_tokens is not None:
        prompt_tokens_list = [torch.from_numpy(np.load(path)) for path in prompt_tokens]

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
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=list(prompt_text) if prompt_text else None,
        prompt_tokens=prompt_tokens_list,
    )

    idx = 0
    codes = []
    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                codes_npy_path = output_dir / f"codes_{idx}.npy"
                np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
                logger.info(f"Saved codes to {codes_npy_path}")
            logger.info("Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
