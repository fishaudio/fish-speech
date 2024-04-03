import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from fish_speech.text.parser import clean_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from fish_speech.models.text2semantic.llama import DualARTransformer, NaiveTransformer
from fish_speech.text import g2p
from fish_speech.text.symbols import pad as pad_symbol
from fish_speech.text.symbols import pu_symbols


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
):
    if previous_tokens is not None and repetition_penalty != 1.0:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=0, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=0, index=previous_tokens, src=score)

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(
            torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
        )
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[0] = False  # keep at least one option
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=0, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    previous_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1], previous_tokens=previous_tokens, **sampling_kwargs
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)
    codebooks = [
        sample(
            x.logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]
    x = x.hidden_states

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    for codebook_idx in range(model.config.num_codebooks):
        input_pos = torch.tensor([codebook_idx], device=x.device, dtype=torch.long)
        logits = model.forward_generate_fast(x, input_pos)
        a = sample(
            logits,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
            **sampling_kwargs,
        )[0]
        x = model.fast_embeddings(a)
        codebooks.append(a)

    return torch.stack(codebooks, dim=0)


def decode_one_token_naive(
    model: NaiveTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    x = model.forward_generate(x, input_pos)

    codebooks = [
        sample(
            x.token_logits,
            previous_tokens=None,  # Disable repetition penalty for the token codebook
            **sampling_kwargs,
        )[0]
    ]

    for i in range(model.config.num_codebooks):
        codebooks.append(
            sample(
                x.codebook_logits[:, :, i],
                previous_tokens=previous_tokens[i + 1]
                if previous_tokens is not None
                else None,
                **sampling_kwargs,
            )[0]
        )

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    eos_token_id: int = 2,
    decode_one_token=decode_one_token_naive,
    **sampling_kwargs,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                **sampling_kwargs,
            )

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        # TODO: use tokenizer's eos
        if cur_token[0, 0, -1] == eos_token_id or (cur_token[0, 1:, -1] == 1).any():
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: NaiveTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int = 2,
    decode_one_token=decode_one_token_naive,
    precision: torch.dtype = torch.bfloat16,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T
            logger.info(f"Truncating max_new_tokens to {max_new_tokens}")

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=T_new, dtype=precision)

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((codebook_dim, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = decode_one_token(
        model, prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        eos_token_id=eos_token_id,
        decode_one_token=decode_one_token,
        **sampling_kwargs,
    )
    # x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def encode_tokens(
    tokenizer,
    string,
    bos=True,
    device="cuda",
    prompt_tokens=None,
    use_g2p=False,
    speaker=None,
    order="zh,jp,en",
    num_codebooks=4,
):
    if use_g2p:
        order = order.split(",")
        prompt = g2p(string, order=order)
        prompt = [
            (f"<p:{i}>" if i not in pu_symbols and i != pad_symbol else i)
            for _, i in prompt
        ]
        string = " ".join(prompt)
    else:
        string = clean_text(string)

    if speaker is not None:
        string = f"[SPK: {speaker}] {string}"

    string = f"[INST] {string} [/INST]"
    new_tokens = tokenizer.encode(
        string,
        add_special_tokens=bos,
        max_length=10**6,
        truncation=False,
    )
    tokens = torch.tensor([new_tokens], dtype=torch.int, device=device)

    # Codebooks
    zeros = torch.zeros((num_codebooks, tokens.size(1)), dtype=torch.int, device=device)
    prompt = torch.cat((tokens, zeros), dim=0)

    if prompt_tokens is None:
        return prompt

    # Get prompt tokens
    if prompt_tokens.ndim == 3:
        assert (
            prompt_tokens.shape[0] == 1
        ), f"3 dim prompt tokens should have shape (1, num_codebooks, seq_len)"
        prompt_tokens = prompt_tokens[0]

    assert prompt_tokens.ndim == 2
    data = prompt_tokens + 2

    if prompt_tokens.shape[0] > num_codebooks:
        logger.warning(
            f"Prompt tokens shape {prompt_tokens.shape} is larger than num_codebooks {num_codebooks}, getting first {num_codebooks} codebooks"
        )
        data = data[:num_codebooks]

    # Since 1.0, we use <s:xxx> to replace <semantic>
    s0_token_id = tokenizer.convert_tokens_to_ids("<s:0>")
    main_token_ids = torch.tensor(
        # TODO: replace this
        [[s0_token_id] * data.size(1)],
        dtype=torch.int,
        device=device,
    )

    data = torch.cat((main_token_ids, data), dim=0)
    prompt = torch.cat((prompt, data), dim=1)

    return prompt


def load_model(config_name, checkpoint_path, device, precision, max_length):
    with initialize(version_base="1.3", config_path="../../fish_speech/configs/model"):
        cfg = compose(
            config_name=config_name, overrides=[f"config.max_seq_len={max_length}"]
        )

    model: Union[NaiveTransformer, DualARTransformer] = instantiate(cfg)

    if "int8" in str(checkpoint_path):
        logger.info("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        logger.info("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    if any(k.startswith("model.") for k in checkpoint):
        checkpoint = {
            k.replace("model.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("model.")
        }

    model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=device, dtype=precision)
    logger.info("Restored model from checkpoint")

    return model.eval(), cfg


def split_text(text, min_length):
    text = clean_text(text)
    segments = []
    curr = ""
    for char in text:
        curr += char
        if char not in [".", ",", "!", "?"]:
            continue

        if len(curr) >= min_length:
            segments.append(curr)
            curr = ""

    if curr:
        segments.append(curr)

    return segments


@click.command()
@click.option("--text", type=str, default="你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.")
@click.option("--prompt-text", type=str, default=None)
@click.option(
    "--prompt-tokens", type=click.Path(path_type=Path, exists=True), default=None
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-k", type=int, default=None)
@click.option("--top-p", type=float, default=0.9)
@click.option("--repetition-penalty", type=float, default=1.2)
@click.option("--temperature", type=float, default=0.7)
@click.option(
    "--checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    default="results/text2semantic_400m_finetune/step_000002000.pth",
)
@click.option("--config-name", type=str, default="dual_ar_8_codebook_small")
@click.option("--tokenizer", type=str, default="fishaudio/speech-lm-v1")
@click.option("--compile/--no-compile", default=False)
@click.option("--use-g2p/--no-g2p", default=True)
@click.option("--seed", type=int, default=42)
@click.option("--speaker", type=str, default=None)
@click.option("--order", type=str, default="zh,jp,en")
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=False)
@click.option("--max-length", type=int, default=2048)
@click.option("--chunk-length", type=int, default=30)
def main(
    text: str,
    prompt_text: Optional[str],
    prompt_tokens: Optional[Path],
    num_samples: int,
    max_new_tokens: int,
    top_k: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: Path,
    config_name: str,
    tokenizer: str,
    compile: bool,
    use_g2p: bool,
    seed: int,
    speaker: Optional[str],
    order: str,
    half: bool,
    iterative_prompt: bool,
    max_length: int,
    chunk_length: int,
) -> None:
    device = "cuda"

    precision = torch.half if half else torch.bfloat16

    logger.info("Loading model ...")
    t0 = time.time()
    model, cfg = load_model(config_name, checkpoint_path, device, precision, max_length)
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    torch.cuda.synchronize()
    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    prompt_tokens = (
        torch.from_numpy(np.load(prompt_tokens)).to(device)
        if prompt_tokens is not None
        else None
    )

    use_prompt = prompt_text is not None and prompt_tokens is not None
    encoded = []
    texts = split_text(text, chunk_length) if iterative_prompt else [text]
    for idx, text in enumerate(texts):
        encoded.append(
            encode_tokens(
                tokenizer,
                string=text,
                bos=idx == 0 and not use_prompt,
                device=device,
                use_g2p=use_g2p,
                speaker=None,
                order=order,
                num_codebooks=model.config.num_codebooks,
            )
        )
        logger.info(f"Encoded text: {text}")

    if use_prompt:
        encoded_prompt = encode_tokens(
            tokenizer,
            prompt_text,
            prompt_tokens=prompt_tokens,
            bos=True,
            device=device,
            use_g2p=use_g2p,
            speaker=speaker,
            order=order,
            num_codebooks=model.config.num_codebooks,
        )

        encoded[0] = torch.cat((encoded_prompt, encoded[0]), dim=1)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = decode_one_token_naive
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

    for idx in range(num_samples):
        torch.cuda.synchronize()
        global_encoded = []
        all_codes = []
        seg_idx = 0

        while seg_idx < len(encoded):
            seg = encoded[seg_idx]
            global_encoded.append(seg)

            lengths = reversed([seg.size(1) for seg in global_encoded])

            # Pick last 2000 tokens
            count = 0
            for i, length in enumerate(lengths):
                count += length
                if count + length > max_length - 1024:
                    break

            if i != 0 and i % 2 == 0:
                i -= 1

            # Rotate the list
            if i < len(global_encoded) - 2:
                partial_encoded = global_encoded[-i:]
            else:
                partial_encoded = global_encoded

            cat_encoded = torch.cat(partial_encoded, dim=1)
            prompt_length = cat_encoded.size(1)

            t0 = time.perf_counter()
            y = generate(
                model=model,
                prompt=cat_encoded,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                decode_one_token=decode_one_token,
                precision=precision,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            if idx == 0 and seg_idx == 0 and compile:
                logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

            torch.cuda.synchronize()
            t = time.perf_counter() - t0

            tokens_generated = y.size(1) - prompt_length
            tokens_sec = tokens_generated / t
            logger.info(
                f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
            )
            logger.info(
                f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s"
            )
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

            # Put the generated tokens
            codes = y[1:, prompt_length:-1].clone()

            codes = codes - 2
            if not (codes >= 0).all():
                global_encoded.pop()
                logger.warning(f"Negative code found: {codes}, retrying ...")
                continue

            global_encoded.append(y[:, prompt_length:-1].clone())
            all_codes.append(codes)
            seg_idx += 1

        codes = torch.cat(all_codes, dim=1)
        assert (codes >= 0).all(), f"Negative code found: {codes}"

        np.save(f"codes_{idx}.npy", codes.cpu().numpy())
        logger.info(f"Saved codes to codes_{idx}.npy")


if __name__ == "__main__":
    main()
