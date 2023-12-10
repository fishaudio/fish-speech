# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from transformers import AutoTokenizer

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future


from fish_speech.models.text2semantic.llama import ModelArgs, Transformer
from fish_speech.text import g2p
from fish_speech.text.symbols import pad as pad_symbol
from fish_speech.text.symbols import pu_symbols
from tools.llama.tp import maybe_init_dist


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
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

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
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[int] = None,
    repetition_penalty: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits[0, -1], previous_tokens, temperature, top_k, top_p, repetition_penalty
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_token(
    model: Transformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    previous_tokens: torch.Tensor = None,
    **sampling_kwargs,
) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model.forward_generate(x, input_pos)
    codebooks = [
        sample(
            logits.token_logits,
            previous_tokens=previous_tokens[0] if previous_tokens is not None else None,
            **sampling_kwargs,
        )[0]
    ]

    # Disable <s> and </s> tokens for codebooks
    if model.config.num_codebooks != 0:
        logits.codebook_logits[:, :, :, :2] = -float("Inf")

        for i in range(model.config.num_codebooks):
            codebooks.append(
                sample(
                    logits.codebook_logits[:, :, i],
                    previous_tokens=previous_tokens[i]
                    if previous_tokens is not None
                    else None,
                    **sampling_kwargs,
                )[0]
            )

    return torch.stack(codebooks, dim=0)


def decode_n_tokens(
    model: Transformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    callback=lambda _: _,
    **sampling_kwargs,
):
    new_tokens = []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_token(
                model,
                cur_token,
                input_pos,
                torch.concat(new_tokens, dim=1) if len(new_tokens) > 0 else None,
                **sampling_kwargs,
            )
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)

        # TODO: use tokenizer's eos
        if (cur_token[0, 0, -1] == 2).any():
            print("EOS detected, stopping generation")
            break

    return new_tokens


def model_forward(model, x, input_pos):
    return model(x, input_pos)


@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    callback=lambda x: x,
    **sampling_kwargs,
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    if T + max_new_tokens > model.config.max_seq_len:
        max_new_tokens = model.config.max_seq_len - T
        print(f"Truncating max_new_tokens to {max_new_tokens}")

    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.max_seq_len)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_len=max_seq_length)

    codebook_dim = 1 + model.config.num_codebooks
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((codebook_dim, T_new), dtype=dtype, device=device)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    next_token = decode_token(
        model, prompt.view(1, codebook_dim, -1), input_pos, **sampling_kwargs
    )
    seq[:, T : T + 1] = next_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    generated_tokens = decode_n_tokens(
        model,
        next_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        callback=callback,
        **sampling_kwargs,
    )
    x = torch.cat(generated_tokens, dim=1)
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x

    return seq


def encode_tokens(tokenizer, string, bos=True, device="cuda"):
    # data/Genshin/Chinese/神里绫华/vo_ayaka_character_idle_04.npy

    prompt = g2p("<zh>算啦，虽然他罪无可恕，但也有可怜的地方嘛。</zh> {string}")
    prompt = [
        (f"<p:{i}>" if i not in pu_symbols and i != pad_symbol else i)
        for _, i in prompt
    ]
    prompt = " ".join(prompt)
    string = f"[INST] {prompt} [/INST]"
    print("Encoding string:", string)

    data = np.load("data/Genshin/Chinese/派蒙/vo_WYLQ103_10_paimon_03.npy")
    codes = [f"<s:{i}>" for i in data[0]]

    tokens = tokenizer.encode(
        string + " ".join(codes),
        max_length=10**6,
        add_special_tokens=bos,
        truncation=False,
    )
    tokens = torch.tensor([tokens], dtype=torch.int, device=device)

    # Codebooks
    # zeros = torch.zeros((4, tokens.size(1)), dtype=torch.int, device=device)
    # prompt = torch.cat((tokens, zeros), dim=0)

    # # Get prompt tokens
    # data = np.load("data/Genshin/Chinese/神里绫华/vo_ayaka_character_idle_02.npy")
    # data = torch.from_numpy(data).to(device=device, dtype=torch.int) + 2

    # zeros = torch.zeros((1, data.size(1)), dtype=torch.int, device=device) + 32311 # 32311 is the <pad> token
    # data = torch.cat((zeros, data), dim=0)
    # prompt = torch.cat((prompt, data), dim=1)
    # print(prompt)

    return tokens


def _load_model(checkpoint_path, device, precision, use_tp):
    with torch.device("meta"):
        # TODO: support different model archs
        model = Transformer(
            ModelArgs(
                max_seq_len=4096,
                vocab_size=36408,
                n_layer=24,
                n_head=16,
                dim=1024,
                rope_base=10000,
                norm_eps=1e-5,
                codebook_size=168,
                num_codebooks=0,
            )
        )

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler

        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 quantization!")
        path_comps = checkpoint_path.name.split(".")
        assert path_comps[-2].startswith("g")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler

        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp

        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()


B_INST, E_INST = "[INST]", "[/INST]"


def main(
    prompt: str = "你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = None,
    top_p: int = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 0.8,
    checkpoint_path: Path = Path(
        "results/text2semantic_400m/checkpoints/step_000025000.ckpt"
    ),
    compile: bool = True,
    profile: Optional[Path] = None,
    tokenizer: str = "fishaudio/speech-lm-v1",
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer."""
    assert checkpoint_path.is_file(), checkpoint_path

    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        torch.cuda.set_device(rank)
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = "cuda"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    print(prompt)
    encoded = encode_tokens(tokenizer, f"{prompt}", bos=True, device=device)
    prompt_length = encoded.size(1)

    torch.manual_seed(1234)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )
    if compile:
        global decode_token
        decode_token = torch.compile(
            decode_token, mode="reduce-overhead", fullgraph=True
        )

    aggregate_metrics = {
        "tokens_per_sec": [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        torch.cuda.synchronize()
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode(".")[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print("".join(buffer), end="", flush=True)
                    buffer.clear()
                # print(, end='', flush=True)

        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib

        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            y = generate(
                model,
                encoded,
                max_new_tokens,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        if not interactive:
            print(tokenizer.decode(y[0, :prompt_length:].tolist()))
            print(f"Generated {y.size(1) - prompt_length} tokens")
            # Find all <s:2769>
            codes = y[0, prompt_length:-1]
            codes = codes - 32311
            # print(codes)
            assert (codes >= 0).all()
            import numpy as np

            np.save(f"codes_{i}.npy", codes[None].cpu().numpy())
        else:
            print()
        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / t
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        print(
            f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")

    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}"
    )
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Your CLI description.")

    parser.add_argument(
        "--prompt",
        type=str,
        default="感情分析関数では、大規模言語モデルに古典的な散文を分析させます。 分析の観点は比較的単純ですが、論理的な誤りはなく、依然として自己一貫性があることがわかります。",
        help="Input prompt.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to launch in interactive mode",
    )
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples.")
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens."
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling.")
    parser.add_argument("--top_p", type=int, default=0.95, help="Top-k for sampling.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature for sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("results/text2semantic_400m/step_000095000_weights.ckpt"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--compile", action="store_true", help="Whether to compile the model."
    )
    parser.add_argument("--profile", type=Path, default=None, help="Profile path.")

    args = parser.parse_args()
    main(
        args.prompt,
        args.interactive,
        args.num_samples,
        args.max_new_tokens,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.temperature,
        args.checkpoint_path,
        args.compile,
        args.profile,
    )
