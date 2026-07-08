# AGENTS.md

Notes verified end-to-end on a real **Tesla T4 (16GB, Turing, sm_75)**. No code changes — this
file only documents what actually happens when you run the documented commands on a 16GB GPU,
since the existing docs (`docs/en/install.md`, `docs/en/inference.md`) recommend 24GB and don't
say what happens below that, or why.

## TL;DR for anyone (or any agent) on a 16GB GPU

The documented `fish_speech/models/text2semantic/inference.py` command, run exactly as written
against the `fishaudio/s2-pro` checkpoint, **will OOM on a 16GB GPU before it produces a single
token** — even for a one-sentence input with no reference audio. This is not about the model
being "too big for its weights" (the bf16 weights are ~8.5GB) — it's because the KV cache is
**always pre-allocated for the full `max_seq_len` (32768) declared in `checkpoints/s2-pro/config.json`,
regardless of how short your prompt or `--max-new-tokens` is.** A one-word input allocates the
exact same ~4.4GB KV cache as a 32k-token input would.

**Config-only workaround that works:** lower `text_config.max_seq_len` in
`checkpoints/s2-pro/config.json` to a value that comfortably covers your real generation length
(we used `4096`, still far larger than any single-utterance TTS request needs) *before* loading
the model. This is a one-line edit to the downloaded checkpoint's `config.json`, not a code
change. With this, inference fits comfortably in a 16GB card with headroom to spare.

## What we measured (reproduced twice each)

Environment: Tesla T4 16GB, driver 550.163.01 (CUDA 12.8 reported by driver's forward
compatibility), Python 3.10.12, torch 2.8.0+cu128, transformers 4.57.3.

| Config | Result | Peak VRAM (nvidia-smi) | Peak VRAM (`torch.cuda.max_memory_reserved`) | Gen time |
|---|---|---|---|---|
| `max_seq_len=32768` (checkpoint default, as-downloaded) | **CUDA OOM**, reproduced 2x, always during the very first forward pass (prefill), before any token is generated | 15593 MiB (both runs, right before crash) | n/a (crashes before completing) | n/a |
| `max_seq_len=4096` (config edited, no code change) | Success, reproduced 2x | 9939 MiB (both runs) | 9.80 GiB / 9.80 GiB (`max_memory_reserved()`, both runs) | 23.75s / 23.67s (143 tokens, ~6.0 tok/s both runs) |

Exact OOM error (verbatim, reproduced twice, identical both times):
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total capacity of 15.56 GiB of which 342.62 MiB is free. Including non-PyTorch memory, this process has 15.22 GiB memory in use. Of the allocated memory 15.08 GiB is allocated by PyTorch, and 19.46 MiB is reserved by PyTorch but unallocated.
```

Root cause (confirmed by reading `fish_speech/models/text2semantic/inference.py` and
`fish_speech/models/text2semantic/llama.py`, not guessed): `main()` and `generate()` both call

```python
model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, dtype=...)
```

unconditionally, right after the model loads and before any text is even tokenized.
`model.config.max_seq_len` comes straight from `config.json`'s `text_config.max_seq_len`
(**32768** for `fishaudio/s2-pro`), not from `--max-new-tokens` or input length. `KVCache`
(`llama.py`) allocates `(max_batch_size, n_local_heads, max_seq_len, head_dim)` tensors for both
K and V, for every one of the 36 text-decoder layers. At `n_local_heads=8`, `head_dim=128`,
bf16 (2 bytes): `36 layers × 2(K+V) × 8 × 32768 × 128 × 2 bytes ≈ 4.5 GB` — spent before a single
token is generated, and identical whether your prompt is 5 tokens or 5000.

Combined with the ~8.5GB of bf16 weights (safetensors, confirmed via HF API blob sizes) and
prefill activation memory, this leaves no room on a 16GB card (T4 reports ~15.56GiB usable via
`nvidia-smi`, not the full 16384MiB nameplate).

## T4 (Turing, sm_75) acceleration checklist — measured, not assumed

| Item | Finding |
|---|---|
| **dtype** | Checkpoint ships bf16 (`config.json`: `"dtype": "bfloat16"`); CLI defaults to bf16 (`precision = torch.half if half else torch.bfloat16` in `inference.py`). `--half` (fp16) is offered ("For GPUs that do not support bf16, you may need to use the `--half` parameter"). Tested both explicitly on T4 (same `max_seq_len=4096` config, same input, same seed): **`--half` does not crash** (unlike some other TTS repos we've tested, where forcing fp16 on a bf16-trained checkpoint produced NaN/Inf) and produces valid output (158 tokens vs bf16's 143 — different token count is expected, since dtype changes the numerics that feed sampling even with a fixed seed). Peak VRAM was essentially identical either way (10.26GB vs 10.28GB `max_memory_reserved` — expected, both are 2 bytes/param, so `--half` **cannot fix the OOM** above). Speed was also essentially identical (6.07 tok/s fp16 vs 6.02-6.04 tok/s bf16, "Bandwidth achieved" logged ~27.5-27.7 GB/s in both cases) — i.e. **no Tensor Core advantage materializes for either dtype here**, because the autoregressive decode loop (`decode_n_tokens`, one token at a time via `SDPBackend.MATH`) is memory-bandwidth-bound, not compute-bound, so Turing's missing bf16 Tensor Core path is irrelevant in practice for this workload. **Conclusion: on T4, `--half` is safe to use but is not a meaningful lever for speed or memory — don't bother with it.** |
| **attention backend** | Confirmed by reading `llama.py`: the main text decoder's `Attention.forward()` has a branch that forces `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` when `mask is None` — which **would** crash on sm_75 (flash-attn requires sm80+). However, tracing the actual call path shows `forward_generate()` (used by the CLI's inference loop) *always* builds an explicit causal mask from a precomputed buffer before calling attention, so `mask` is never `None` in practice — this branch is dead code on the inference path, not a live risk. Separately, `decode_n_tokens()` wraps every decode step in `with sdpa_kernel(SDPBackend.MATH):`, so the *actual* backend used during generation is deliberately forced to MATH (not flash, not mem-efficient, not auto-selected) — this is already portable/T4-safe by the repo's own design; no fix needed. |
| **int8 / quantization** | `tools/llama/quantize.py` (`WeightOnlyInt8QuantHandler`/`WeightOnlyInt4QuantHandler`) exists but is only invoked from `llama.py`'s `from_pretrained` when the **checkpoint directory path string contains the literal substring `"int8"` or `"int4"`** — there's no CLI flag for it, and the standard `fishaudio/s2-pro` checkpoint path doesn't match. It's also written for the older single-file checkpoint format, not the current sharded-safetensors `DualARTransformer` used by S2-Pro. Even if wired up, weight-only quantization would not touch the ~4.5GB KV-cache allocation above, which is the dominant OOM cause — so it would not by itself fix 16GB-card OOM. (Community reports in [#1168](https://github.com/fishaudio/fish-speech/issues/1168) corroborate this: a third-party w4a16-quantized checkpoint still used 21GB+.) |
| **CUDA Graphs** | `grep -rn "torch\.cuda\.graph\|CUDAGraph\|make_graphed_callables"` → 0 matches anywhere in the repo. Not used. |
| **torch.compile** | Present (`--compile` flag) but out of scope for this note. |

## Other things worth knowing

- **Seed**: unlike some peer TTS repos, this CLI does fix `torch.manual_seed(seed)` /
  `torch.cuda.manual_seed(seed)` with `--seed 42` as default — two runs of the exact same command
  produced the exact same token count (143) and near-identical timing, so results here are
  reproducible run-to-run.
- **Install extras vs plain pip**: `docs/en/install.md`'s Conda section suggests
  `pip install -e .[cu129]` (or `cu126`/`cu128`). The CUDA-index redirection
  (`[tool.uv.sources]`/`[tool.uv.index]` in `pyproject.toml`) is **UV-specific** — plain `pip`
  does not read it, so `pip install -e .[cu126]` silently resolves `torch==2.8.0` from the
  default PyPI index instead of the `cu126` wheel index. On this box that happened to still work
  (PyPI's default torch 2.8.0 wheel is a `cu128` build, and the driver's forward compatibility
  covers it — confirmed via `torch.cuda.is_available()` → `True`), but on a host that actually
  needs an older CUDA wheel, `pip install -e .[cuXXX]` silently gives you the wrong build.
- **UV extras are not "sticky" across invocations — this is the bigger trap.** `uv sync --extra
  cu126` does correctly resolve `torch==2.8.0+cu126` (verified: `2.8.0 12.6 True`). But the base
  `dependencies` list in `pyproject.toml` just pins bare `torch==2.8.0` with no CUDA index — the
  `[tool.uv.sources]` redirection only applies while a matching extra (`cpu`/`cu126`/`cu128`/
  `cu129`) is active *on that specific command*. Running a later `uv run python ...` **without**
  repeating `--extra cu126` triggers an implicit re-sync that drops back to a different default
  build — reproduced directly: `uv sync --extra cu126` → `2.8.0+cu126`; plain `uv run python3 -c
  "import torch; print(torch.__version__)"` right after → silently switched to `2.8.0+cu128`;
  `uv run --extra cu126 python3 ...` → back to `2.8.0+cu126`. **You must pass `--extra cuXXX` on
  every `uv run`/`uv sync` invocation for the life of the project, not just the first setup step**
  — a very easy thing to forget once install is "done."
