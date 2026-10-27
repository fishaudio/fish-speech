# DGX Spark performance and optimization report

**Date:** 2026-10-26  
**Hardware:** NVIDIA DGX Spark, GB10, 121 GiB unified memory  
**Repository:** `dgxspark.fish-speech` fork of `fishaudio/fish-speech`  
**Scope:** native Fish Speech 3.0 S2-Pro and Drama 3 on the Spark, including
the API, WebUI, MCP server, streaming and cold-load behavior.

## Executive summary

The source branch reports end-to-end runs of both models on one Spark. These
hardware results have not been independently reproduced during this review.

**Measurement caveat:** historical `first_byte_s` values below measure completion
of a buffered 64 KiB read, not arrival of the legacy buffered-read latency sample. Streaming cases
have only one measured run each. Benchmark version 2 records `first_audio_s` using
unbuffered-response reads and WAV sample detection; rerun at least three times.
The archived JSON files are marked `measurement_version: 1` and `legacy_measurement: true`; they retain their original values and measurement semantics.

- **S2-Pro** runs at **RTF ≈ 1.56** for medium/long text with `--compile`,
  compared with RTF ≈ 2.25 eager. It uses about 21 GiB and cold-starts in
  roughly 2–2.5 minutes.
- **Drama 3** runs at **RTF 1.16–1.22** for medium/long reference-conditioned
  speech, uses **41.3 GiB** peak memory, and now cold-starts in **67–70 s**.
  Its measured generation times are lower in these runs; perceptual quality was not evaluated.
- Drama 3 without a reference voice is **unreliable**: 5 of 8 tested seeds
  failed to emit EOS and ran to the token cap, appending near-silence. With a
  reference, all four tested seeds stopped naturally at 5.6–5.8 s. Use a
  reference for production.
- The Drama 3 checkpoint loader was the startup bottleneck. Replacing 62,053
  per-expert copies with per-layer bulk FP8 transfers cut cold load from about
  **5.5 minutes to 41 s for the backbone** (about 67–70 s total startup).

Two environment blockers were fixed earlier in the fork:

1. The Spark environment pinned Protobuf 5, while TensorBoard’s generated code
   required Protobuf 6.31+. The API failed at import until the Spark override
   moved to `protobuf>=6.31.1,<7.0.0` (resolved to 6.33.6).
2. PyTorch 2.10’s CUDA 13 wheel supports through SM 12.0, while the GB10 reports
   SM 12.1. The tested BF16 kernels, SDPA, DAC codec, FP8 MoE path and
   end-to-end generation work through `compute_120` PTX, but the warning and
   compatibility risk remain.

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 (SM 121) |
| Unified memory | 121 GiB |
| Driver | 580.173.02 |
| CUDA runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu130 |
| TorchAudio / TorchCodec | 2.10.0 / 0.10.0 |
| Triton | 3.6.0 |
| Protobuf | 6.33.6 |
| S2-Pro checkpoint | HF cache `fishaudio/s2-pro` |
| Drama 3 checkpoint | `fishaudio/drama-3` HF snapshot, 41.5 GB |

The upstream root project still pins PyTorch 2.8. The Spark environment is
isolated in `environments/dgx-spark` with its own lockfile.

## What changed

### 1. Protobuf 6 and configurable Inductor mode

The inherited `protobuf<6` override broke API import. The Spark override is now
`protobuf>=6.31.1,<7.0.0`, resolving to 6.33.6.

`--compile` honors `FISH_COMPILE_MODE` (`default`) and
`FISH_COMPILE_DYNAMIC` (`1`). `start.sh` enables compilation for S2-Pro only;
Drama 3 rejects `--compile`.

### 2. One-command launcher, WebUI and MCP

`start.sh` creates the locked environment, builds the WebUI once and starts
the API, whose resolver downloads missing files into the shared HF cache. S2-Pro remains the default. Select Drama 3 explicitly with
`./start.sh --model fishaudio/drama-3` and supply a reference voice. `start-mcp.sh` runs the MCP server in
its own small environment so MCP dependencies cannot alter the CUDA/Pydantic
stack.

### 3. S2-Pro streaming chunking

`chunk_length` was ignored for untagged text, so one audio segment was emitted
only after the whole request had been generated. `split_s2_text()` now applies
the splitter to untagged text and preserves the tagged-speaker path.

### 4. Drama 3 batched FP8 loader

The checkpoint stores experts per expert. The original loader copied 62,053
tensors individually from the checkpoint into the stacked GPU parameters. The
new loader accumulates one layer/projection on CPU, stacks it once, validates
every slot, and performs one bulk transfer. It reduced backbone load from about
5.5 minutes to 41 s while preserving the existing FP8/dense loader tests.

### 5. Drama 3 `chunk_length`

`generate_long_drama3()` now passes `chunk_length` to `split_drama3_text()`.
Previously Drama 3 always used its internal 300-byte default, so requests with a
smaller chunk length were ignored. This is what makes sub-2-second legacy buffered-read latency
possible for long text.

## Results

### S2-Pro

| Case | Eager wall / RTF | Compiled wall / RTF |
| --- | ---: | ---: |
| 12 chars, 0.98 s | 2.37 s / 2.43 | 1.79 s / 1.84 |
| 99 chars, 6.36 s | 14.30 s / 2.25 | 9.72 s / 1.57 |
| 250 chars, 16.07 s | 36.72 s / 2.29 | 24.24 s / 1.56 |

| Metric | Eager | Compiled |
| --- | ---: | ---: |
| Decode throughput | ~7.6 tokens/s | ~14.5 tokens/s |
| Reported effective bandwidth | ~34.5 GB/s | ~66 GB/s |
| Peak GPU memory | ~20.3 GiB | ~21.1 GiB |
| Startup to ready | ~99 s | ~145–160 s |
| GPU utilization / power | ~91% / 26 W | ~94% / 39 W |

S2-Pro streaming, 250-character case:

| Configuration | Legacy buffered-read latency | Total | Audio | RTF |
| --- | ---: | ---: | ---: | ---: |
| Before chunking fix, `chunk_length=200` | 21.5 s | 21.7 s | ~14.4 s | ~1.51 |
| Fixed, `chunk_length=200` | 10.0 s | 24.3 s | 15.51 s | 1.56 |
| Fixed, `chunk_length=100` | **1.43 s** | 23.0 s | 14.44 s | 1.60 |

`chunk_length=100` is the interactive setting. Raw samples are in
`benchmarks/s2pro-compiled.json`, `benchmarks/s2pro-eager.json`,
`benchmarks/s2pro-streaming-chunk100.json` and
`benchmarks/s2pro-streaming-chunk200.json`.

### Drama 3

Cold load and footprint:

| Metric | Before loader change | After loader change |
| --- | ---: | ---: |
| Backbone weights | ~5.5 min | **41 s** |
| Total startup to ready | ~7 min | **67–70 s** |
| Peak GPU memory | — | **41.3 GiB** |
| Parameter slots loaded | 62,053 | 62,053 |

The tower adds about 13 s and the codec about 3 s. CUDA-graph capture takes
about 1.5 s.

Warm reference-conditioned performance (S2-generated 6.1 s reference
`drama-ref`, seed 42, three runs):

| Case | Audio | Wall | RTF |
| --- | ---: | ---: | ---: |
| 12 chars | 0.98 s | 1.66 s | 1.70 |
| 99 chars | 5.62 s | 6.84 s | 1.22 |
| 250 chars | 13.84 s | 16.05 s | 1.16 |

Drama 3 streaming, 250-character case:

| Chunk size | Legacy buffered-read latency | Total | Audio | RTF |
| --- | ---: | ---: | ---: | ---: |
| `chunk_length=200` | 7.02 s | 16.39 s | 13.61 s | 1.20 |
| `chunk_length=100` | **1.35 s** | 17.15 s | 12.91 s | 1.33 |

The smaller chunk gives sub-1.5-second legacy buffered-read latency at a small total-time cost.
Raw samples are in `benchmarks/drama3-reference.json`,
`benchmarks/drama3-streaming-chunk100.json` and
`benchmarks/drama3-streaming-chunk200.json`.

### Drama 3 EOS reliability

Same 99-character text, `max_new_tokens=400`:

| Condition | Seeds tested | Natural EOS | Hit cap / trailing silence |
| --- | ---: | ---: | ---: |
| No reference | 8 | 3 | **5** |
| With reference | 4 | **4** | 0 |

Natural reference-conditioned clips were 5.6–5.8 s. Capped no-reference clips
were 18.58 s (or 27.86 s at `max_new_tokens=600`) with a final-2-second RMS
around `0.0002`, i.e. near-silence. The model is trained for reference
conditioning; the no-reference path should not be used as a production default.

## Analysis

S2-Pro is a 4B dense model with a 10-codebook fast-AR loop. At batch 1, weight
traffic and kernel-launch overhead dominate; Inductor reduces the latter by
about 32%. It remains above real time on GB10, and further gains require
lower-precision S2 weights or a different serving engine.

Drama 3 is a 35B-A3B block-FP8 hybrid-attention model. The FP8 expert path keeps
the checkpoint at 41.5 GB and the resident footprint at 41.3 GiB, and the
CUDA-graph decode reaches RTF 1.16 for long reference-conditioned text. That is
faster than compiled S2-Pro on the same Spark, without establishing a perceptual quality advantage.

The largest avoidable cost was cold load, not decode. Per-expert copies were
latency-bound on the CPU/GPU submission path, not disk-bound. Bulk stacking
turned 62k transfers into 80 weight transfers plus their scales. The tower remains a
smaller 13 s load cost.

Drama 3’s no-reference EOS failure is a model/sampling behavior, not an
attention-kernel issue: outputs contain real speech followed by near-silence
until the cap. A reference prompt makes EOS reliable across the tested seeds.
We keep the reference path as the recommended production mode.

## Recommendations

- Evaluate **Drama 3 with a reference voice**; the small sample suggests better EOS behavior.
  Expect RTF 1.16–1.22 on medium/long text.
- For interactive legacy buffered-read latency, set `chunk_length=100`; legacy buffered-read latency is 1.35 s
  versus 7.02 s at the default 200, with about 5% more total time.
- Use `FISH_MODEL=fishaudio/s2-pro ./start.sh` when memory or a smaller model matters. Expect RTF ≈ 1.56 with `--compile`.
- Keep one model server resident at a time. Drama 3 leaves about 80 GiB of the
  121 GiB pool for the OS, other services and CUDA contexts.
- Use `FISH_DRAMA3_CUDA_GRAPH=0` only to debug graph capture. Use
  `FISH_DRAMA3_FP8_EXPERTS=0` only as a dequantized fallback; it needs much more
  memory.
- Use the MCP server for agent integration; S2 and Drama 3 both use the same
  six-tool surface.

## Limitations

- S2-Pro numbers are single-request, batch-1, no reference audio; they are not a
  throughput/concurrency benchmark.
- Drama 3 numbers use one S2-generated reference voice and one language.
- No formal perceptual quality evaluation was performed. WAVs were checked for
  format, duration, sample rate, channels and successful decode; listening tests
  remain for the user.
- The no-reference Drama 3 path can run to the token cap and append silence. It
  is documented, not recommended.
- Smaller chunk sizes can change prosody at boundaries.
- The SM 12.1 / PyTorch 12.0 warning remains.
- `nvidia-smi` reports no total memory for GB10; GPU memory figures come from
  PyTorch’s allocator and per-process accounting.

## Reproduce

```bash
cd /home/spark/opt/TTS/dgxspark.fish-speech

# S2-Pro
FISH_MODEL=fishaudio/s2-pro ./start.sh
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 --runs 3 \
    --output benchmarks/s2pro-compiled.json

# Drama 3 with a reference voice registered as "narrator"
FISH_MODEL=fishaudio/drama-3 ./start.sh
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 \
    --voice narrator --runs 3 --output benchmarks/drama3-reference.json

# Streaming, interactive chunk size
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 \
    --voice narrator --streaming --chunk-length 100 --runs 3 \
    --output benchmarks/drama3-streaming-chunk100.json
```

## Deliverables

| File | Purpose |
| --- | --- |
| `start.sh` | One-command environment/model/WebUI/API launcher (S2-Pro default). |
| `start-mcp.sh` | Starts the MCP server in its own environment. |
| `tools/mcp_server.py` | Six MCP tools over the HTTP API. |
| `tools/benchmark_tts.py` | Reproducible non-streaming/streaming benchmark. |
| `fish_speech/.../backbone.py` | Batched FP8 expert loader. |
| `fish_speech/.../inference.py` | S2/Drama 3 streaming chunking. |
| `docs/en/dgx-spark.md` | Operating guide. |
| `docs/en/dgx-spark-report.md` | This report. |
| `docs/en/mcp.md` | MCP setup and tool reference. |
| `benchmarks/s2pro-*.json` | S2 samples. |
| `benchmarks/drama3-reference.json` | Drama 3 reference-conditioned samples. |
| `benchmarks/drama3-no-reference.json` | No-reference runaway sample for comparison. |
| `benchmarks/drama3-streaming-chunk100.json` | Drama 3 interactive streaming. |
| `benchmarks/drama3-streaming-chunk200.json` | Drama 3 default streaming. |
| `tests/test_fp8_loader_order.py` | Shuffled/interleaved FP8 loader bookkeeping. |
| `tests/test_benchmark_tts.py` | First-audio WAV reader regression test. |
| `tests/test_s2_chunking.py` | S2 streaming regression test. |

## Integration review corrections

- Batched FP8 loading preserves pending down projections when gate/up complete.
  Incoming slots are validated immediately, including duplicates and expert IDs.
- The branch is based on current main, retaining HF cache selection, offline and
  revision handling, codec fallback, and MCP format forwarding.
- Drama 3 chunk length is now available in the WebUI.
- S2-Pro remains the default; model changes require explicit selection.
- RTF above 1 means generation is slower than real time. Cross-model comparisons
  here use different reference conditioning and are not controlled quality tests.
- New benchmark output includes request settings and measurement version 2.
  Cold-load figures are reported observations, not independently reproduced logs.
