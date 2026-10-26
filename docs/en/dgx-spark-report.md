# DGX Spark performance and optimization report

**Date:** 2026-10-26  
**Hardware:** NVIDIA DGX Spark, GB10, 121 GiB unified memory  
**Repository:** `dgxspark.fish-speech` fork of `fishaudio/fish-speech`  
**Scope:** native Fish Speech 3.0 on the Spark, S2-Pro validated; Drama 3
deferred because its checkpoint could not be transferred in a reasonable time.

## Executive summary

The fork now starts S2-Pro with one command, serves the API and WebUI, and
exposes an MCP server. Two real blockers were found and fixed:

1. The Spark environment pinned Protobuf 5, while TensorBoard’s generated code
   required Protobuf 6.31+. The API failed at import until the Spark override
   was moved to `protobuf>=6.31.1,<7.0.0` (resolved to 6.33.6).
2. PyTorch 2.10’s CUDA 13 wheel supports through SM 12.0, while the GB10 reports
   SM 12.1. The tested BF16 kernels, SDPA, DAC codec, and end-to-end generation
   work through `compute_120` PTX, but the warning and compatibility risk remain.

On S2-Pro, enabling `torch.compile` improves steady-state decode from about
**7.6 to 14.5 tokens/s** and cuts medium/long-text latency from **RTF ≈ 2.3 to
RTF ≈ 1.56**. Peak GPU memory is about 20–21 GiB. The GPU is 91–94%
utilized but draws only 26–39 W, which is consistent with a launch-bound,
small-batch decode rather than a bandwidth-saturated workload.

Streaming had a separate bug: `chunk_length` was ignored for untagged text, so
the API generated the whole request before emitting one audio segment. After the
fix, first audio on the 250-character test drops from **21.5 s to 10.0 s** at
`chunk_length=200`, and to **1.43 s** at `chunk_length=100`, with total RTF
essentially unchanged.

S2-Pro is **not faster than real time** on this GB10 in the tested configuration.
Reaching RTF below 1 will require lower-precision weights, a more specialized
serving engine, batching, or a combination of those. This report separates what
was measured from what remains speculative.

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA GB10 |
| Compute capability | 12.1 |
| Unified memory | 121 GiB |
| Driver | 580.173.02 |
| CUDA runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu130 |
| TorchAudio | 2.10.0 |
| TorchCodec | 0.10.0 |
| Triton | 3.6.0 |
| Protobuf | 6.33.6 |
| OS | Linux 6.17.0-1029-nvidia, aarch64 |
| Network used for weights | 2.4 GHz Wi-Fi, ~1.5–4 MB/s observed |

The upstream root project still pins PyTorch 2.8. The Spark environment is
isolated in `environments/dgx-spark` with its own lockfile; the root project’s
pins and indexes are unchanged.

## What changed

### 1. Protobuf 6 for the Spark environment

The API imports `audiotools`, which imports `torch.utils.tensorboard`, whose
generated `event_pb2` code requires Protobuf 6.31 or newer. The inherited
`protobuf<6` override produced:

```text
google.protobuf.runtime_version.VersionError: Detected incompatible
Protobuf Gencode/Runtime versions ... gencode 6.31.1 runtime 5.29.6
```

The Spark override is now:

```toml
protobuf>=6.31.1,<7.0.0
```

The lock resolves 6.33.6, and `import tools.api_server` succeeds.

### 2. Configurable Inductor mode

`--compile` now honors two environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `FISH_COMPILE_MODE` | `default` | `torch.compile(mode=...)` |
| `FISH_COMPILE_DYNAMIC` | `1` | `torch.compile(dynamic=...)` |

`start.sh` enables `--compile` for S2-Pro by default and never for Drama 3,
which rejects it. `FISH_COMPILE=0` restores eager mode.

`reduce-overhead` (Inductor CUDA graphs) was tested but did not finish compiling
within five minutes; it was abandoned rather than made a fragile default.

### 3. One-command launcher and local checkpoint preference

`start.sh` creates the locked environment if needed, downloads the selected
model into the shared HF cache, builds the WebUI once, and starts the API.
If `checkpoints/s2-pro` already contains weights, it uses that local checkpoint
instead of re-downloading the same model into the HF cache.

`start-mcp.sh` creates a separate, small MCP environment. MCP was deliberately
not installed into the CUDA environment: its dependency set upgraded Pydantic
from the fork’s pinned 2.9.2 to 2.14.1. The separate environment keeps the
inference stack on its lockfile.

### 4. MCP server

`tools/mcp_server.py` exposes six tools: health, defaults, voice list/add/delete,
and TTS. It is a thin HTTP client with no Torch import, so it can run on a
laptop. Generated files are written on the MCP host, not necessarily the API
host. Nine unit tests pass without loading a model. A live MCP check registered a
reference voice, synthesized with an inline reference clip, listed the voice, and
deleted it successfully.

### 5. Streaming chunking fix

`chunk_length` was effectively ignored for text without `<|speaker:N|>` tags:
`split_text_by_speaker()` returned an empty turn list, and `generate_long()`
fell back to `[text]`. The API therefore generated the entire request before
emitting one audio segment. The WAV header arrived immediately, but first audio
for the 250-character test took 21.5 s.

`split_s2_text()` now applies `split_by_length()` to untagged text, honoring
`chunk_length` and preferring sentence boundaries. Tagged text keeps the old
speaker-turn grouping. Three unit tests cover short text, long untagged text,
and tagged turns.

## Methodology

The benchmark used `tools/benchmark_tts.py` against a live API. Each request
used `seed=42`, `format=wav`, `normalize=true`, and no reference audio. Audio
duration was read from the returned WAV header; for streamed WAVs it was
computed from the received data bytes because the header has no final size. RTF
is wall time divided by audio duration, so lower is better and RTF < 1 is faster
than real time.

| Case | Text length | Audio duration |
| --- | ---: | ---: |
| short | 12 chars | ~1.0 s |
| medium | 99 chars | ~6.4 s |
| long | 250 chars | ~16.1 s |

Eager numbers are single runs on an otherwise idle GPU. Compiled numbers are
medians of three runs after one warmup; the raw samples are in
`benchmarks/s2pro-compiled.json`.

## Results

### S2-Pro, eager vs compiled

| Case | Eager wall | Eager RTF | Compiled wall | Compiled RTF |
| --- | ---: | ---: | ---: | ---: |
| short | 2.37 s | 2.43 | 1.79 s | 1.84 |
| medium | 14.30 s | 2.25 | 9.72 s | 1.57 |
| long | 36.72 s | 2.29 | 24.24 s | 1.56 |

| Metric | Eager | Compiled |
| --- | ---: | ---: |
| Decode throughput | ~7.6 tokens/s | ~14.5 tokens/s |
| Effective bandwidth reported by inference loop | ~34.5 GB/s | ~66 GB/s |
| Peak GPU memory | ~20.3 GiB | ~21.1 GiB |
| Startup to ready | ~99 s | ~145–160 s |
| GPU utilization | ~91% | ~94% |
| Power draw | ~26 W | ~39 W |
| Temperature | ~49 °C | ~57 °C |

The compiled path is about **34% faster** on the long case in steady state. The shortest case
benefits less because fixed request overhead is a larger fraction of its wall
time.

### End-to-end launcher check

`start.sh` was verified with S2-Pro on a local checkpoint:

- `/v1/health` returned HTTP 200,
- `/ui` returned HTTP 200 and 342,785 bytes of built HTML,
- `POST /v1/tts` returned HTTP 200, a valid 1.35 s, 44.1 kHz mono WAV,
- the log showed the expected local checkpoint and codec paths.

### Streaming time-to-first-audio

The 250-character long case, streaming WAV, seed 42. “First audio” is the time
until the first non-header audio bytes arrived; the WAV header itself arrives in
about 2–20 ms.

| Configuration | First audio | Total | Audio | RTF |
| --- | ---: | ---: | ---: | ---: |
| Before chunking fix, `chunk_length=200` | 21.5 s | 21.7 s | ~14.4 s | ~1.51 |
| Fixed, `chunk_length=200` | 10.0 s | 24.3 s | 15.51 s | 1.56 |
| Fixed, `chunk_length=100` | **1.43 s** | 23.0 s | 14.44 s | 1.60 |

`chunk_length=100` is the useful interactive setting: it gives sub-2-second
first audio at essentially the same total RTF. The default remains 200 to avoid
unnecessary chunk boundaries. Smaller chunks are a latency/quality tradeoff;
check prosody at boundaries for the voice you use. Raw samples are in
`benchmarks/s2pro-streaming-chunk100.json` and
`benchmarks/s2pro-streaming-chunk200.json`.

## Analysis

The GB10 has roughly 273 GB/s of memory bandwidth. S2-Pro’s decode path reads
the main 4B model once per audio frame and then runs a 4-layer fast model ten
times, once per codebook. That is a lot of weight traffic for a batch of one.
At ~14.5 frames/s the main model alone accounts for on the order of 100 GB/s;
the loop’s reported 66 GB/s is plausible once cache reuse and the fast model are
accounted for.

The GPU is 94% utilized but at low power. That pattern fits many small kernels
and a serial codebook loop, not a saturated matrix engine. `torch.compile`
reduces kernel launches and fuses pointwise work, which explains the 32% gain.
The failed `reduce-overhead` experiment was aimed at the same overhead via CUDA
graphs, but its compile time was unacceptable for a default.

Simple parameter changes (BF16 vs FP16, larger chunk length, different sampling
defaults) do not change the fundamental per-frame weight traffic. The credible
next steps are:

1. **Lower-precision S2-Pro weights.** FP8 W8A8 could roughly halve weight
   traffic. The fork already has block-FP8 MoE kernels for Drama 3, but applying
   them to S2-Pro’s dense layers is a separate change with a quality evaluation.
2. **A specialized serving engine.** Upstream recommends SGLang-Omni or
   vLLM-Omni for efficiency. Their ARM64/SM 12.1 support and Fish Speech model
   integration were not validated in this pass; adopting one would replace much
   of the native generation path.
3. **Batching.** Serving multiple requests together amortizes weight reads, but
   it trades away single-request latency and needs a queue/scheduler.

None of these was implemented, so the measured 1.56 RTF is the honest native
result, not a claim about what a different engine could achieve.

## Drama 3 status

Drama 3 is implemented in the fork: block-FP8 experts, a 40-layer hybrid
attention backbone, a 4-layer codebook tower, and CUDA-graph decode. It was not
validated on this machine.

The checkpoint is about **41.5 GB**: eight safetensors shards of roughly 5 GB
each plus a 1.87 GB codec. It was not present in the HF cache. Only S2-Pro’s
weights were available locally under `checkpoints/s2-pro`. A download over the
available 2.4 GHz Wi-Fi stalled repeatedly at roughly 1.5–4 MB/s, so the
checkpoint was left partially downloaded and the benchmark was deferred rather
than reported from an unvalidated or incomplete model.

To finish it:

```bash
uv run --project environments/dgx-spark --locked --extra cu130 \
    hf download fishaudio/drama-3

# Stop other model servers first; Drama 3 needs most of the unified memory.
FISH_MODEL=fishaudio/drama-3 FISH_LISTEN=127.0.0.1:8080 ./start.sh

python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 \
    --runs 1 --warmup 1 --output benchmarks/drama3.json
```

Watch for `FishOmni experts: block-FP8 W8A8`, backbone placement, and CUDA-graph
capture in the startup log. If graph capture fails, retry with
`FISH_DRAMA3_CUDA_GRAPH=0` to isolate graph capture from kernel issues. If FP8
kernels fail on SM 12.1, `FISH_DRAMA3_FP8_EXPERTS=0` is the dequantized fallback,
but it needs substantially more memory.

A complete Drama 3 report should add: startup time, peak memory, frames/s,
time-to-first-segment for streaming, RTF for short/medium/long text, and a
quality comparison against S2-Pro on the same reference voice.

## Deliverables

| File | Purpose |
| --- | --- |
| `start.sh` | One-command environment/model/WebUI/API launcher. |
| `start-mcp.sh` | Starts the MCP server in its own environment. |
| `tools/mcp_server.py` | Six MCP tools over the HTTP API. |
| `tools/benchmark_tts.py` | Reproducible non-streaming/streaming benchmark. |
| `environments/mcp/` | Locked MCP client environment. |
| `environments/dgx-spark/` | CUDA 13 environment, Protobuf 6 fix, lockfile. |
| `docs/en/mcp.md` | MCP setup, client config, tool reference. |
| `docs/en/dgx-spark.md` | Spark installation and operating guide. |
| `docs/en/dgx-spark-report.md` | This report. |
| `benchmarks/s2pro-compiled.json` | Raw compiled benchmark samples. |
| `benchmarks/s2pro-eager.json` | Raw eager benchmark samples. |
| `benchmarks/s2pro-streaming-chunk100.json` | Streaming, interactive chunk size. |
| `benchmarks/s2pro-streaming-chunk200.json` | Streaming, default chunk size. |
| `tests/test_s2_chunking.py` | Regression tests for the streaming chunking fix. |

## Recommendations

- Use S2-Pro with the default compiled launcher for the best quality/latency
  balance currently validated on this Spark. Expect RTF ≈ 1.56 for medium/long
  text, not sub-real-time.
- For interactive first audio, set `chunk_length=100`; the 250-character test
  dropped from 10.0 s to 1.43 s with essentially unchanged total RTF.
- Keep one model server per Spark. Loading S2-Pro and Drama 3 together is
  possible in 121 GiB but leaves little headroom and slows both.
- Use wired networking for the 41.5 GB Drama 3 checkpoint. The Wi-Fi path is
  the practical blocker, not the model code.
- If sub-real-time S2-Pro is a requirement, evaluate FP8 S2-Pro or a serving
  engine as a separate, measured project. Do not assume the H200 RTF figures
  transfer to GB10.
- Use the MCP server for agent integrations and the existing HTTP API for
  applications that need streaming or direct byte handling.

## Limitations

- S2-Pro numbers are single-request, batch-1, no reference audio, and one voice;
  they are not a throughput or concurrency benchmark.
- Eager numbers are single runs; compiled numbers are three-run medians. The
  eager sample is therefore less stable, especially on the short case.
- No formal perceptual quality evaluation was performed. The generated WAVs
  were checked for format, duration, and validity; listening tests remain for
  the user. Smaller chunk sizes can change prosody at boundaries.
- Drama 3 has no measured numbers here.
- `nvidia-smi` reports no memory total for GB10; memory figures come from
  PyTorch’s allocator.
- The SM 12.1 / PyTorch 12.0 warning remains. The tested path works, but a
  different kernel could hit a missing architecture.
