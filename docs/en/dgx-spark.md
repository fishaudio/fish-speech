# DGX Spark setup

This fork runs Fish Speech 3.0 on the NVIDIA DGX Spark (GB10). The validated
path is **S2-Pro**; Drama 3 is implemented but was not benchmarked here because
its checkpoint was not available locally. See [Drama 3](#drama-3) for the exact
resume and validation steps.

Validated on 2026-10-26 with:

| Component | Version |
| --- | --- |
| GPU | NVIDIA GB10, compute capability 12.1, 121 GiB unified memory |
| Driver | 580.173.02 |
| CUDA runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu130 |
| TorchAudio / TorchCodec | 2.10.0 / 0.10.0 |
| Triton | 3.6.0 |
| Protobuf | 6.33.6 |

The upstream root project still pins PyTorch 2.8. The Spark environment is a
separate uv project, `environments/dgx-spark`, with its own lockfile. This keeps
upstream installs reproducible while allowing CUDA 13 ARM64 wheels here.

## Quick start

```bash
cd /home/spark/opt/TTS/dgxspark.fish-speech
./start.sh
```

On the first run `start.sh`:

1. creates `environments/dgx-spark/.venv` with the locked CUDA 13 environment,
2. builds the Awesome WebUI once, and
3. starts the API on `0.0.0.0:8080`, resolving `fishaudio/s2-pro` through the shared
   Hugging Face cache and enabling S2-Pro `--compile`.

Open `http://<SPARK_IP>:8080/ui`. The API is at
`http://<SPARK_IP>:8080/v1/`. Startup takes about 2–2.5 minutes because the
model is loaded and the decode function is compiled; later requests reuse the
Inductor cache in `~/.cache/fish-speech/inductor`.

For the MCP server, see [MCP server](mcp.md):

```bash
./start.sh mcp          # same as ./start-mcp.sh
```

### Useful environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FISH_MODEL` | `fishaudio/s2-pro` | `fishaudio/s2-pro`, `fishaudio/drama-3`, or a local path. |
| `FISH_COMPILE` | `1` for S2-Pro | Set `0` for eager mode (slower steady state, faster startup). |
| `FISH_COMPILE_MODE` | `default` | Passed to `torch.compile`; `reduce-overhead` was not stable here. |
| `FISH_DEVICE` | unset | Example: `cuda:0`; passed as `--device`. |
| `FISH_LISTEN` | `0.0.0.0:8080` | Example: `127.0.0.1:8080` for loopback only. |
| `FISH_OFFLINE` | `0` | Set `1` to pass `--offline` and never contact the Hub. |
| `FISH_SKIP_WEBUI` | `0` | Set `1` to skip the one-time WebUI build. |

Examples:

```bash
# Eager S2-Pro, loopback only
FISH_COMPILE=0 FISH_LISTEN=127.0.0.1:8080 ./start.sh

# Select Drama 3 without automatic compilation
./start.sh --model fishaudio/drama-3

# Use an explicit local checkpoint; let the resolver find the codec
./start.sh --llama-checkpoint-path /data/checkpoints/my-s2
```

CLI model selection overrides `FISH_MODEL`. Existing `checkpoints/` files do not
override the default shared HF cache. Local checkpoints must be selected explicitly;
`FISH_MODEL=/data/checkpoints/my-s2` also works. The codec resolver checks the local
checkpoint, then a sibling `s2-pro/codec.pth`, then the HF cache. Use
`--decoder-checkpoint-path` to override it.

`--revision` and `--offline` are passed directly to the API resolver. There is no
separate startup download; `FISH_SKIP_DOWNLOAD` is no longer needed. To prevent
Hub requests, use `--offline` or `FISH_OFFLINE=1`.

The older `./start-api.sh` and `./start-webui.sh` launchers still work and accept
the same API flags. `start.sh` is the supported one-command path.

## Why PyTorch 2.10 and CUDA 13?

PyTorch 2.9 has CUDA 13 ARM64 wheels, but TorchAudio 2.9+ uses TorchCodec for
`torchaudio.load`, which Fish Speech uses for reference audio. The TorchCodec
0.8/0.9 wheels compatible with PyTorch 2.9 did not include Linux ARM64 in the
indexes checked by the fork. The PyTorch 2.10 / TorchCodec 0.10 pair provides
official ARM64 wheels without a source build.

The host must still have a working NVIDIA driver compatible with CUDA 13.
`nvidia-smi` reports the driver; `nvcc --version` reports the separately
installed toolkit. Neither proves that the Python environment has a CUDA build.
The diagnostic below checks the Python side.

## Manual install

```bash
uname -m                       # expected: aarch64
nvidia-smi                     # must detect the GB10
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev \
    libsox-dev libsndfile1 ffmpeg

uv sync --project environments/dgx-spark --locked --python 3.12 --extra cu130
```

Or run `./install-dgx-spark.sh`, which performs the same steps.

The environment lives in `environments/dgx-spark/.venv`. Keep both `--project
environments/dgx-spark` and `--extra cu130` on manual uv commands. An existing
root `.venv` is not reused.

### Protobuf 6 is required on the Spark

The upstream override caps Protobuf below 6, but the installed TensorBoard
(transitive through `descript-audiotools`) contains generated code that requires
Protobuf 6.31 or newer. With Protobuf 5 the API fails at import with
`Detected incompatible Protobuf Gencode/Runtime versions`. The Spark project
overrides this to `protobuf>=6.31.1,<7.0.0`; the root project is unchanged.
The lock currently resolves to 6.33.6.

### PyTorch and GB10 compute capability 12.1

PyTorch 2.10’s CUDA 13 wheel advertises support through SM 12.0. The GB10
reports SM 12.1, so PyTorch prints a warning. In practice BF16 matmul, SDPA, the
DAC codec, and end-to-end S2-Pro generation all run; the wheel contains
`compute_120` PTX, which the driver JITs for SM 12.1. Treat this as a
compatibility risk for less common kernels, not as a blocker for the tested
S2-Pro path.

## Verify CUDA and reference-audio decoding

```bash
uv run --project environments/dgx-spark --locked --extra cu130 python - <<'PY'
import io
import tempfile
import wave
from pathlib import Path

import torch
import torchaudio
import torchcodec

print("Torch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("TorchCodec:", torchcodec.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA unavailable: check driver and selected extra"
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Wheel architectures:", torch.cuda.get_arch_list())
assert torch.version.cuda == "13.0", "Expected a cu130 Torch build"

x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x.T).all().item()
q = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
).all().item()
torch.cuda.synchronize()
print("CUDA BF16 matmul and attention: OK")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "probe.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 1600)
    for source in (str(path), io.BytesIO(path.read_bytes())):
        audio, rate = torchaudio.load(source)
        assert rate == 16000 and tuple(audio.shape) == (1, 1600)
print("Reference audio decoding (file and bytes): OK")
PY
```

## Measured S2-Pro performance

Measured on the idle GB10 with `tools/benchmark_tts.py`, seed 42, no reference
audio, 44.1 kHz mono WAV, three runs per case. “RTF” is wall time divided by
generated audio duration; lower is better, and RTF < 1 is faster than real time.

| Case | Audio | Eager wall / RTF | Compiled wall / RTF |
| --- | ---: | ---: | ---: |
| short, 12 chars | 0.98 s | 2.37 s / 2.43 | 1.79 s / 1.84 |
| medium, 99 chars | 6.36 s | 14.30 s / 2.25 | 9.72 s / 1.57 |
| long, 250 chars | 16.07 s | 36.72 s / 2.29 | 24.24 s / 1.56 |

Decode throughput rises from about **7.6 to 14.5 tokens/s**; effective bandwidth
reported by the inference loop rises from about **34.5 to 66 GB/s**. Peak GPU
memory is about **20–21 GiB**. GPU utilization is 91–94%, power draw 26–39 W,
and temperature 49–57 °C. Compilation adds roughly 50 s to startup.

S2-Pro therefore runs at **RTF ≈ 1.56** on this GB10 in the tested
single-request configuration: about 1.5 seconds of compute per second of audio,
not faster than real time. See the [DGX Spark report](dgx-spark-report.md) for
the analysis and limitations.

### Interactive streaming

The streaming WAV header arrives in about 20 ms, but audio arrives per text
chunk. On the 250-character test, the chunking fix reduced first audio from
21.5 s to **10.0 s** at the default `chunk_length=200`; using
`chunk_length=100` reduces it to **1.43 s** with essentially the same total RTF
(1.60 vs 1.56). For interactive use, pass `chunk_length=100` and verify prosody
at chunk boundaries for your voice.

The benchmarks can be reproduced against a running server:

```bash
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 --runs 3 \
    --output benchmarks/s2pro-compiled.json
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 --streaming \
    --chunk-length 100 --runs 1 --output benchmarks/s2pro-streaming-chunk100.json
```

## Drama 3

Drama 3 is a 35B-A3B model with block-FP8 experts. The checkpoint is about
**41.5 GB** (eight ~5 GB safetensors shards plus the 1.87 GB codec) and fits in
the GB10’s 121 GiB unified memory. The fork’s native path keeps the experts in
FP8 on SM89+ GPUs and captures CUDA graphs for decode.

The checkpoint was **not cached** on this Spark and the available Wi-Fi link was
too slow and unstable to complete the download, so Drama 3 was deferred. Do not
treat the native Drama 3 path as hardware-validated here yet.

To finish the validation:

```bash
# 1. Download into the shared HF cache (41.5 GB; use wired networking).
uv run --project environments/dgx-spark --locked --extra cu130 \
    hf download fishaudio/drama-3

# 2. Stop any other model server first: Drama 3 needs most of the memory.
#    FISH_COMPILE is not passed for Drama 3 (--compile is unsupported).
FISH_MODEL=fishaudio/drama-3 FISH_LISTEN=127.0.0.1:8080 ./start.sh

# 3. In another terminal, exercise it.
python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 \
    --runs 1 --warmup 1 --output benchmarks/drama3.json
```

Useful Drama 3 switches:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FISH_DRAMA3_FP8_EXPERTS` | auto (`1` on SM89+) | `0` dequantizes experts to BF16; quality check/fallback, much more memory. |
| `FISH_DRAMA3_CUDA_GRAPH` | `1` | `0` disables CUDA-graph decode for debugging. |

At startup the log should show `FishOmni experts: block-FP8 W8A8`, the backbone
placement, and the captured CUDA-graph buckets. Peak memory and generated audio
duration should be recorded before comparing it with S2-Pro. If CUDA-graph
capture fails, retry once with `FISH_DRAMA3_CUDA_GRAPH=0` to separate graph
capture from model/kernel problems.

## WebUI

The API serves Awesome WebUI at `/ui`. `start.sh` builds it once if `dist` is
missing:

```bash
(cd awesome_webui && npm ci && npm run build)
./start.sh
```

For loopback-only access:

```bash
FISH_LISTEN=127.0.0.1:8080 ./start.sh
# On your desktop:
ssh -L 8080:127.0.0.1:8080 spark@YOUR_SPARK
```

Then open `http://127.0.0.1:8080/ui`. There is no Gradio public-share tunnel.
See the [Python API client](local-api-client.md) and [MCP server](mcp.md).

## If anything fails

Include the failing command, full traceback, `nvidia-smi` output, and the
diagnostic output above. For install failures include `uv --version`; for
TorchCodec shared-library failures include `ffmpeg -version`. A warning that the
TorchAudio `backend` argument is ignored is expected with 2.9+.

This guide covers the native ARM64 installation. The Docker/Compose defaults
still describe the upstream CUDA 12 environment and are not a validated Spark
path.

## Drama 3 optimization review

Select Drama 3 explicitly with `./start.sh --model fishaudio/drama-3`; S2-Pro
remains the default. Prefer a reference voice: the recorded small-sample tests
found trailing silence and token-cap termination without one.

The FP8 loader batches expert transfers while preserving checkpoint-order
independence. Drama 3 now honors `chunk_length` (100–1000 bytes), including the
WebUI slider. Try 100 for shorter initial waits; smaller chunks can alter prosody
and increase total generation time. `single_chunk=true` bypasses this splitting.

Historical benchmark JSON uses a buffered-read latency measurement. Rerun with
`tools/benchmark_tts.py --voice narrator --streaming --chunk-length 100 --runs 3`
to measure `first_audio_s` with the corrected reader. See the performance report
for evidence limitations; no quality advantage is established by timing alone.
