# DGX Spark setup

This fork runs Fish Speech 3.0 on the NVIDIA DGX Spark (GB10). Both **S2-Pro**
and **Drama 3** are validated in the native PyTorch path. S2-Pro is the small,
fast model; Drama 3 is the 35B-A3B flagship and is the better quality/voice
cloning choice. Drama 3 needs a reference voice for reliable end-of-speech
behavior; see [Drama 3](#drama-3).

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
2. downloads `fishaudio/drama-3` into the shared Hugging Face cache if needed,
3. builds the Awesome WebUI once, and
4. starts the API on `0.0.0.0:8080` with Drama 3.

Open `http://<SPARK_IP>:8080/ui`. The API is at
`http://<SPARK_IP>:8080/v1/`. Drama 3 cold-starts in about **67–70 s** after the
batched FP8 loader change (41.3 GiB peak, 62k parameter slots). S2-Pro startup is
about 2–2.5 minutes because its decode function is compiled; later requests reuse
the Inductor cache in `~/.cache/fish-speech/inductor`.

For the MCP server, see [MCP server](mcp.md):

```bash
./start.sh mcp          # same as ./start-mcp.sh
```

### Useful environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FISH_MODEL` | `fishaudio/drama-3` | `fishaudio/drama-3`, `fishaudio/s2-pro`, or a local path. |
| `FISH_COMPILE` | `1` for S2-Pro | Set `0` for eager mode (slower steady state, faster startup). |
| `FISH_COMPILE_MODE` | `default` | Passed to `torch.compile`; `reduce-overhead` was not stable here. |
| `FISH_DEVICE` | unset | Example: `cuda:0`; passed as `--device`. |
| `FISH_LISTEN` | `0.0.0.0:8080` | Example: `127.0.0.1:8080` for loopback only. |
| `FISH_OFFLINE` | `0` | Set `1` to pass `--offline` and never contact the Hub. |
| `FISH_SKIP_DOWNLOAD` | `0` | Set `1` to skip the startup cache check. |
| `FISH_SKIP_WEBUI` | `0` | Set `1` to skip the one-time WebUI build. |

Examples:

```bash
# S2-Pro instead of the default Drama 3, loopback only
FISH_MODEL=fishaudio/s2-pro FISH_LISTEN=127.0.0.1:8080 ./start.sh

# Faster S2-Pro startup (no Inductor compile, slower steady state)
FISH_MODEL=fishaudio/s2-pro FISH_COMPILE=0 ./start.sh

# Drama 3 is the default; pass chunk_length=100 in requests for lower first audio
FISH_MODEL=fishaudio/drama-3 ./start.sh
```

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

### Drama 3 warm performance

With reference voice `drama-ref` (an S2-generated 6.1 s clip), 1.7B CustomVoice
equivalent, seed 42, three runs:

| Case | Audio | Wall | RTF |
| --- | ---: | ---: | ---: |
| 12 chars | 0.98 s | 1.66 s | 1.70 |
| 99 chars | 5.62 s | 6.84 s | 1.22 |
| 250 chars | 13.84 s | 16.05 s | 1.16 |

Drama 3 is therefore **faster than S2-Pro in steady state** and produces the
better-quality model, at the cost of 41.3 GiB peak memory and a 67–70 s cold
load. The short case is dominated by request/first-frame overhead.

Streaming, 250-character case:

| Chunk size | First audio | Total |
| --- | ---: | ---: |
| `chunk_length=200` | 7.02 s | 16.39 s |
| `chunk_length=100` | **1.35 s** | 17.15 s |

Use `chunk_length=100` for interactive first audio. Drama 3 now honors the
request's `chunk_length`; previously it always used the 300-byte default and
emitted one segment for this text.

## Drama 3

Drama 3 is a 35B-A3B hybrid-attention model with block-FP8 experts, a 4-layer
codebook tower and CUDA-graph decode. The checkpoint is about **41.5 GB** (eight
~5 GB shards plus the 1.87 GB codec); peak resident memory is **41.3 GiB**.
`fishaudio/drama-3` is now the default `start.sh` model.

Cold start on this Spark: **67–70 s**. The original per-expert loader copied
62,053 small tensors one at a time and took about **5.5 minutes**; the new
batched FP8 loader stacks one layer/projection and performs one bulk copy,
cutting backbone load to around 41 s. The tower takes another ~13 s and the codec
~3 s.

### Use a reference voice

Drama 3 without a reference voice is **not reliable** on this model: in an
8-seed test, 5 seeds failed to emit EOS and ran to the token cap, appending
near-silence (18.6–27.9 s for a 99-character line). With a reference, all four
tested seeds stopped naturally at 5.6–5.8 s with normal speech energy. Use a
reference for production, especially voice cloning:

```python
from fishapi import FishAPI
api = FishAPI('http://127.0.0.1:8080')
api.add_voice('narrator', 'reference.wav', 'Exact transcript of the reference.')
```

Then send `reference_id: "narrator"` with each request. The service keeps
references under the repository root; see [the API client guide](local-api-client.md).

### Run and measure

```bash
# Drama 3 is the default now; use FISH_MODEL=fishaudio/s2-pro to switch.
./start.sh

python3 tools/benchmark_tts.py --api-url http://127.0.0.1:8080 \
    --voice narrator --runs 3 --output benchmarks/drama3-reference.json
```

Useful switches:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FISH_DRAMA3_FP8_EXPERTS` | auto (`1` on SM89+) | `0` dequantizes experts to BF16; quality check/fallback, much more memory. |
| `FISH_DRAMA3_CUDA_GRAPH` | `1` | `0` disables CUDA-graph decode for debugging. |
| `FISH_COMPILE` | `0` for Drama 3 | Drama 3 rejects `--compile`; ignored by `start.sh`. |

At startup the log should show `FishOmni experts: block-FP8 W8A8`, the backbone
placement, and `Drama 3 decode CUDA graphs ... captured`. If graph capture fails,
retry once with `FISH_DRAMA3_CUDA_GRAPH=0` to separate graph capture from
model/kernel problems. If FP8 kernels fail on SM 12.1,
`FISH_DRAMA3_FP8_EXPERTS=0` is the dequantized fallback, but it needs
substantially more memory.

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
