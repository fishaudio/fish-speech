# DGX Spark setup

This fork tracks Fish Speech 3.0, including S2-Pro and Drama-3. CUDA 13 is
opt-in and isolated from upstream's default PyTorch 2.8 installation. GPU kernels,
audio decoding, and end-to-end generation need validation on an actual Spark.

## Selected dependencies

| Package | Version | Source |
| --- | --- | --- |
| Python | 3.12 | System Python or uv-managed Python |
| torch | 2.10.0 | Official PyTorch cu130 index |
| torchaudio | 2.10.0 | Official PyTorch cu130 index |
| torchcodec | 0.10.0 | Official PyTorch cu130 index |

PyTorch 2.9 does provide CUDA 13 ARM64 wheels. However, TorchAudio 2.9+ uses
TorchCodec for `torchaudio.load`, which Fish Speech uses for reference audio.
The TorchCodec 0.8/0.9 wheels compatible with PyTorch 2.9 do not include Linux
ARM64 in the checked PyPI, CPU, cu128, and cu130 indexes. The 2.10/0.10 pair
provides official ARM64 wheels without a source build or an audio-loader rewrite.

The wheel supplies its CUDA runtime dependencies. The host still needs a working
NVIDIA driver compatible with CUDA 13. `nvidia-smi` reports driver capability;
`nvcc --version` reports the separately installed toolkit. Neither alone proves
that the Python environment has a CUDA-enabled PyTorch build.

Sources: [PyTorch versions](https://pytorch.org/get-started/previous-versions/),
[TorchCodec compatibility](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions),
[TorchCodec cu130 wheels](https://download.pytorch.org/whl/cu130/torchcodec/),
[TorchAudio loading](https://docs.pytorch.org/audio/2.9.0/generated/torchaudio.load.html).

## Install

Run these commands on the Spark, in this repository. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first if needed.

```bash
uname -m                       # expected: aarch64
nvidia-smi                     # must detect the GB10
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev libsox-dev libsndfile1 ffmpeg

uv sync --project environments/dgx-spark --locked --python 3.12 --extra cu130
```

Or run `./install-dgx-spark.sh`, which checks the host, installs the Ubuntu
system dependencies, and runs the same locked uv command.

The environment lives in `environments/dgx-spark/.venv`. Keep both
`--project environments/dgx-spark` and `--extra cu130` on manual uv commands;
the launch scripts supply them for you. An existing root `.venv` is not reused.
Model checkpoints and saved references stay in the repository root.

### Why a separate project?

The root `pyproject.toml` retains upstream's exact Torch/TorchAudio 2.8.0 pins,
`stable`, `cpu`, `cu126`, `cu128`, and `cu129` extras, indexes, and conflicts.
A bare `pip install -e .` or `uv sync` therefore still selects 2.8.0.
An extra adds requirements; it cannot replace a mandatory 2.8.0 requirement with
2.10.0. The Spark project scopes uv overrides to an explicit `cu130` environment
and installs the root package editably. It has its own checked-in lockfile.

Use uv for the Spark environment. Pip does not honor uv overrides or sources;
`pip install -e '.[cu130]'` is not supported, and `--extra-index-url` alone
cannot resolve contradictory Torch pins. Root pip installation instructions
remain unchanged. This implements the compatibility goal of
[the upstream review](https://github.com/fishaudio/fish-speech/pull/1348#issuecomment-6095246512)
using a separate project rather than a root extra.

## Verify CUDA and reference-audio decoding

Run this before downloading the model. It prints the environment, runs actual
CUDA operations, and decodes a generated WAV from both a path and memory using
the same TorchAudio API used by Fish Speech. It needs no checkpoints.

```bash
uv run --project environments/dgx-spark --locked --extra cu130 python - <<'PY'
import io
import platform
import tempfile
import wave
from pathlib import Path

import torch
import torchaudio
import torchcodec

print("Architecture:", platform.machine())
print("Python:", platform.python_version())
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
y = x @ x.T
assert torch.isfinite(y).all().item()
q = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
z = torch.nn.functional.scaled_dot_product_attention(q, q, q)
assert torch.isfinite(z).all().item()
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

## Download and run

For S2-Pro:

```bash
uv run --project environments/dgx-spark --locked --extra cu130 hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
./start-api.sh
```

For Drama-3, download its weights as described in the
[upstream installation guide](install.md#download-weights), then run:

```bash
./start-api.sh --llama-checkpoint-path checkpoints/drama-3 --device cuda:0
```

Drama-3 requires at least 48 GB of GPU memory according to upstream and does
not support `--compile`. GB10 execution has not been tested here. S2-Pro remains
the default checkpoint. To enable compilation explicitly for S2-Pro:

```bash
./start-api.sh --compile
```

The compilation cache defaults to `$HOME/.cache/fish-speech/inductor`; override
it with `TORCHINDUCTOR_CACHE_DIR`. First test generation with and without a WAV
reference before enabling compilation.

### WebUI

Upstream removed the Gradio launcher. The API now serves Awesome WebUI at `/ui`.
Build it once with Node.js and npm, and rebuild after frontend updates:

```bash
(cd awesome_webui && npm ci && npm run build)
./start-webui.sh
```

Open `http://<SPARK_LOCAL_IP>:8080/ui`. Both launchers start the same API server;
run only one for a given port/model. They bind to `0.0.0.0:8080` for LAN access,
forward arguments unchanged, and work when launched from another directory.
For loopback-only access or an SSH tunnel:

```bash
./start-webui.sh --listen 127.0.0.1:8080
# On your desktop:
ssh -L 8080:127.0.0.1:8080 spark@YOUR_SPARK
```

Then open `http://127.0.0.1:8080/ui`. There is no Gradio public share tunnel.
See [the Python API client guide](local-api-client.md) for the standalone client.

## If anything fails

Send the failing command, full traceback, `nvidia-smi` output, and the diagnostic
output above. For install failures, also include `uv --version`. If TorchCodec
cannot load its shared libraries, include `ffmpeg -version`; Ubuntu's FFmpeg
shared libraries must be available. A warning that the TorchAudio `backend`
argument is ignored is expected with 2.9+; installing SoundFile alone does not
replace TorchCodec in these versions.

This guide covers native installation. Existing Docker/Compose defaults still
describe the upstream CUDA 12 environment and are not a validated Spark path.
