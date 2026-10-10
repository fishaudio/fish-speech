# DGX Spark setup

This guide covers native Linux ARM64 installation on the DGX Spark's GB10 GPU.
The model and inference code are unchanged. This branch now makes CUDA 13.0 the default dependency stack for the whole Python project. Dependency resolution has been
checked for Linux ARM64 / Python 3.12, and a DGX Spark user reported successfully
running the WebUI with this dependency profile. Compiled inference and all
reference-audio formats have not been independently validated on the hardware.

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

On DGX Spark's Ubuntu host, run the installer as your normal user:

```bash
./install-dgx-spark.sh
```

It checks for Linux ARM64 and uv, uses sudo for the required Ubuntu system
packages, and runs `uv sync --python 3.12` as your user.
It does not download model weights or change the NVIDIA driver. You can rerun
it to synchronize the environment after pulling updates.

Alternatively, perform the same setup manually:

```bash
uname -m                       # expected: aarch64
nvidia-smi                     # must detect the GB10
sudo apt-get update
sudo apt-get install -y build-essential python3-dev portaudio19-dev libsox-dev libsndfile1 ffmpeg

uv sync --python 3.12
```

This creates/updates the project's `.venv`, not your global Python environment.
CUDA 13 is selected by default for `uv sync` and `uv run`; no `--extra` flag is needed. This changes the project's default dependency stack and is not appropriate for non-CUDA-13 hosts.

Use uv for this setup: pip does not read `[tool.uv.sources]` or the existing protobuf override. The configuration now targets CUDA 13.0 by default; the older CPU/CUDA 12 extras are no longer provided by this branch.

## Verify CUDA and reference-audio decoding

Run this before downloading the model. It prints the environment, runs actual
CUDA operations, and decodes a generated WAV from both a path and memory using
the same TorchAudio API used by Fish Speech. It needs no checkpoints.

```bash
uv run python - <<'PY'
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
assert torch.cuda.is_available(), "CUDA unavailable: check driver and PyTorch installation"
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

```bash
uv run hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro
uv run python tools/run_webui.py --device cuda
```

Open the local URL printed by Gradio (normally `http://127.0.0.1:7860`).
For a remote Spark, forward port 7860 over SSH from your desktop, for example
`ssh -L 7860:127.0.0.1:7860 spark@YOUR_SPARK`, then open that URL locally.

First generate speech without reference audio; then try a short WAV reference
and its transcript. The first command deliberately omits `--compile` so that
basic GPU execution is checked before Triton compilation. After both succeed,
you can try:

```bash
uv run python tools/run_webui.py --device cuda --compile
```

Compilation is a separate, unverified step and may have GB10/Triton-specific
issues. If only compilation fails, omit `--compile` while investigating it.

For the API instead of Gradio:

```bash
uv run python tools/api_server.py --device cuda --listen 127.0.0.1:8080
```

## If anything fails

Send the failing command, full traceback, `nvidia-smi` output, and the diagnostic
output above. For install failures, also include `uv --version`. If TorchCodec
cannot load its shared libraries, include `ffmpeg -version`; Ubuntu's FFmpeg
shared libraries must be available. A warning that the TorchAudio `backend`
argument is ignored is expected with 2.9+; installing SoundFile alone does not
replace TorchCodec in these versions.

This guide covers native installation. Existing Docker/Compose defaults still
describe the upstream CUDA 12 environment and are not a validated Spark path.
