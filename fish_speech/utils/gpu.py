"""GPU detection, VRAM guidance, and ROCm gfx arch auto-detection."""

import os
import subprocess

import torch
from loguru import logger

# Known ROCm gfx arch overrides for GPUs not yet in PyTorch's HIP target list.
# Maps PCI device ID prefixes to the closest supported gfx version.
_ROCM_GFX_OVERRIDES = {
    "0x7550": "12.0.0",  # Navi 48 — RX 9070/9070 XT (gfx1201 → gfx1200)
    "0x7551": "12.0.0",  # Navi 48 variants
}

# Approximate model memory requirements (in GB) for VRAM guidance.
_MODEL_ESTIMATE_BF16 = 10.3
_MODEL_ESTIMATE_INT8 = 5.1
_DECODER_ESTIMATE_BF16 = 3.6
_DECODER_ESTIMATE_INT8 = 1.8


def _get_amd_device_id() -> str | None:
    """Read the PCI device ID from the first AMD render node."""
    try:
        result = subprocess.run(
            ["cat", "/sys/class/drm/renderD128/device/device"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def auto_detect_rocm_gfx():
    """Set HSA_OVERRIDE_GFX_VERSION if running on an unrecognized AMD GPU.

    Only acts when:
    - torch.cuda (HIP) is available
    - HSA_OVERRIDE_GFX_VERSION is not already set
    - The GPU's PCI device ID matches a known override
    """
    if not torch.cuda.is_available():
        return
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return

    device_id = _get_amd_device_id()
    if device_id is None:
        return

    for prefix, gfx_ver in _ROCM_GFX_OVERRIDES.items():
        if device_id == prefix:
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_ver
            logger.info(
                f"Auto-detected AMD GPU (device {device_id}), "
                f"setting HSA_OVERRIDE_GFX_VERSION={gfx_ver}"
            )
            return


def check_vram_and_advise(checkpoint_path: str):
    """Log VRAM guidance if the model may not fit.

    Estimates memory usage based on whether INT8 quantization is active
    and the configured MAX_SEQ_LEN, then compares against available VRAM.
    """
    if not torch.cuda.is_available():
        return

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1e9

    is_int8 = "int8" in str(checkpoint_path)
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "32768"))

    model_gb = _MODEL_ESTIMATE_INT8 if is_int8 else _MODEL_ESTIMATE_BF16
    decoder_gb = _DECODER_ESTIMATE_BF16
    # KV cache: ~1.2GB at 8192, scales linearly
    kv_gb = (max_seq_len / 8192) * 1.2
    # Inference scratch/activations overhead
    overhead_gb = 0.5

    estimated_gb = model_gb + decoder_gb + kv_gb + overhead_gb

    logger.info(
        f"GPU: {props.name}, VRAM: {total_gb:.1f}GB | "
        f"Estimated usage: {estimated_gb:.1f}GB "
        f"(model={'INT8' if is_int8 else 'bf16'}, "
        f"seq_len={max_seq_len}, decoder=bf16)"
    )

    if estimated_gb > total_gb:
        shortfall = estimated_gb - total_gb
        suggestions = []
        if not is_int8:
            suggestions.append(
                "quantize to INT8 (saves ~5GB): "
                "python tools/llama/quantize.py --checkpoint-path <path> --mode int8"
            )
        if max_seq_len > 4096:
            suggestions.append(
                f"reduce MAX_SEQ_LEN (current: {max_seq_len}, try 4096 to save ~{(max_seq_len - 4096) / 8192 * 1.2:.1f}GB)"
            )
        suggestions.append("set VRAM_FRACTION=0.95 to prevent system freeze on OOM")

        logger.warning(
            f"Estimated VRAM ({estimated_gb:.1f}GB) exceeds available ({total_gb:.1f}GB) "
            f"by {shortfall:.1f}GB. Suggestions:"
        )
        for i, s in enumerate(suggestions, 1):
            logger.warning(f"  {i}. {s}")
