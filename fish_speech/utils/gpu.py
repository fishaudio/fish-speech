"""GPU detection, VRAM guidance, and ROCm gfx arch auto-detection."""

import os

import torch
from loguru import logger

# Known ROCm gfx arch overrides for GPUs not yet in PyTorch's HIP target list.
# Maps gcnArchName to the closest supported HSA_OVERRIDE_GFX_VERSION.
_ROCM_GFX_OVERRIDES = {
    "gfx1201": "12.0.0",  # Navi 48 — RX 9070/9070 XT → fallback to gfx1200
}

# Approximate model memory requirements (in GB) for VRAM guidance.
_MODEL_ESTIMATE_BF16 = 10.3
_MODEL_ESTIMATE_INT8 = 5.1
_DECODER_ESTIMATE_BF16 = 3.6
_DECODER_ESTIMATE_INT8 = 1.8


def _is_rocm() -> bool:
    """Check if running on ROCm (AMD HIP backend)."""
    return (
        torch.cuda.is_available()
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    )


def auto_detect_rocm_gfx():
    """Set HSA_OVERRIDE_GFX_VERSION if running on an unrecognized AMD GPU.

    Only acts when:
    - Running on ROCm (HIP backend)
    - HSA_OVERRIDE_GFX_VERSION is not already set
    - The GPU's gcnArchName matches a known override
    """
    if not _is_rocm():
        return
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
        return

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    if arch is None:
        return

    gfx_ver = _ROCM_GFX_OVERRIDES.get(arch)
    if gfx_ver is not None:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = gfx_ver
        logger.info(
            f"Auto-detected AMD GPU arch {arch}, "
            f"setting HSA_OVERRIDE_GFX_VERSION={gfx_ver}"
        )


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
