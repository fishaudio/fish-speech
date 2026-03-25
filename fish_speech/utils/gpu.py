"""GPU detection, VRAM guidance, ROCm gfx arch auto-detection, and GTT weight offloading."""

import os
import subprocess

import torch
import torch.nn as nn
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


def _is_rocm() -> bool:
    """Check if running on ROCm (AMD HIP backend)."""
    return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None


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
        if _is_rocm():
            suggestions.append("set OFFLOAD_WEIGHTS_TO_GTT=true to stream weights from system RAM")
        suggestions.append("set VRAM_FRACTION=0.95 to prevent system freeze on OOM")

        logger.warning(
            f"Estimated VRAM ({estimated_gb:.1f}GB) exceeds available ({total_gb:.1f}GB) "
            f"by {shortfall:.1f}GB. Suggestions:"
        )
        for i, s in enumerate(suggestions, 1):
            logger.warning(f"  {i}. {s}")


def setup_gtt_offload(model: nn.Module, device: torch.device):
    """Offload transformer layer weights to CPU (GTT) and stream them to GPU on demand.

    ROCm only. Keeps KV caches on GPU for low-latency attention access.
    Weights are moved to pinned CPU memory and streamed layer-by-layer
    with async prefetching of the next layer via a separate HIP stream.

    Enable with OFFLOAD_WEIGHTS_TO_GTT=true.
    """
    if not os.environ.get("OFFLOAD_WEIGHTS_TO_GTT", "").lower() in ("true", "1"):
        return False

    if not _is_rocm():
        logger.warning("OFFLOAD_WEIGHTS_TO_GTT is only supported on AMD ROCm devices, ignoring.")
        return False

    # Find the transformer layers (self.layers in BaseTransformer)
    if not hasattr(model, "layers"):
        logger.warning("Model has no 'layers' attribute, cannot offload weights.")
        return False

    layers = model.layers
    n_layers = len(layers)
    prefetch_stream = torch.cuda.Stream()

    # Move each layer's parameters to pinned CPU memory, but keep KV caches on GPU.
    gpu_mem_before = torch.cuda.memory_allocated()
    for layer in layers:
        _offload_layer_to_cpu(layer, device)
    gpu_mem_after = torch.cuda.memory_allocated()
    saved_gb = (gpu_mem_before - gpu_mem_after) / 1e9

    logger.info(
        f"GTT offload: moved {n_layers} layers to pinned CPU memory, "
        f"freed {saved_gb:.1f}GB VRAM. KV caches remain on GPU."
    )

    # Also handle fast_layers if present (DualAR fast transformer)
    fast_layers = getattr(model, "fast_layers", None)
    if fast_layers is not None:
        for layer in fast_layers:
            _offload_layer_to_cpu(layer, device)
        logger.info(f"GTT offload: also moved {len(fast_layers)} fast layers to CPU.")

    # Register hooks for streaming weights layer-by-layer
    _register_offload_hooks(layers, prefetch_stream, device)
    if fast_layers is not None:
        _register_offload_hooks(fast_layers, prefetch_stream, device)

    return True


def _offload_layer_to_cpu(layer: nn.Module, device: torch.device):
    """Move a layer's parameters and non-KV buffers to pinned CPU memory."""
    for name, param in layer.named_parameters():
        # Pin the CPU tensor for faster PCIe transfers
        cpu_data = param.data.to("cpu").pin_memory()
        param.data = cpu_data

    for name, buf in layer.named_buffers():
        # Keep KV caches on GPU — they're the whole point of this optimization
        if "k_cache" in name or "v_cache" in name:
            continue
        cpu_data = buf.data.to("cpu").pin_memory()
        buf.data = cpu_data


def _register_offload_hooks(layers: nn.ModuleList, prefetch_stream, device):
    """Register forward hooks that stream weights GPU↔CPU with prefetching."""
    n_layers = len(layers)

    def make_pre_hook(layer_idx):
        def pre_hook(module, args):
            # Wait for prefetch to complete (if this layer was prefetched)
            torch.cuda.current_stream().wait_stream(prefetch_stream)

            # If this is the first layer (no prefetch), move synchronously
            if not hasattr(module, "_on_gpu"):
                _move_layer_to_device(module, device)
            module._on_gpu = True

            # Prefetch next layer asynchronously
            if layer_idx + 1 < n_layers:
                next_layer = layers[layer_idx + 1]
                with torch.cuda.stream(prefetch_stream):
                    _move_layer_to_device(next_layer, device)
                    next_layer._on_gpu = True

        return pre_hook

    def make_post_hook(layer_idx):
        def post_hook(module, args, output):
            # Move this layer back to CPU after use
            _move_layer_to_cpu(module)
            module._on_gpu = False
            return output

        return post_hook

    for idx, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_pre_hook(idx))
        layer.register_forward_hook(make_post_hook(idx))


def _move_layer_to_device(layer: nn.Module, device):
    """Move layer parameters and non-KV buffers to GPU."""
    for name, param in layer.named_parameters():
        if param.device.type != "cpu":
            continue
        param.data = param.data.to(device, non_blocking=True)

    for name, buf in layer.named_buffers():
        if "k_cache" in name or "v_cache" in name:
            continue
        if buf.device.type != "cpu":
            continue
        buf.data = buf.data.to(device, non_blocking=True)


def _move_layer_to_cpu(layer: nn.Module):
    """Move layer parameters and non-KV buffers back to pinned CPU memory."""
    for name, param in layer.named_parameters():
        if param.device.type == "cpu":
            continue
        param.data = param.data.to("cpu", non_blocking=True).pin_memory()

    for name, buf in layer.named_buffers():
        if "k_cache" in name or "v_cache" in name:
            continue
        if buf.device.type == "cpu":
            continue
        buf.data = buf.data.to("cpu", non_blocking=True).pin_memory()
