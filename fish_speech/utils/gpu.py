"""GPU detection, VRAM guidance, ROCm gfx arch auto-detection, and CPU weight offloading."""

import os

import torch
import torch.nn as nn
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
    return torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None


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
        suggestions.append("set OFFLOAD_WEIGHTS_TO_CPU=true to stream weights from system RAM")
        suggestions.append("set VRAM_FRACTION=0.95 to prevent system freeze on OOM")

        logger.warning(
            f"Estimated VRAM ({estimated_gb:.1f}GB) exceeds available ({total_gb:.1f}GB) "
            f"by {shortfall:.1f}GB. Suggestions:"
        )
        for i, s in enumerate(suggestions, 1):
            logger.warning(f"  {i}. {s}")


def _is_kv_cache(name: str) -> bool:
    return "k_cache" in name or "v_cache" in name


def _pin_layer(layer: nn.Module):
    """Pin all non-KV parameters and buffers for faster DMA transfers."""
    with torch.inference_mode(False):
        for param in layer.parameters():
            if param.device.type == "cpu" and not param.data.is_pinned():
                param.data = param.data.pin_memory()
        for name, buf in layer.named_buffers():
            if not _is_kv_cache(name) and buf.device.type == "cpu" and not buf.data.is_pinned():
                buf.data = buf.data.pin_memory()


def _layer_to_cpu(layer: nn.Module, pin: bool = False):
    """Move a layer to CPU memory, preserving KV caches on GPU.

    Args:
        pin: If True, pin memory for faster DMA. Only use for initial offload,
             not during streaming (pin_memory() overhead exceeds DMA savings).
    """
    kv_refs = {}
    for name, buf in layer.named_buffers():
        if _is_kv_cache(name):
            kv_refs[name] = buf.data
    with torch.inference_mode(False):
        layer.to("cpu")
    if pin:
        _pin_layer(layer)
    for name, buf in layer.named_buffers():
        if name in kv_refs:
            buf.data = kv_refs[name]


def _layer_to_gpu(layer: nn.Module, device: torch.device):
    """Move a layer to GPU, preserving KV caches (already on GPU)."""
    kv_refs = {}
    for name, buf in layer.named_buffers():
        if _is_kv_cache(name):
            kv_refs[name] = buf.data
    with torch.inference_mode(False):
        layer.to(device)
    for name, buf in layer.named_buffers():
        if name in kv_refs:
            buf.data = kv_refs[name]


class LayerStreamer:
    """Streams transformer layers between pinned CPU memory and GPU with prefetching.

    Replaces `for layer in self.layers: x = layer(x, ...)` with
    `x = model._layer_streamer.run(self.layers, x, *args)`.

    Weights are kept in pinned (page-locked) CPU memory between uses,
    which enables faster DMA transfers over PCIe when loading to GPU.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.prefetch_stream = torch.cuda.Stream()

    def run(self, layers: nn.ModuleList, x, *args, **kwargs):
        """Execute layers sequentially, streaming weights from pinned CPU memory."""
        n = len(layers)

        # Prefetch first layer
        with torch.cuda.stream(self.prefetch_stream):
            _layer_to_gpu(layers[0], self.device)

        for i, layer in enumerate(layers):
            # Wait for current layer to be on GPU
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)

            # Prefetch next layer while current one computes
            if i + 1 < n:
                with torch.cuda.stream(self.prefetch_stream):
                    _layer_to_gpu(layers[i + 1], self.device)

            # Run the layer
            x = layer(x, *args, **kwargs)

            # Move current layer back to CPU — the next iteration's
            # wait_stream() ensures prefetch is done before we proceed
            _layer_to_cpu(layer)

        return x


def setup_cpu_offload(model: nn.Module, device: torch.device):
    """Offload transformer layer weights to pinned CPU memory and stream on demand.

    Keeps KV caches on GPU for low-latency attention access. Weights are
    stored in pinned (page-locked) host memory for faster PCIe DMA transfers
    and streamed layer-by-layer with async prefetching via a separate HIP/CUDA stream.

    Enable with OFFLOAD_WEIGHTS_TO_CPU=true.
    """
    if not os.environ.get("OFFLOAD_WEIGHTS_TO_CPU", "").lower() in ("true", "1"):
        return False

    if not hasattr(model, "layers"):
        logger.warning("Model has no 'layers' attribute, cannot offload weights.")
        return False

    layers = model.layers
    n_layers = len(layers)

    # Move each layer to pinned CPU memory, keeping KV caches on GPU
    gpu_mem_before = torch.cuda.memory_allocated()
    for layer in layers:
        _layer_to_cpu(layer, pin=True)
    gpu_mem_after = torch.cuda.memory_allocated()
    saved_gb = (gpu_mem_before - gpu_mem_after) / 1e9

    logger.info(
        f"CPU offload: moved {n_layers} layers to pinned host memory, "
        f"freed {saved_gb:.1f}GB VRAM. KV caches remain on GPU."
    )

    # Keep fast_layers on GPU — they're small (~200MB) but called 10× per token
    # (once per codebook). Offloading them would add 40 PCIe round-trips per token.
    fast_layers = getattr(model, "fast_layers", None)
    if fast_layers is not None:
        fast_mb = sum(p.numel() * p.element_size() for p in fast_layers.parameters()) / 1e6
        logger.info(f"CPU offload: keeping {len(fast_layers)} fast layers on GPU ({fast_mb:.0f}MB).")

    # Attach a LayerStreamer to the model — the forward methods will use it
    model._layer_streamer = LayerStreamer(device)

    return True
