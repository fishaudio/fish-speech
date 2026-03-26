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
        suggestions.append("set OFFLOAD_WEIGHTS_TO_CPU=true to run slow layers on CPU")
        suggestions.append("set VRAM_FRACTION=0.95 to prevent system freeze on OOM")

        logger.warning(
            f"Estimated VRAM ({estimated_gb:.1f}GB) exceeds available ({total_gb:.1f}GB) "
            f"by {shortfall:.1f}GB. Suggestions:"
        )
        for i, s in enumerate(suggestions, 1):
            logger.warning(f"  {i}. {s}")


class CPUOffloadExecutor:
    """Runs slow transformer layers on CPU (using AVX-512/VNNI), keeps fast path on GPU.

    Instead of streaming layers GPU↔CPU (72 PCIe round-trips per token),
    this executes the slow transformer entirely on CPU and only transfers
    the final hidden state (~10KB) to GPU for the fast transformer + decoder.

    For batch=1 single-token inference, CPU execution with DDR5 bandwidth
    (~80-100 GB/s) and AVX-512 is competitive with the PCIe streaming approach
    while eliminating all allocation overhead.
    """

    def __init__(self, gpu_device: torch.device):
        self.gpu_device = gpu_device

    def run(self, layers: nn.ModuleList, x, *args, **kwargs):
        """Execute layers on CPU, return result on pinned memory for fast GPU transfer."""
        # Move hidden state and all positional args to CPU
        x_cpu = x.to("cpu")
        args_cpu = tuple(a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args)
        kwargs_cpu = {
            k: v.to("cpu") if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        # Run all layers on CPU — weights are already here, no PCIe needed
        for layer in layers:
            x_cpu = layer(x_cpu, *args_cpu, **kwargs_cpu)

        # Pin the result for faster DMA to GPU, then transfer non-blocking
        return x_cpu.pin_memory().to(self.gpu_device, non_blocking=True)


def _has_int8_weights(module: nn.Module) -> bool:
    """Check if any submodule uses INT8 quantized weights."""
    for child in module.modules():
        if hasattr(child, "weight") and hasattr(child, "scales") and child.weight.dtype == torch.int8:
            return True
    return False


def setup_cpu_offload(model: nn.Module, device: torch.device):
    """Offload slow transformer layers to CPU execution.

    Moves slow layer weights + KV caches to CPU. The slow transformer runs
    entirely on CPU using AVX-512, and only the final hidden state is
    transferred to GPU for the fast transformer and decoder.

    Fast layers stay on GPU (small footprint, called 10x per token).

    Enable with OFFLOAD_WEIGHTS_TO_CPU=true.
    Requires native bf16 weights — INT8 quantized models are not supported
    because autoregressive decode (M=1) cannot use VNNI _int_mm (requires M>16),
    and the dequant+bf16 fallback is ~30% slower than native bf16 matmuls.
    """
    if not os.environ.get("OFFLOAD_WEIGHTS_TO_CPU", "").lower() in ("true", "1"):
        return False

    if not hasattr(model, "layers"):
        logger.warning("Model has no 'layers' attribute, cannot offload weights.")
        return False

    if _has_int8_weights(model):
        logger.warning(
            "CPU offload requires native bf16 weights. INT8 quantized models are not supported "
            "because autoregressive decode (batch=1) cannot use VNNI INT8 matmuls (requires M>16), "
            "and the dequant+bf16 fallback is ~30% slower than native bf16. "
            "Please use the original (non-quantized) checkpoint with OFFLOAD_WEIGHTS_TO_CPU=true."
        )
        return False

    # Use physical cores only — HyperThreading causes cache contention
    # on Zen 4 and hurts bf16 matmul throughput (~37% slower with HT).
    physical_cores = os.cpu_count() // 2 if os.cpu_count() else 8
    torch.set_num_threads(physical_cores)
    logger.info(f"CPU offload: set torch threads to {physical_cores} (physical cores only)")

    layers = model.layers
    n_layers = len(layers)

    # Move slow layers entirely to CPU (including KV caches)
    gpu_mem_before = torch.cuda.memory_allocated()
    with torch.inference_mode(False):
        for layer in layers:
            layer.to("cpu")
    gpu_mem_after = torch.cuda.memory_allocated()
    saved_gb = (gpu_mem_before - gpu_mem_after) / 1e9

    # Move shared slow-path modules to CPU.
    # Keep causal_mask and fast_freqs_cis on GPU (shared with fast path).
    for name in ("norm", "embeddings", "codebook_embeddings", "output"):
        module = getattr(model, name, None)
        if module is not None:
            with torch.inference_mode(False):
                if isinstance(module, nn.Module):
                    module.to("cpu")

    gpu_mem_final = torch.cuda.memory_allocated()
    total_saved_gb = (gpu_mem_before - gpu_mem_final) / 1e9

    logger.info(
        f"CPU offload: moved {n_layers} slow layers + shared modules to CPU, "
        f"freed {total_saved_gb:.1f}GB VRAM. Fast layers + decoder remain on GPU."
    )

    # Keep fast_layers on GPU — small footprint, called 10x per token
    fast_layers = getattr(model, "fast_layers", None)
    if fast_layers is not None:
        logger.info(f"CPU offload: keeping {len(fast_layers)} fast layers on GPU.")

    # Attach executor — forward_generate will use it
    model._layer_streamer = CPUOffloadExecutor(device)

    return True
