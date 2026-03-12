# Quantization Guide for Low VRAM GPUs

## Problem

The S2-Pro model (4B parameters) requires more than 16GB VRAM for inference, making it inaccessible to users with consumer GPUs (RTX 3060/3070/4060/4070).

## Solution

fish-speech includes built-in quantization support that can reduce memory usage by 50-75%.

## Quick Start

### INT8 Quantization (~50% memory reduction)

```bash
python tools/llama/quantize.py \
  --checkpoint-path checkpoints/fish-speech-1.5 \
  --mode int8
```

This creates `checkpoints/fs-1.2-int8-{timestamp}/` with quantized weights.

**Memory:** ~8-10GB VRAM (fits 12GB cards)

### INT4 Quantization (~75% memory reduction)

```bash
python tools/llama/quantize.py \
  --checkpoint-path checkpoints/fish-speech-1.5 \
  --mode int4 \
  --groupsize 128
```

**Memory:** ~5-6GB VRAM (fits 8GB cards)

## Usage

After quantization, use the new checkpoint:

```bash
python fish_speech/text_to_speech.py \
  --checkpoint checkpoints/fs-1.2-int4-{timestamp} \
  --text "Hello world"
```

## Benchmarks

| Mode | VRAM | Quality Loss |
|------|------|--------------|
| FP32 | 16GB+ | 0% |
| BF16 | 12GB+ | <1% |
| INT8 | 8-10GB | 1-2% |
| INT4 | 5-6GB | 3-5% |

## Requirements

- PyTorch 2.0+
- CUDA 11.8+
- `torch.ops.aten._weight_int4pack_mm` support (PyTorch 2.2+)

## Troubleshooting

**"require in_features % 1024 == 0"**

Some layers will be padded. This is normal and has minimal impact.

**"CUDA out of memory during quantization"**

Quantization runs on CPU by default. If you still see OOM, try:
```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tools/llama/quantize.py ...
```

## Related

- Issue #1168: Quantization request for S2-Pro
- Issue #852: Low accuracy after training

---

**Note:** Quantization is only for inference, not training. For training with low VRAM, use gradient checkpointing and mixed precision.
