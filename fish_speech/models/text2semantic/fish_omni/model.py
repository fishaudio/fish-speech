import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from loguru import logger
from safetensors import safe_open
from tokenizers import Tokenizer
from torch import Tensor
from torch.types import Device

from fish_speech.models.text2semantic.fish_omni.backbone import (
    CHECKPOINT_BACKBONE_PREFIX,
    CHECKPOINT_LM_HEAD_KEY,
    FishOmniBackbone,
    load_backbone_weights,
)
from fish_speech.models.text2semantic.fish_omni.config import (
    FISH_OMNI_MODEL_TYPE,
    FP8_QUANT_METHOD,
    FishOmniConfig,
)
from fish_speech.models.text2semantic.fish_omni.fp8_moe import fp8_gemm_supported
from fish_speech.models.text2semantic.fish_omni.tower import (
    FishOmniAudioDecoder,
    load_audio_decoder_weights,
)

if TYPE_CHECKING:
    from fish_speech.models.text2semantic.fish_omni.generate import S3DecodeGraphs

AUDIO_DECODER_PREFIXES = (
    "audio_decoder.",
    "audio_decoder_mtp_shared.",
    "audio_decoder_mtp.",
)


class FishOmniModel(nn.Module):
    """S3 TTS model: backbone (over one or more devices) + codebook tower + tokenizer."""

    def __init__(
        self,
        config: FishOmniConfig,
        backbone: FishOmniBackbone,
        audio_decoder: FishOmniAudioDecoder,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.audio_decoder = audio_decoder
        self.tokenizer = tokenizer
        # Decode state reused across requests; managed by fish_omni.generate.
        self.cache_capacity = 0
        self.decode_graphs: "S3DecodeGraphs | None" = None

    @property
    def dtype(self) -> torch.dtype:
        return self.backbone.dtype

    @property
    def device(self) -> torch.device:
        """Where frames are sampled and codes come out (the tower's device)."""
        return self.audio_decoder.device

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        *,
        devices: Sequence[Device],
        tower_device: Device,
        dtype: torch.dtype,
    ) -> "FishOmniModel":
        """Load backbone (spread over devices), tower and tokenizer.

        FP8 checkpoint experts stay FP8 when every backbone device is SM89+ and are
        dequantized to dtype otherwise; env FISH_S3_FP8_EXPERTS=0/1 forces either way.
        """
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "config.json", encoding="utf-8") as f:
            config = FishOmniConfig.from_dict(json.load(f))
        fp8_available = config.fp8_experts and fp8_gemm_supported(devices)
        forced = os.environ.get("FISH_S3_FP8_EXPERTS", "").strip()
        fp8_experts = forced == "1" if forced in ("0", "1") else fp8_available
        if fp8_experts and not fp8_available:
            raise ValueError(
                f"FP8 experts need a {FP8_QUANT_METHOD} checkpoint and SM89+ CUDA "
                f"devices with Triton; got {list(devices)}"
            )
        logger.info(
            "FishOmni experts: "
            + ("block-FP8 W8A8" if fp8_experts else f"dequantized to {dtype}")
        )

        with torch.device("meta"):
            backbone = FishOmniBackbone(
                config.text, fp8_experts=fp8_experts, dtype=dtype
            )
            audio_decoder = FishOmniAudioDecoder(config, dtype=dtype)
        backbone.materialize(devices)
        load_backbone_weights(
            backbone,
            iter_checkpoint_tensors(
                checkpoint_dir, skip_prefixes=AUDIO_DECODER_PREFIXES
            ),
        )
        audio_decoder.to_empty(device=tower_device)
        load_audio_decoder_weights(
            audio_decoder,
            iter_checkpoint_tensors(
                checkpoint_dir,
                skip_prefixes=(CHECKPOINT_BACKBONE_PREFIX, CHECKPOINT_LM_HEAD_KEY),
            ),
        )
        tokenizer = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
        return cls(config, backbone, audio_decoder, tokenizer).eval()


def iter_checkpoint_tensors(
    checkpoint_dir: Path, *, skip_prefixes: tuple[str, ...]
) -> Iterator[tuple[str, Tensor]]:
    """(name, CPU tensor) of every safetensors tensor not under skip_prefixes."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted({checkpoint_dir / name for name in weight_map.values()})
    else:
        shards = [checkpoint_dir / "model.safetensors"]
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not key.startswith(skip_prefixes):
                    yield key, f.get_tensor(key)


def is_fish_omni_checkpoint(path: str | Path) -> bool:
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return False
    with open(config_path, encoding="utf-8") as f:
        return json.load(f).get("model_type") == FISH_OMNI_MODEL_TYPE
