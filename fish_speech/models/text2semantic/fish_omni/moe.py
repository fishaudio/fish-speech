import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.types import Device

from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    FP8_BLOCK_SIZE,
    FP8_DTYPE,
    fp8_moe_forward,
)

# Up to this many (token, expert) pairs run every expert in one batched matmul
# (decode); larger batches loop over the experts they hit (prefill).
GATHERED_EXPERT_MAX_ROWS = 64
FLOAT16_MAX = torch.finfo(torch.float16).max


def dequantize_fp8_blocks(
    weight: Tensor, scale_inv: Tensor, dtype: torch.dtype
) -> Tensor:
    """Block-FP8 [..., out, in] with fp32 scales [..., out/128, in/128] -> dtype."""
    *lead, rows, cols = weight.shape
    row_blocks, col_blocks = scale_inv.shape[-2:]
    blocks = weight.view(*lead, row_blocks, FP8_BLOCK_SIZE, col_blocks, FP8_BLOCK_SIZE)
    scaled = blocks.to(torch.float32) * scale_inv[..., :, None, :, None]
    return scaled.view(*lead, rows, cols).to(dtype)


def clamp_to_float16(tensor: Tensor, name: str) -> Tensor:
    """Saturate the values float16 cannot hold, logging how many there were."""
    overflow = int((tensor.abs() > FLOAT16_MAX).sum())
    if overflow:
        logger.warning(f"{name}: {overflow} values exceed the float16 range; clamped")
        tensor = tensor.clamp(-FLOAT16_MAX, FLOAT16_MAX)
    return tensor


def select_fp8(weight: Tensor, index: Tensor | int) -> Tensor:
    """weight[index] for a float8 tensor; advanced indexing has no float8 kernel."""
    return weight.view(torch.uint8)[index].view(weight.dtype)


class DenseMLP(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        *,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        linear = dict(bias=False, dtype=dtype, device=device)
        self.gate_proj = nn.Linear(hidden, intermediate, **linear)
        self.up_proj = nn.Linear(hidden, intermediate, **linear)
        self.down_proj = nn.Linear(intermediate, hidden, **linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEExperts(nn.Module):
    """Routed experts stacked per layer: w13 [E, 2I, H] (gate then up), w2 [E, H, I].

    With fp8 the weights are float8_e4m3fn buffers with fp32 block scales; on CUDA
    they run on the W8A8 kernel, elsewhere they are dequantized per forward.
    """

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        intermediate: int,
        *,
        fp8: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.fp8 = fp8
        w13_shape = (num_experts, 2 * intermediate, hidden)
        w2_shape = (num_experts, hidden, intermediate)
        if not fp8:
            self.w13_weight = nn.Parameter(
                torch.empty(w13_shape, dtype=dtype, device=device)
            )
            self.w2_weight = nn.Parameter(
                torch.empty(w2_shape, dtype=dtype, device=device)
            )
            return
        if hidden % FP8_BLOCK_SIZE or intermediate % FP8_BLOCK_SIZE:
            raise ValueError(
                f"block-FP8 experts need hidden and intermediate sizes divisible by "
                f"{FP8_BLOCK_SIZE}; got {hidden} and {intermediate}"
            )
        for name, (experts, rows, cols) in (("w13", w13_shape), ("w2", w2_shape)):
            self.register_buffer(
                f"{name}_weight",
                torch.empty((experts, rows, cols), dtype=FP8_DTYPE, device=device),
            )
            self.register_buffer(
                f"{name}_scale_inv",
                torch.empty(
                    (experts, rows // FP8_BLOCK_SIZE, cols // FP8_BLOCK_SIZE),
                    dtype=torch.float32,
                    device=device,
                ),
            )

    def expert_weights(
        self, index: Tensor | int, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        """Gate-up and down weights of the experts index selects, in dtype."""
        if not self.fp8:
            return self.w13_weight[index].to(dtype), self.w2_weight[index].to(dtype)
        return (
            dequantize_fp8_blocks(
                select_fp8(self.w13_weight, index), self.w13_scale_inv[index], dtype
            ),
            dequantize_fp8_blocks(
                select_fp8(self.w2_weight, index), self.w2_scale_inv[index], dtype
            ),
        )

    def forward(self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor) -> Tensor:
        """x [N, H], topk_ids / topk_weights [N, k] -> weighted expert sum [N, H]."""
        if self.fp8 and x.is_cuda:
            return fp8_moe_forward(
                x,
                topk_ids,
                topk_weights,
                self.w13_weight,
                self.w13_scale_inv,
                self.w2_weight,
                self.w2_scale_inv,
            )
        if topk_ids.numel() <= GATHERED_EXPERT_MAX_ROWS:
            return self.forward_gathered(x, topk_ids, topk_weights)
        return self.forward_grouped(x, topk_ids, topk_weights)

    def forward_gathered(
        self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        """One batched matmul over every (token, expert) pair; no host sync."""
        tokens, top_k = topk_ids.shape
        w13, w2 = self.expert_weights(topk_ids.reshape(-1), x.dtype)
        rows = x.repeat_interleave(top_k, dim=0).unsqueeze(-1)  # [M, H, 1]
        gate, up = torch.bmm(w13, rows).squeeze(-1).chunk(2, dim=-1)
        hidden = (F.silu(gate) * up).unsqueeze(-1)
        out = torch.bmm(w2, hidden).squeeze(-1)  # [M, H]
        out = out.float() * topk_weights.reshape(-1, 1).float()
        return out.view(tokens, top_k, self.hidden).sum(dim=1).to(x.dtype)

    def forward_grouped(
        self, x: Tensor, topk_ids: Tensor, topk_weights: Tensor
    ) -> Tensor:
        """Loop over the experts the batch hits, each on its own tokens."""
        tokens, top_k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        token_rows = order // top_k
        pair_weights = topk_weights.reshape(-1)[order]
        counts = torch.bincount(flat_ids, minlength=self.num_experts).tolist()
        out = torch.zeros(tokens, self.hidden, dtype=torch.float32, device=x.device)
        start = 0
        for expert, count in enumerate(counts):
            if count == 0:
                continue
            rows = token_rows[start : start + count]
            weights = pair_weights[start : start + count]
            start += count
            w13, w2 = self.expert_weights(expert, x.dtype)
            gate, up = F.linear(x[rows], w13).chunk(2, dim=-1)
            expert_out = F.linear(F.silu(gate) * up, w2)
            out.index_add_(0, rows, expert_out.float() * weights[:, None].float())
        return out.to(x.dtype)


class SparseMoeBlock(nn.Module):
    """Top-k routed experts plus a sigmoid-gated shared expert."""

    def __init__(
        self,
        config: FishOmniTextConfig,
        *,
        fp8: bool,
        dtype: torch.dtype | None = None,
        device: Device = None,
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        linear = dict(bias=False, dtype=dtype, device=device)
        self.gate = nn.Linear(hidden, config.num_experts, **linear)
        self.experts = MoEExperts(
            config.num_experts,
            hidden,
            config.moe_intermediate_size,
            fp8=fp8,
            dtype=dtype,
            device=device,
        )
        self.shared_expert = DenseMLP(
            hidden, config.shared_expert_intermediate_size, dtype=dtype, device=device
        )
        self.shared_expert_gate = nn.Linear(hidden, 1, **linear)

    def forward(self, x: Tensor) -> Tensor:
        flat = x.reshape(-1, x.shape[-1])
        probabilities = torch.softmax(self.gate(flat).float(), dim=-1)
        topk_weights, topk_ids = probabilities.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        out = self.experts(flat, topk_ids, topk_weights.to(flat.dtype))
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (out + shared_gate * self.shared_expert(flat)).view_as(x)
