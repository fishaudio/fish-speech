from collections.abc import Iterable

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton ships only with the CUDA / ROCm builds of torch.
    triton = None

FP8_DTYPE = torch.float8_e4m3fn
# Weight block edge of bsgemm-moe checkpoints; also the activation quant group width.
FP8_BLOCK_SIZE = 128
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
FP8_MIN_SCALE = 1e-10
GEMM_BLOCK_N = 128
GEMM_NUM_WARPS = 4
GEMM_NUM_STAGES = 3


def fp8_gemm_supported(devices: Iterable[str | torch.device]) -> bool:
    """Every device is a CUDA SM89+ GPU (FP8 tensor cores) and Triton is available.

    ROCm also reports its GPUs as "cuda" with a capability, but the kernels use the
    e4m3fn format, which AMD GPUs lack (MI300 uses fnuz, RDNA3 has no FP8).
    """
    if triton is None or not torch.cuda.is_available() or torch.version.hip:
        return False
    devices = [torch.device(device) for device in devices]
    return all(
        device.type == "cuda" and torch.cuda.get_device_capability(device) >= (8, 9)
        for device in devices
    )


if triton is not None:

    @triton.jit
    def quant_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        stride_x,
        cols,
        fp8_max: tl.constexpr,
        min_scale: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """Dynamic FP8 quantization of one (row, channel group)."""
        row = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        offs = group * GROUP + tl.arange(0, GROUP)
        mask = offs < cols
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x)), min_scale) / fp8_max
        q = tl.clamp(x / scale, -fp8_max, fp8_max)
        tl.store(out_ptr + row * cols + offs, q.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row * tl.num_programs(1) + group, scale)

    @triton.jit
    def silu_mul_quant_kernel(
        x_ptr,
        out_ptr,
        scale_ptr,
        stride_x,
        half,
        fp8_max: tl.constexpr,
        min_scale: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        """silu(gate) * up of x = [gate | up], then FP8 quantization per group."""
        row = tl.program_id(0).to(tl.int64)
        group = tl.program_id(1)
        offs = group * GROUP + tl.arange(0, GROUP)
        mask = offs < half
        gate = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
        up = tl.load(x_ptr + row * stride_x + half + offs, mask=mask, other=0.0)
        gate, up = gate.to(tl.float32), up.to(tl.float32)
        y = gate * tl.sigmoid(gate) * up
        scale = tl.maximum(tl.max(tl.abs(y)), min_scale) / fp8_max
        q = tl.clamp(y / scale, -fp8_max, fp8_max)
        tl.store(out_ptr + row * half + offs, q.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(scale_ptr + row * tl.num_programs(1) + group, scale)

    @triton.jit
    def fused_moe_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        a_scale_ptr,
        b_scale_ptr,
        topk_weights_ptr,
        sorted_ids_ptr,
        expert_ids_ptr,
        num_post_padded_ptr,
        N,
        K,
        num_valid,
        stride_am,
        stride_be,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_asm,
        stride_bse,
        stride_bsn,
        stride_bsk,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        TOP_K: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
    ):
        """C[pair] = A[pair // TOP_K] @ B[expert(pair)]^T for one output tile.

        Rows are (token, slot) pairs sorted by expert, BLOCK_M rows per expert;
        padding rows hold num_valid and are masked out.
        """
        pid = tl.program_id(0)
        EM = tl.load(num_post_padded_ptr)
        num_pid_m = tl.cdiv(EM, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        # The grid is sized for the worst case; blocks past the real layout exit.
        if pid >= num_pid_m * num_pid_n:
            return
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        pair = tl.load(sorted_ids_ptr + offs_row).to(tl.int64)
        pair_mask = pair < num_valid
        expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N
        offs_k = tl.arange(0, BLOCK_K)
        token = pair // TOP_K
        a_ptrs = a_ptr + token[:, None] * stride_am + offs_k[None, :]
        b_ptrs = (
            b_ptr
            + expert * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        a_scale_ptrs = a_scale_ptr + token * stride_asm
        b_scale_ptrs = (
            b_scale_ptr + expert * stride_bse + (offs_n // GROUP) * stride_bsn
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=pair_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
            k_group = (k * BLOCK_K) // GROUP
            a_scale = tl.load(a_scale_ptrs + k_group, mask=pair_mask, other=0.0)
            b_scale = tl.load(b_scale_ptrs + k_group * stride_bsk)
            acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            w = tl.load(topk_weights_ptr + pair, mask=pair_mask, other=0.0)
            acc = acc * w.to(tl.float32)[:, None]
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + pair[:, None] * stride_cm + offs_cn[None, :]
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=pair_mask[:, None] & (offs_cn[None, :] < N),
        )


def quantize_groups(
    kernel: "triton.JITFunction", x: Tensor, cols: int
) -> tuple[Tensor, Tensor]:
    """Run a per-token-group quant kernel: FP8 [M, cols] + fp32 scales [M, groups]."""
    rows = x.shape[0]
    groups = triton.cdiv(cols, FP8_BLOCK_SIZE)
    quantized = torch.empty((rows, cols), dtype=FP8_DTYPE, device=x.device)
    scales = torch.empty((rows, groups), dtype=torch.float32, device=x.device)
    kernel[(rows, groups)](
        x,
        quantized,
        scales,
        x.stride(0),
        cols,
        fp8_max=FP8_MAX,
        min_scale=FP8_MIN_SCALE,
        GROUP=FP8_BLOCK_SIZE,
        num_warps=1,
    )
    return quantized, scales


def moe_align(
    topk_ids: Tensor, num_experts: int, block_m: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Sort (token, slot) pairs by expert, padding each expert to a multiple of block_m.

    Stays on the device (no host sync). Returns sorted_ids [max_rows] (padding rows
    hold the pair count), expert_ids [max_rows / block_m] and num_post_padded [1].
    """
    flat = topk_ids.reshape(-1).to(torch.int64)
    pairs = flat.numel()
    device = flat.device
    # At most min(pairs, experts) experts are hit, each padded by < block_m rows.
    hit_bound = min(pairs, num_experts)
    max_rows = triton.cdiv(pairs + hit_bound * (block_m - 1), block_m) * block_m

    # bincount syncs with the host on CUDA; scatter_add does not.
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + block_m - 1) // block_m * block_m
    ends = padded.cumsum(0)
    starts = ends - padded
    count_starts = counts.cumsum(0) - counts
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    rank = torch.arange(pairs, device=device) - count_starts[sorted_experts]
    sorted_ids = torch.full((max_rows,), pairs, dtype=torch.int32, device=device)
    sorted_ids[starts[sorted_experts] + rank] = order.to(torch.int32)

    block_starts = torch.arange(0, max_rows, block_m, device=device)
    expert_ids = torch.searchsorted(ends, block_starts, right=True)
    expert_ids = expert_ids.clamp_(max=num_experts - 1).to(torch.int32)
    return sorted_ids, expert_ids, ends[-1:].to(torch.int32)


def fused_moe_gemm(
    a: Tensor,
    a_scale: Tensor,
    b: Tensor,
    b_scale: Tensor,
    out: Tensor,
    topk_weights: Tensor | None,
    sorted_ids: Tensor,
    expert_ids: Tensor,
    num_post_padded: Tensor,
    *,
    top_k: int,
    block_m: int,
    group_m: int,
) -> None:
    _, n, k = b.shape
    grid = (triton.cdiv(sorted_ids.numel(), block_m) * triton.cdiv(n, GEMM_BLOCK_N),)
    fused_moe_kernel[grid](
        a,
        b,
        out,
        a_scale,
        b_scale,
        # Any valid pointer; the kernel only reads it with MUL_ROUTED_WEIGHT.
        a_scale if topk_weights is None else topk_weights,
        sorted_ids,
        expert_ids,
        num_post_padded,
        n,
        k,
        a.shape[0] * top_k,
        a.stride(0),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        out.stride(0),
        a_scale.stride(0),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        GROUP=FP8_BLOCK_SIZE,
        BLOCK_M=block_m,
        BLOCK_N=GEMM_BLOCK_N,
        BLOCK_K=FP8_BLOCK_SIZE,
        GROUP_M=group_m,
        TOP_K=top_k,
        MUL_ROUTED_WEIGHT=topk_weights is not None,
        num_warps=GEMM_NUM_WARPS,
        num_stages=GEMM_NUM_STAGES,
    )


def fp8_moe_forward(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    w13_weight: Tensor,
    w13_scale_inv: Tensor,
    w2_weight: Tensor,
    w2_scale_inv: Tensor,
) -> Tensor:
    """Route x [N, H] through block-FP8 experts (W8A8) -> weighted expert sum [N, H].

    w13_weight [E, 2I, H] holds gate rows then up rows, w2_weight [E, H, I]; both
    float8_e4m3fn with fp32 [E, rows / 128, cols / 128] block scales.
    """
    tokens, hidden = x.shape
    if tokens == 0:
        return torch.zeros_like(x)
    num_experts, gate_up_rows, _ = w13_weight.shape
    top_k = topk_ids.shape[-1]
    pairs = tokens * top_k
    # Decode routes a handful of rows per expert: smaller M tiles waste less.
    if pairs <= 64:
        block_m, group_m = 16, 1
    elif pairs <= 1024:
        block_m, group_m = 32, 1
    else:
        block_m, group_m = 64, 8

    # Triton launches on the current device; layers may live on another GPU.
    with torch.cuda.device(x.device):
        sorted_ids, expert_ids, num_post_padded = moe_align(
            topk_ids, num_experts, block_m
        )
        layout = dict(top_k=top_k, block_m=block_m, group_m=group_m)
        x_q, x_scale = quantize_groups(quant_kernel, x.contiguous(), hidden)
        gate_up = torch.empty((pairs, gate_up_rows), dtype=x.dtype, device=x.device)
        fused_moe_gemm(
            x_q,
            x_scale,
            w13_weight,
            w13_scale_inv,
            gate_up,
            None,
            sorted_ids,
            expert_ids,
            num_post_padded,
            **layout,
        )
        h_q, h_scale = quantize_groups(
            silu_mul_quant_kernel, gate_up, gate_up_rows // 2
        )
        down = torch.empty((pairs, hidden), dtype=torch.float32, device=x.device)
        fused_moe_gemm(
            h_q,
            h_scale,
            w2_weight,
            w2_scale_inv,
            down,
            topk_weights.reshape(-1).contiguous(),
            sorted_ids,
            expert_ids,
            num_post_padded,
            top_k=1,
            block_m=block_m,
            group_m=group_m,
        )
    return down.view(tokens, top_k, hidden).sum(dim=1).to(x.dtype)
