import re
import unittest

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from fish_speech.models.text2semantic.fish_omni.backbone import (
    FishOmniBackbone,
    load_backbone_weights,
)
from fish_speech.models.text2semantic.fish_omni.config import FishOmniTextConfig
from fish_speech.models.text2semantic.fish_omni.fp8_moe import (
    FP8_BLOCK_SIZE,
    FP8_DTYPE,
    FP8_MAX,
    fp8_gemm_supported,
    fp8_moe_forward,
)
from fish_speech.models.text2semantic.fish_omni.layers import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from fish_speech.models.text2semantic.fish_omni.moe import (
    MoEExperts,
    dequantize_fp8_blocks,
)

VOCAB_SIZE = 96
SEQUENCE_LENGTH = 150  # crosses two GDN chunk boundaries


def tiny_hf_config() -> Qwen3_5MoeTextConfig:
    config = Qwen3_5MoeTextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        num_hidden_layers=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
    )
    config._attn_implementation = "eager"
    return config


def randomize_fp8_experts(experts: MoEExperts) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill FP8 experts with random weights; returns the dequantized w13 and w2."""
    device = experts.w13_weight.device
    for name in ("w13", "w2"):
        weight = getattr(experts, f"{name}_weight")
        scale = getattr(experts, f"{name}_scale_inv")
        weight.copy_(torch.randn(weight.shape, device=device).to(FP8_DTYPE))
        scale.copy_(torch.rand(scale.shape, device=device) * 0.01)
    return (
        dequantize_fp8_blocks(experts.w13_weight, experts.w13_scale_inv, torch.float32),
        dequantize_fp8_blocks(experts.w2_weight, experts.w2_scale_inv, torch.float32),
    )


def fp8_checkpoint(
    state_dict: dict[str, torch.Tensor],
) -> list[tuple[str, torch.Tensor]]:
    """state_dict with fused HF experts rewritten as per-expert block-FP8 tensors."""
    checkpoint = []
    for name, tensor in state_dict.items():
        match = re.fullmatch(
            r"model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)", name
        )
        if match is None:
            checkpoint.append((name, tensor))
            continue
        for expert, weight in enumerate(tensor):
            if match[2] == "gate_up_proj":
                projections = list(zip(("gate_proj", "up_proj"), weight.chunk(2)))
            else:
                projections = [("down_proj", weight)]
            for projection, matrix in projections:
                rows, cols = matrix.shape
                blocks = matrix.reshape(
                    rows // FP8_BLOCK_SIZE,
                    FP8_BLOCK_SIZE,
                    cols // FP8_BLOCK_SIZE,
                    FP8_BLOCK_SIZE,
                )
                scale = blocks.abs().amax(dim=(1, 3)) / FP8_MAX
                quantized = (blocks / scale[:, None, :, None]).reshape(rows, cols)
                key = f"model.layers.{match[1]}.mlp.experts.{expert}.{projection}"
                checkpoint.append((f"{key}.weight", quantized.to(FP8_DTYPE)))
                checkpoint.append((f"{key}.weight_scale_inv", scale.float()))
    return checkpoint


class BackboneMatchesHuggingFaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        hf_config = tiny_hf_config()
        cls.reference = Qwen3_5MoeForCausalLM(hf_config).eval()
        with torch.no_grad():
            for parameter in cls.reference.parameters():
                parameter.normal_(0.0, 0.1 if parameter.ndim >= 2 else 0.3)
        cls.config = FishOmniTextConfig.from_dict(hf_config.to_dict())
        cls.tokens = torch.randint(0, VOCAB_SIZE, (2, SEQUENCE_LENGTH))
        with torch.no_grad():
            cls.expected = cls.reference(cls.tokens, use_cache=False).logits

    def backbone(self) -> FishOmniBackbone:
        model = FishOmniBackbone(self.config, dtype=torch.float32, device="cpu").eval()
        load_backbone_weights(model, self.reference.state_dict().items())
        return model

    def assert_logits_close(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-4)

    @torch.no_grad()
    def test_whole_sequence_without_cache(self) -> None:
        model = self.backbone()
        logits = model.compute_logits(model(self.tokens))
        self.assert_logits_close(logits, self.expected)

    @torch.no_grad()
    def test_cached_prefill_decode_and_chunked_continuation(self) -> None:
        model = self.backbone()
        model.setup_caches(max_batch_size=2, max_seq_len=256)
        # Prefill, single-token decodes, a multi-token continuation, more decodes.
        segments = [(0, 70), *[(p, p + 1) for p in range(70, 75)], (75, 95)]
        segments += [(p, p + 1) for p in range(95, 98)]
        for start, end in segments:
            hidden = model(
                self.tokens[:, start:end], input_pos=torch.arange(start, end)
            )
            self.assert_logits_close(
                model.compute_logits(hidden), self.expected[:, start:end]
            )

    @torch.no_grad()
    def test_new_sequence_overwrites_cached_state(self) -> None:
        model = self.backbone()
        model.setup_caches(max_batch_size=2, max_seq_len=256)
        model(self.tokens[:, :40].flip(0), input_pos=torch.arange(40))
        hidden = model(self.tokens[:, :30], input_pos=torch.arange(30))
        next_hidden = model(self.tokens[:, 30:31], input_pos=torch.tensor([30]))
        hidden = torch.cat([hidden, next_hidden], dim=1)
        self.assert_logits_close(model.compute_logits(hidden), self.expected[:, :31])

    @torch.no_grad()
    def test_rejects_gaps_and_continuation_without_cache(self) -> None:
        model = self.backbone()
        with self.assertRaisesRegex(AssertionError, "setup_caches"):
            model(self.tokens[:, :2], input_pos=torch.tensor([3, 4]))
        model.setup_caches(max_batch_size=2, max_seq_len=16)
        with self.assertRaisesRegex(AssertionError, "contiguous"):
            model(self.tokens[:, :2], input_pos=torch.tensor([0, 2]))
        with self.assertRaisesRegex(AssertionError, "exceed the caches"):
            model(self.tokens[:, :17], input_pos=torch.arange(17))

    def test_loader_requires_every_slot_once(self) -> None:
        state = list(self.reference.state_dict().items())
        with self.assertRaisesRegex(ValueError, "not loaded"):
            load_backbone_weights(
                FishOmniBackbone(self.config, dtype=torch.float32, device="cpu"),
                [item for item in state if item[0] != "model.norm.weight"],
            )
        with self.assertRaisesRegex(ValueError, "filled twice"):
            load_backbone_weights(
                FishOmniBackbone(self.config, dtype=torch.float32, device="cpu"),
                state + state[:1],
            )


class GatedDeltaRuleTest(unittest.TestCase):
    @torch.no_grad()
    def test_chunked_matches_recurrent_with_initial_state(self) -> None:
        torch.manual_seed(1)
        batch, length, heads, key_dim, value_dim = 2, 133, 3, 8, 6
        query = torch.randn(batch, length, heads, key_dim)
        key = torch.randn(batch, length, heads, key_dim)
        value = torch.randn(batch, length, heads, value_dim)
        decay = -torch.rand(batch, length, heads)
        beta = torch.rand(batch, length, heads)
        state = torch.randn(batch, heads, key_dim, value_dim)
        chunked = chunk_gated_delta_rule(query, key, value, decay, beta, state)
        recurrent = recurrent_gated_delta_rule(query, key, value, decay, beta, state)
        torch.testing.assert_close(chunked[0], recurrent[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(chunked[1], recurrent[1], rtol=1e-4, atol=1e-4)


class Fp8ExpertsTest(unittest.TestCase):
    @torch.no_grad()
    def setUp(self) -> None:
        torch.manual_seed(2)
        hidden, intermediate, count = 2 * FP8_BLOCK_SIZE, FP8_BLOCK_SIZE, 6
        self.experts = MoEExperts(count, hidden, intermediate, fp8=True, device="cpu")
        self.w13, self.w2 = randomize_fp8_experts(self.experts)

    def test_block_dequantization(self) -> None:
        experts = self.experts
        row, col = FP8_BLOCK_SIZE + 3, 2 * FP8_BLOCK_SIZE - 1
        expected = (
            experts.w13_weight[4, row, col].float()
            * experts.w13_scale_inv[4, row // FP8_BLOCK_SIZE, col // FP8_BLOCK_SIZE]
        )
        torch.testing.assert_close(self.w13[4, row, col], expected)

    @torch.no_grad()
    def test_gathered_and_grouped_paths_match_reference(self) -> None:
        x = torch.randn(5, self.experts.hidden)
        topk_ids = torch.tensor([[0, 3], [3, 5], [1, 0], [5, 5], [2, 4]])
        topk_weights = torch.rand(5, 2)
        expected = torch.zeros_like(x)
        for token in range(5):
            for slot in range(2):
                expert = topk_ids[token, slot]
                gate, up = (self.w13[expert] @ x[token]).chunk(2)
                expected[token] += topk_weights[token, slot] * (
                    self.w2[expert] @ (F.silu(gate) * up)
                )
        for path in (self.experts.forward_gathered, self.experts.forward_grouped):
            torch.testing.assert_close(
                path(x, topk_ids, topk_weights), expected, rtol=1e-4, atol=1e-4
            )


@unittest.skipUnless(fp8_gemm_supported(["cuda:0"]), "needs an SM89+ GPU")
class NativeFp8ExpertsTest(unittest.TestCase):
    """The W8A8 kernel against a reference with the same activation quantization."""

    @staticmethod
    def fake_quant(v: torch.Tensor) -> torch.Tensor:
        groups = v.float().reshape(v.shape[0], -1, FP8_BLOCK_SIZE)
        scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-10) / FP8_MAX
        return ((groups / scale).to(FP8_DTYPE).float() * scale).reshape(v.shape)

    @torch.no_grad()
    def test_matches_quantized_reference(self) -> None:
        torch.manual_seed(4)
        device = "cuda:0"
        count, top_k = 16, 4
        hidden, intermediate = 2 * FP8_BLOCK_SIZE, 3 * FP8_BLOCK_SIZE
        experts = MoEExperts(count, hidden, intermediate, fp8=True, device=device)
        w13, w2 = randomize_fp8_experts(experts)
        for tokens in (1, 7, 300):
            x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
            weights, ids = torch.rand(tokens, count, device=device).topk(top_k, dim=-1)
            weights = weights.to(torch.bfloat16)
            x_q = self.fake_quant(x)
            expected = torch.zeros(tokens, hidden, device=device)
            for slot in range(top_k):
                e = ids[:, slot]
                gate_up = torch.einsum("tnk,tk->tn", w13[e], x_q).bfloat16().float()
                gate, up = gate_up.chunk(2, dim=-1)
                h = self.fake_quant(F.silu(gate) * up)
                expected += weights[:, slot, None].float() * torch.einsum(
                    "tnk,tk->tn", w2[e], h
                )
            got = experts(x, ids, weights).float()
            self.assertEqual(got.shape, (tokens, hidden))
            relative = (got - expected).norm() / expected.norm()
            self.assertLess(relative.item(), 1e-2, f"tokens={tokens}")

    @unittest.skipUnless(torch.cuda.device_count() > 1, "needs two GPUs")
    @torch.no_grad()
    def test_runs_on_a_non_current_device(self) -> None:
        device = "cuda:1"
        count, hidden, intermediate = 4, FP8_BLOCK_SIZE, FP8_BLOCK_SIZE
        w13 = torch.randn(count, 2 * intermediate, hidden, device=device)
        w2 = torch.randn(count, hidden, intermediate, device=device)
        s13 = torch.full((count, 2, 1), 0.01, device=device)
        s2 = torch.full((count, 1, 1), 0.01, device=device)
        x = torch.randn(3, hidden, device=device, dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1], [2, 3], [1, 1]], device=device)
        weights = torch.full((3, 2), 0.5, device=device, dtype=torch.bfloat16)
        with torch.cuda.device(0):
            out = fp8_moe_forward(
                x, ids, weights, w13.to(FP8_DTYPE), s13, w2.to(FP8_DTYPE), s2
            )
        self.assertEqual(out.device, torch.device(device))
        self.assertTrue(torch.isfinite(out).all())


class Fp8CheckpointLoadTest(unittest.TestCase):
    @torch.no_grad()
    def test_dense_and_fp8_experts_agree(self) -> None:
        """Per-expert FP8 tensors load into dense (dequantized) and FP8 experts alike."""
        torch.manual_seed(3)
        hf_config = tiny_hf_config()
        hf_config.hidden_size = FP8_BLOCK_SIZE
        hf_config.moe_intermediate_size = FP8_BLOCK_SIZE
        reference = Qwen3_5MoeForCausalLM(hf_config).eval()
        for parameter in reference.parameters():
            parameter.normal_(0.0, 0.1 if parameter.ndim >= 2 else 0.3)
        config = FishOmniTextConfig.from_dict(hf_config.to_dict())
        checkpoint = fp8_checkpoint(reference.state_dict())

        tokens = torch.randint(0, VOCAB_SIZE, (1, 40))
        logits = {}
        for keep_fp8 in (False, True):
            model = FishOmniBackbone(
                config, fp8_experts=keep_fp8, dtype=torch.float32, device="cpu"
            ).eval()
            load_backbone_weights(model, checkpoint)
            logits[keep_fp8] = model.compute_logits(model(tokens))
        torch.testing.assert_close(logits[False], logits[True], rtol=1e-5, atol=1e-5)
        # The reference with its experts replaced by the same dequantized values.
        dequantized = dict(checkpoint)
        for layer_index, layer in enumerate(reference.model.layers):
            experts = layer.mlp.experts
            for expert in range(experts.gate_up_proj.shape[0]):
                key = f"model.layers.{layer_index}.mlp.experts.{expert}"
                weights = {
                    projection: dequantize_fp8_blocks(
                        dequantized[f"{key}.{projection}.weight"],
                        dequantized[f"{key}.{projection}.weight_scale_inv"],
                        torch.float32,
                    )
                    for projection in ("gate_proj", "up_proj", "down_proj")
                }
                experts.gate_up_proj[expert] = torch.cat(
                    (weights["gate_proj"], weights["up_proj"])
                )
                experts.down_proj[expert] = weights["down_proj"]
        expected = reference(tokens, use_cache=False).logits
        torch.testing.assert_close(logits[False], expected, rtol=1e-4, atol=2e-4)

        missing_scale = "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv"
        incomplete = [item for item in checkpoint if item[0] != missing_scale]
        with self.assertRaisesRegex(ValueError, "lack a weight or scale"):
            load_backbone_weights(
                FishOmniBackbone(config, dtype=torch.float32, device="cpu"), incomplete
            )

    def test_float16_keeps_a_float32_residual_stream(self) -> None:
        config = FishOmniTextConfig.from_dict(tiny_hf_config().to_dict())
        with torch.device("meta"):
            half = FishOmniBackbone(config, dtype=torch.float16)
            bfloat = FishOmniBackbone(config, dtype=torch.bfloat16)
        self.assertEqual(half.residual_dtype, torch.float32)
        self.assertEqual(bfloat.residual_dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
