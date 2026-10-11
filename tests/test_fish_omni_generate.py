import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fish_speech.models.text2semantic import inference
from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni import generate as generate_module
from fish_speech.models.text2semantic.fish_omni.backbone import FishOmniBackbone
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
from fish_speech.models.text2semantic.fish_omni.config import (
    AudioDecoderConfig,
    FishOmniConfig,
    FishOmniTextConfig,
)
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt,
    generate_drama3,
    resolve_drama3_devices,
    resolve_drama3_dtype,
)
from fish_speech.models.text2semantic.fish_omni.layers import ZeroCenteredRMSNorm
from fish_speech.models.text2semantic.fish_omni.model import (
    FishOmniModel,
    is_fish_omni_checkpoint,
)
from fish_speech.models.text2semantic.fish_omni.tower import (
    TOWER_CHECKPOINT_PREFIXES,
    FishOmniAudioDecoder,
    FP32OutputLinear,
    TowerAttention,
    load_audio_decoder_weights,
    sample_top_k_top_p,
)
from fish_speech.models.text2semantic.inference import (
    generate_long,
    init_model,
    resolve_codec_path,
)

PROMPT = torch.tensor([[3, 0, 0, 0], [34, 2, 4, 5], [3, 0, 0, 0], [4, 0, 0, 0]])
SAMPLER = "fish_speech.models.text2semantic.fish_omni.tower.sample_top_k_top_p"


def tiny_omni_config() -> FishOmniConfig:
    text = FishOmniTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        rms_norm_eps=1e-6,
        max_position_embeddings=128,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
    )
    audio_decoder = AudioDecoderConfig(
        dim=16,
        intermediate_size=32,
        n_head=2,
        n_local_heads=1,
        head_dim=8,
        norm_eps=1e-6,
        rope_base=10000.0,
        rotary_percent=0.5,
        vocab_size=8,
        num_codebooks=3,
        shared_enrichment_layers=2,
        vq_backbone_codebooks=1,
    )
    return FishOmniConfig(
        text=text,
        audio_decoder=audio_decoder,
        semantic_start_token_id=32,
        semantic_end_token_id=39,
        eos_token_id=2,
    )


def randomized(module: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.1)
    return module.eval()


def greedy(logits: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return logits.argmax(-1)


class RecordingTokenizer:
    """Records every literal it encodes; each one becomes two tokens."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def token_to_id(self, token: str) -> int | None:
        return {"<|semantic:0|>": 32, "<|semantic:7|>": 39, "<|im_end|>": 2}.get(token)

    def encode(self, text: str) -> SimpleNamespace:
        self.parts.append(text)
        return SimpleNamespace(ids=[3, 4])


def tiny_omni_model() -> FishOmniModel:
    config = tiny_omni_config()
    backbone = randomized(FishOmniBackbone(config.text, dtype=torch.float32))
    tower = randomized(FishOmniAudioDecoder(config, dtype=torch.float32))
    return FishOmniModel(config, backbone, tower, RecordingTokenizer()).eval()


class Drama3PromptTest(unittest.TestCase):
    def test_no_reference_literal_boundaries(self) -> None:
        tokenizer = RecordingTokenizer()
        rows = build_drama3_prompt("hello", tokenizer, tiny_omni_config())
        self.assertEqual(
            tokenizer.parts,
            [
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "<|speaker:0|> hello",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ],
        )
        self.assertEqual(rows.shape, (14, 4))
        self.assertFalse(rows[:, 1:].any())

    def test_reference_codes_and_whitespace(self) -> None:
        tokenizer = RecordingTokenizer()
        codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
        rows = build_drama3_prompt(
            "<|speaker:0|>hello",
            tokenizer,
            tiny_omni_config(),
            ["  <|speaker:0|> reference"],
            [codes],
        )
        self.assertIn(
            "convert the provided text to speech reference to the following:",
            tokenizer.parts,
        )
        self.assertIn(
            "\n\n[Reference 1]\nText: <|speaker:0|>   reference\nSpeech: ",
            tokenizer.parts,
        )
        self.assertIn("<|speaker:0|>hello", tokenizer.parts)
        audio = rows[(rows[:, 0] >= 32) & (rows[:, 0] <= 39)]
        torch.testing.assert_close(audio[:, 1:].T, codes)
        torch.testing.assert_close(audio[:, 0], codes[0] + 32)

    def test_speakers_0_to_15_are_kept_everywhere(self) -> None:
        tokenizer, config = RecordingTokenizer(), tiny_omni_config()
        codes = torch.tensor([[1], [2], [3]])
        text = "<|speaker:15|> hi\n<|speaker:0|> yes"
        build_drama3_prompt(
            text,
            tokenizer,
            config,
            ["<|speaker:2|>ref"],
            [codes],
            history=[("<|speaker:3|> before", codes)],
        )
        self.assertIn(text, tokenizer.parts)
        self.assertIn("<|speaker:3|> before", tokenizer.parts)
        self.assertIn(
            "\n\n[Reference 1]\nText: <|speaker:2|>ref\nSpeech: ", tokenizer.parts
        )
        with self.assertRaisesRegex(ValueError, "history 1"):
            build_drama3_prompt(
                "hi", tokenizer, config, history=[("<|speaker:16|> no", codes)]
            )

    def test_multi_speaker_chunks_pass_validation(self) -> None:
        tokenizer, config = RecordingTokenizer(), tiny_omni_config()
        script = "".join(
            f"<|speaker:{i % 5}|>Line {i} of the scene goes here. " for i in range(30)
        )
        chunks = split_drama3_text(script)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            build_drama3_prompt(chunk, tokenizer, config)
            self.assertLessEqual(len(set(re.findall(r"speaker:(\d+)", chunk))), 3)

    def test_invalid_text_codes_and_tokenizer(self) -> None:
        config = tiny_omni_config()
        for text in (
            " ",
            "hello <|voice|>",
            "<|speaker:16|> hello",
            "<|speaker:00|> hi",
            "<|speaker:01|> hi",
            "<|speaker:x|> hi",
        ):
            with self.assertRaises(ValueError):
                build_drama3_prompt(text, RecordingTokenizer(), config)
        for codes in (
            torch.zeros(2, 3, dtype=torch.long),
            torch.full((3, 2), 8),
            torch.zeros(3, 2),
        ):
            with self.assertRaises(ValueError):
                build_drama3_prompt("hi", RecordingTokenizer(), config, ["ref"], [codes])
        with self.assertRaisesRegex(ValueError, "transcript and codes"):
            build_drama3_prompt("hi", RecordingTokenizer(), config, ["ref"], [])
        config.eos_token_id = 1
        with self.assertRaisesRegex(ValueError, "tokenizer/config"):
            build_drama3_prompt("hello", RecordingTokenizer(), config)


class TowerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.config = tiny_omni_config()
        self.tower = randomized(FishOmniAudioDecoder(self.config, dtype=torch.float32))

    @torch.no_grad()
    def test_q_gate_layout_and_interleaved_rope(self) -> None:
        attention = self.tower.shared_layers[0].transformer_layer.attention
        x, pos = torch.randn(3, 16), torch.arange(3)
        q, gate, k, v = F.linear(x, attention.wqkv.weight).split([16, 16, 8, 8], -1)
        q = attention.q_norm(q.reshape(3, 2, 8))
        k = attention.k_norm(k.reshape(3, 1, 8))
        angles = pos[:, None].float() / 10000.0 ** (torch.arange(2).float()[None] / 2)
        for states in (q, k):
            original = states.clone()
            for pair in range(2):
                c, s = angles[:, pair].cos()[:, None], angles[:, pair].sin()[:, None]
                a, b = original[:, :, 2 * pair], original[:, :, 2 * pair + 1]
                states[:, :, 2 * pair] = a * c - b * s
                states[:, :, 2 * pair + 1] = a * s + b * c
        q, k = q.transpose(0, 1), k.transpose(0, 1).repeat_interleave(2, 0)
        v = v.reshape(3, 1, 8).transpose(0, 1).repeat_interleave(2, 0)
        scores = (q @ k.transpose(-1, -2)) / 8**0.5
        scores.masked_fill_(torch.ones(3, 3, dtype=torch.bool).triu(1), -torch.inf)
        attended = (scores.softmax(-1) @ v).transpose(0, 1).reshape(3, 16)
        expected = F.linear(attended * gate.sigmoid(), attention.wo.weight)
        torch.testing.assert_close(attention(x, pos), expected)

    @torch.no_grad()
    def test_temporal_attention_cached_matches_full(self) -> None:
        layer = self.tower.shared_layers[0]
        hidden, codes = torch.randn(7, 16), torch.arange(7)
        semantic = torch.tensor([False, True, True, False, True, True, True])
        extra = torch.randn(7, 16)
        expected = layer(hidden, codes, semantic, torch.arange(7), extra)
        layer.transformer_layer.attention.setup_cache(12)
        chunks = [
            layer(
                hidden[s:e], codes[s:e], semantic[s:e], torch.arange(s, e), extra[s:e]
            )
            for s, e in [(0, 3), (3, 4), (4, 7)]
        ]
        torch.testing.assert_close(torch.cat(chunks), expected, atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_shifted_prefill_matches_teacher_forced_steps(self) -> None:
        frames = torch.zeros(7, 4, dtype=torch.long)
        frames[2:, 1:] = torch.randint(0, 8, (5, 3))
        frames[:, 0] = 3
        frames[2:, 0] = frames[2:, 1] + 32
        hidden = torch.randn(6, 16)
        attentions = [
            layer.transformer_layer.attention
            for layer in (*self.tower.shared_layers, *self.tower.residual_layers)
        ]
        self.tower.setup_caches(10)
        self.tower.prefill(hidden, torch.arange(6), frames[:-1], frames[1:])
        expected = [attention.k_cache.clone() for attention in attentions]
        self.tower.setup_caches(10)
        for p in range(6):
            self.tower.prefill(
                hidden[p : p + 1],
                torch.tensor([p]),
                frames[p : p + 1],
                frames[p + 1 : p + 2],
            )
        for attention, key in zip(attentions, expected):
            torch.testing.assert_close(attention.k_cache, key, atol=1e-6, rtol=1e-5)

    def test_loader_requires_all_weights_once(self) -> None:
        to_checkpoint = {new: old for old, new in TOWER_CHECKPOINT_PREFIXES.items()}
        weights = []
        for name, tensor in self.tower.named_parameters():
            new, old = next(
                (new, old) for new, old in to_checkpoint.items() if name.startswith(new)
            )
            weights.append((old + name[len(new) :], tensor.detach()))
        target = FishOmniAudioDecoder(self.config, dtype=torch.float32)
        load_audio_decoder_weights(target, weights)
        for actual, expected in zip(target.parameters(), self.tower.parameters()):
            torch.testing.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "not loaded"):
            load_audio_decoder_weights(target, weights[:-1])
        with self.assertRaisesRegex(ValueError, "twice"):
            load_audio_decoder_weights(target, weights + weights[:1])
        invalid = list(weights)
        invalid[0] = (weights[0][0], torch.full_like(weights[0][1], torch.inf))
        with self.assertRaises(FloatingPointError):
            load_audio_decoder_weights(target, invalid)

    def test_sampling_top_p_boundary_and_nonfinite(self) -> None:
        logits = torch.tensor([[0.6, 0.3, 0.1]]).log().expand(100, -1)
        self.assertFalse(sample_top_k_top_p(logits, 1.0, 3, 0.7).any())
        with self.assertRaises(FloatingPointError):
            sample_top_k_top_p(torch.tensor([[float("nan")]]), 1.0, 1, 1.0)


class FP32OutputLinearTest(unittest.TestCase):
    @torch.no_grad()
    def check_wide_output(self, device: str) -> None:
        linear = FP32OutputLinear(8, 4, bias=False, device=device, dtype=torch.float16)
        linear.weight.fill_(100)
        x = torch.full((2, 3, 8), 100.0, dtype=torch.float16, device=device)
        self.assertFalse(torch.isfinite(F.linear(x, linear.weight)).all())
        out = linear(x)
        self.assertEqual(linear.weight.dtype, torch.float16)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, torch.full_like(out, 80000))

    def test_fp16_projection_retains_out_of_range_values(self) -> None:
        self.check_wide_output("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_gemm_with_fp32_output(self) -> None:
        self.check_wide_output("cuda:0")


class Drama3GenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.model = tiny_omni_model()

    def test_chunked_prefill_matches_whole_prefill_and_resets_cache(self) -> None:
        with patch(SAMPLER, side_effect=greedy):
            expected, reason = generate_drama3(
                self.model, PROMPT, max_new_tokens=4, prefill_chunk_size=128
            )
            actual, other_reason = generate_drama3(
                self.model, PROMPT, max_new_tokens=4, prefill_chunk_size=1
            )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(reason, other_reason)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_graph_decode_matches_eager(self) -> None:
        model = self.model.to("cuda:0")
        results = {}
        with patch(SAMPLER, side_effect=greedy):
            for mode in ("0", "1", "1"):
                with patch.dict("os.environ", {"FISH_DRAMA3_CUDA_GRAPH": mode}):
                    frames, reason = generate_drama3(model, PROMPT, max_new_tokens=6)
                results.setdefault(mode, []).append((frames, reason))
        self.assertIsNotNone(model.decode_graphs)
        expected, reason = results["0"][0]
        self.assertGreater(len(expected), 0)
        for frames, other_reason in results["1"]:
            torch.testing.assert_close(frames, expected)
            self.assertEqual(other_reason, reason)

    def test_eos_is_not_an_audio_frame(self) -> None:
        eos_frame = torch.tensor([[2, 0, 0, 0]])
        with patch.object(
            self.model.audio_decoder, "decode_step", return_value=eos_frame
        ) as step:
            frames, reason = generate_drama3(self.model, PROMPT, max_new_tokens=3)
        self.assertEqual((frames.shape, reason, step.call_count), ((0, 4), "eos", 1))
        self.assertEqual(step.call_args.args[2].item(), len(PROMPT) - 1)

    def test_tower_always_applies_zero_centered_qk_norm(self) -> None:
        # Released Drama 3 configs say attention_qk_norm=false, but training ran q/k norm
        # scaled by (1 + weight); a plain RMSNorm or skipping it would both be wrong.
        norm = ZeroCenteredRMSNorm(4, 1e-6, dtype=torch.float32)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([0.0, 0.5, -0.5, 1.0]))
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(norm(x), x / rms * (1 + norm.weight))

        tower = self.model.audio_decoder
        attentions = [m for m in tower.modules() if isinstance(m, TowerAttention)]
        self.assertTrue(attentions)
        for attention in attentions:
            for qk_norm in (attention.q_norm, attention.k_norm):
                self.assertIsInstance(qk_norm, ZeroCenteredRMSNorm)
                self.assertEqual(qk_norm.eps, self.model.config.audio_decoder.norm_eps)
        q_norm, k_norm = attentions[0].q_norm, attentions[0].k_norm
        with (
            patch(SAMPLER, side_effect=greedy),
            patch.object(q_norm, "forward", wraps=q_norm.forward) as q_call,
            patch.object(k_norm, "forward", wraps=k_norm.forward) as k_call,
        ):
            generate_drama3(self.model, PROMPT, max_new_tokens=1)
        self.assertTrue(q_call.called and k_call.called)

    def test_ras_resamples_repeated_audio_tokens_but_not_eos(self) -> None:
        # Main candidates are [eos=2, semantic 32..39]; the prompt's main column holds
        # the semantic token 34 and an eos, both inside the RAS window.
        prompt = torch.tensor([[3, 0, 0, 0], [34, 2, 4, 5], [2, 0, 0, 0], [4, 0, 0, 0]])

        def sampler(first_index: int):
            def sample(logits, temperature, top_k, top_p, check_finite=True):
                if logits.shape[-1] != 9:  # a codebook head
                    return logits.argmax(-1)
                # The RAS retry is the draw made with check_finite=False.
                index = first_index if check_finite else 5  # retry draws token 36
                return torch.full(logits.shape[:-1], index, dtype=torch.long)

            return sample

        with patch(SAMPLER, side_effect=sampler(0)):  # first draw: eos
            frames, reason = generate_drama3(self.model, prompt, max_new_tokens=3)
        self.assertEqual((frames.shape, reason), ((0, 4), "eos"))
        with patch(SAMPLER, side_effect=sampler(3)):  # first draw: 34, a repeat
            frames, reason = generate_drama3(self.model, prompt, max_new_tokens=1)
        self.assertEqual((frames[0, 0].item(), reason), (36, "length"))

    def test_rejects_requests_beyond_the_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit"):
            generate_drama3(self.model, PROMPT, max_new_tokens=128)

    def test_rejects_top_k_below_one(self) -> None:
        for top_k in (0, -1):
            with self.assertRaisesRegex(InvalidInputError, "top_k"):
                generate_drama3(self.model, PROMPT, max_new_tokens=1, top_k=top_k)

    def test_generate_long_dispatches_to_drama3(self) -> None:
        steps = [
            torch.tensor([[33, 1, 2, 3]]),
            torch.tensor([[35, 3, 4, 5]]),
            torch.tensor([[2, 0, 0, 0]]),
        ]
        with patch.object(self.model.audio_decoder, "decode_step", side_effect=steps):
            responses = list(
                generate_long(
                    model=self.model,
                    device="cpu",
                    decode_one_token=None,
                    text="hi",
                    max_new_tokens=5,
                    prompt_text=["ref"],
                    prompt_tokens=[torch.tensor([[1], [2], [3]])],
                )
            )
        self.assertEqual([r.action for r in responses], ["sample", "next"])
        torch.testing.assert_close(
            responses[0].codes, torch.tensor([[1, 3], [2, 4], [3, 5]])
        )
        self.assertIn(
            "\n\n[Reference 1]\nText: <|speaker:0|> ref\nSpeech: ",
            self.model.tokenizer.parts,
        )

    def test_generate_long_clamps_max_new_tokens_to_graph_capacity(self) -> None:
        eos_frame = torch.tensor([[2, 0, 0, 0]])
        for capacity in (40, 1_000_000):  # graph capacity, then the 128-row context
            with (
                patch.object(generate_module, "DRAMA3_GRAPH_CAPACITY", capacity),
                patch.object(
                    self.model.audio_decoder, "decode_step", return_value=eos_frame
                ),
                patch.object(inference, "generate_drama3", wraps=generate_drama3) as run,
            ):
                list(
                    generate_long(
                        model=self.model,
                        device="cpu",
                        decode_one_token=None,
                        text="hi",
                        max_new_tokens=100_000,
                    )
                )
            prompt_rows = len(run.call_args.args[1])
            limit = min(capacity, 128) - prompt_rows
            self.assertEqual(run.call_args.kwargs["max_new_tokens"], limit)
            self.assertLessEqual(self.model.cache_capacity, min(capacity, 128))

    def test_prompt_writes_history_as_completed_turns(self) -> None:
        tokenizer, config = self.model.tokenizer, self.model.config
        codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
        plain = build_drama3_prompt("now", tokenizer, config)
        prompt = build_drama3_prompt("now", tokenizer, config, history=[("before", codes)])
        self.assertEqual(
            tokenizer.parts[-9:],
            [
                "<|im_start|>user\n",
                "<|speaker:0|> before",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                "<|speaker:0|> now",
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ],
        )
        # Five two-token literals plus the two audio frames.
        self.assertEqual(len(prompt) - len(plain), 12)
        audio = torch.tensor([[33, 1, 3, 5], [34, 2, 4, 6]])
        start = int((prompt == audio[0]).all(-1).nonzero()[0])
        torch.testing.assert_close(prompt[start : start + 2], audio)

    def run_chunked(self, chunks: list[str], **kwargs) -> tuple[list, list, list]:
        kwargs.setdefault("max_new_tokens", 0)  # the per-chunk / single_chunk caps
        frame, eos = torch.tensor([[33, 1, 2, 3]]), torch.tensor([[2, 0, 0, 0]])
        with (
            patch.object(inference, "split_drama3_text", return_value=chunks) as split,
            patch.object(inference, "DRAMA3_CHUNK_MAX_NEW_TOKENS", 7),
            patch.object(inference, "DRAMA3_DEFAULT_MAX_NEW_TOKENS", 9),
            patch.object(
                self.model.audio_decoder,
                "decode_step",
                side_effect=[frame, eos] * len(chunks),
            ),
            patch.object(inference, "build_drama3_prompt", wraps=build_drama3_prompt) as build,
            patch.object(inference, "generate_drama3", wraps=generate_drama3) as run,
        ):
            responses = list(
                generate_long(
                    model=self.model,
                    device="cpu",
                    decode_one_token=None,
                    text=" ".join(chunks),
                    **kwargs,
                )
            )
        self.split_calls = split.call_count
        histories = [
            [text for text, _ in call.kwargs["history"]]
            for call in build.call_args_list
            if "history" in call.kwargs
        ]
        budgets = [call.kwargs["max_new_tokens"] for call in run.call_args_list]
        return responses, histories, budgets

    def test_generate_long_drama3_chunks_with_history(self) -> None:
        responses, histories, budgets = self.run_chunked(["one", "two", "three"])
        self.assertEqual([r.action for r in responses], ["sample"] * 3 + ["next"])
        self.assertEqual([r.text for r in responses[:3]], ["one", "two", "three"])
        torch.testing.assert_close(responses[0].codes, torch.tensor([[1], [2], [3]]))
        # The first chunk stays pinned; only the latest other chunk is kept.
        self.assertEqual(histories, [[], ["one"], ["one", "two"]])
        self.assertEqual(budgets, [7, 7, 7])

    def test_single_chunk_generates_the_whole_text_once(self) -> None:
        responses, histories, budgets = self.run_chunked(
            ["one", "two"], single_chunk=True
        )
        self.assertEqual(self.split_calls, 0)
        self.assertEqual([r.action for r in responses], ["sample", "next"])
        self.assertEqual(responses[0].text, "one two")
        self.assertEqual((histories, budgets), ([[]], [9]))

    def test_bad_text_in_a_later_chunk_fails_before_generating(self) -> None:
        chunk_prompt = inference.build_drama3_chunk_prompt
        with patch.object(
            inference, "build_drama3_chunk_prompt", wraps=chunk_prompt
        ) as build:
            with self.assertRaisesRegex(InvalidInputError, "speaker tags"):
                self.run_chunked(["fine", "<|speaker:16|> bad"])
        build.assert_not_called()

    def test_history_is_dropped_when_the_prompt_would_be_too_long(self) -> None:
        rows = len(build_drama3_prompt("x", self.model.tokenizer, self.model.config))
        with patch.object(generate_module, "DRAMA3_MAX_PROMPT_ROWS", rows):
            responses, histories, _ = self.run_chunked(["one", "two"])
        self.assertEqual([r.action for r in responses], ["sample", "sample", "next"])
        self.assertEqual(histories, [[], ["one"], []])


class Drama3DispatchTest(unittest.TestCase):
    def test_device_layout(self) -> None:
        with patch("torch.cuda.device_count", return_value=8):
            self.assertEqual(
                resolve_drama3_devices("cuda"), (["cuda:0", "cuda:1", "cuda:2"], "cuda:3")
            )
            self.assertEqual(resolve_drama3_devices("cuda:5"), (["cuda:5"], "cuda:5"))
            self.assertEqual(
                resolve_drama3_devices("cuda:4, cuda:6"), (["cuda:4"], "cuda:6")
            )
            for bad in ("cpu", "cuda:8", "cuda:0,cuda:0"):
                with self.assertRaises(ValueError):
                    resolve_drama3_devices(bad)
        with patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(resolve_drama3_devices("cuda"), (["cuda:0"], "cuda:1"))

    def test_dtype_falls_back_to_float16_before_sm80(self) -> None:
        with patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            self.assertEqual(
                resolve_drama3_dtype(torch.bfloat16, ["cuda:0"]), torch.float16
            )
        with patch("torch.cuda.get_device_capability", return_value=(8, 0)):
            self.assertEqual(
                resolve_drama3_dtype(torch.bfloat16, ["cuda:0"]), torch.bfloat16
            )
        with self.assertRaises(ValueError):
            resolve_drama3_dtype(torch.float32, [])

    def test_checkpoint_type_and_codec_path(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            s2, drama3 = Path(root, "s2-pro"), Path(root, "drama-3")
            for path, model_type in ((s2, "fish_qwen3_omni"), (drama3, "fish_omni")):
                path.mkdir()
                (path / "config.json").write_text(
                    json.dumps({"model_type": model_type})
                )
            self.assertFalse(is_fish_omni_checkpoint(s2))
            self.assertTrue(is_fish_omni_checkpoint(drama3))
            with self.assertRaises(FileNotFoundError):
                resolve_codec_path(drama3)
            (s2 / "codec.pth").write_bytes(b"")
            self.assertEqual(resolve_codec_path(drama3), s2 / "codec.pth")
            self.assertEqual(resolve_codec_path(s2), s2 / "codec.pth")
            self.assertEqual(resolve_codec_path(drama3, s2 / "codec.pth"), s2 / "codec.pth")
            # A fine-tuned S2-Pro directory without its own codec uses the sibling one.
            finetuned = Path(root, "my-s2-finetune")
            finetuned.mkdir()
            self.assertEqual(resolve_codec_path(finetuned), s2 / "codec.pth")
            with self.assertRaises(FileNotFoundError):
                resolve_codec_path(drama3, Path(root, "missing.pth"))

    def test_init_model_loads_fish_omni(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            Path(root, "config.json").write_text(
                json.dumps({"model_type": "fish_omni"})
            )
            sentinel = object()
            with (
                patch.object(
                    inference,
                    "resolve_drama3_devices",
                    return_value=(["cuda:0"], "cuda:1"),
                ),
                patch.object(inference, "resolve_drama3_dtype", return_value=torch.float16),
                patch.object(
                    FishOmniModel, "from_pretrained", return_value=sentinel
                ) as load,
            ):
                self.assertEqual(
                    init_model(root, "cuda", torch.bfloat16), (sentinel, None)
                )
                with self.assertRaisesRegex(ValueError, "compile"):
                    init_model(root, "cuda", torch.bfloat16, compile=True)
            load.assert_called_once_with(
                root, devices=["cuda:0"], tower_device="cuda:1", dtype=torch.float16
            )


if __name__ == "__main__":
    unittest.main()
