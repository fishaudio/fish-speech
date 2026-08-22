import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from fish_speech.models.text2semantic.inference import generate
from fish_speech.models.text2semantic.llama import (
    Attention,
    BaseModelArgs,
    DualARModelArgs,
    DualARTransformer,
    KVCache,
    NaiveModelArgs,
    NaiveTransformer,
    precompute_freqs_cis,
)

CACHE_CAPACITY = 8
PROMPT_LENGTH = 3


class _TokenizerStub:
    eos_token_id = 15

    def get_token_id(self, _token: str) -> int:
        return self.eos_token_id


def _base_args(**overrides) -> dict:
    args = {
        "vocab_size": 16,
        "n_layer": 1,
        "n_head": 2,
        "n_local_heads": 1,
        "dim": 8,
        "intermediate_size": 16,
        "head_dim": 4,
        "max_seq_len": CACHE_CAPACITY,
        "dropout": 0.0,
        "tie_word_embeddings": False,
        "codebook_size": 8,
        "num_codebooks": 2,
        "semantic_begin_id": 4,
        "semantic_end_id": 11,
        "use_gradient_checkpointing": False,
    }
    args.update(overrides)
    return args


class ActiveKVAttentionTest(unittest.TestCase):
    @staticmethod
    def _attention() -> Attention:
        config = BaseModelArgs(**_base_args())
        attention = Attention(config).eval()
        attention.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_len=CACHE_CAPACITY,
            n_heads=1,
            head_dim=4,
            dtype=torch.float32,
        )
        return attention

    @staticmethod
    def _seed_cache(attention: Attention) -> None:
        torch.manual_seed(100)
        attention.kv_cache.k_cache[:, :, :2] = torch.randn(1, 1, 2, 4)
        attention.kv_cache.v_cache[:, :, :2] = torch.randn(1, 1, 2, 4)

    def test_attention_passes_only_active_kv_prefix_to_sdpa(self) -> None:
        torch.manual_seed(1)
        attention = self._attention()
        self._seed_cache(attention)
        x = torch.randn(1, 1, 8)
        freqs_cis = precompute_freqs_cis(CACHE_CAPACITY, 4)[2:3]
        mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        original_sdpa = F.scaled_dot_product_attention
        observed_shapes = []

        def capture_sdpa(query, key, value, **kwargs):
            observed_shapes.append((key.shape, value.shape))
            return original_sdpa(query, key, value, **kwargs)

        with patch(
            "fish_speech.models.text2semantic.llama.F.scaled_dot_product_attention",
            side_effect=capture_sdpa,
        ):
            attention(x, freqs_cis, mask, input_pos=torch.tensor([2]))

        self.assertEqual(
            observed_shapes,
            [(torch.Size([1, 2, 3, 4]), torch.Size([1, 2, 3, 4]))],
        )
        self.assertEqual(attention.kv_cache.k_cache.shape[2], CACHE_CAPACITY)

    def test_active_prefix_matches_masked_full_cache_with_tolerance(self) -> None:
        torch.manual_seed(2)
        active_attention = self._attention()
        self._seed_cache(active_attention)
        full_attention = copy.deepcopy(active_attention)
        x = torch.randn(1, 1, 8)
        freqs_cis = precompute_freqs_cis(CACHE_CAPACITY, 4)[2:3]
        active_mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        full_mask = torch.zeros(1, 1, 1, CACHE_CAPACITY, dtype=torch.bool)
        full_mask[..., :3] = True
        input_pos = torch.tensor([2])

        active_output = active_attention(
            x,
            freqs_cis,
            active_mask,
            input_pos=input_pos,
        )
        full_output = full_attention(
            x,
            freqs_cis,
            full_mask,
            input_pos=input_pos,
        )

        torch.testing.assert_close(active_output, full_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            active_output.argmax(dim=-1),
            full_output.argmax(dim=-1),
            rtol=0,
            atol=0,
        )


class ActiveKVProgressionTest(unittest.TestCase):
    @staticmethod
    def _model() -> NaiveTransformer:
        torch.manual_seed(3)
        model = NaiveTransformer(NaiveModelArgs(**_base_args())).eval()
        model.setup_caches(1, CACHE_CAPACITY, dtype=torch.float32)
        return model

    @staticmethod
    def _tokens(length: int) -> torch.Tensor:
        main = torch.arange(1, length + 1).remainder(4).view(1, 1, length)
        codebooks = torch.arange(2 * length).remainder(8).view(1, 2, length)
        return torch.cat((main, codebooks), dim=1)

    def test_kv_len_validation(self) -> None:
        model = self._model()
        token = self._tokens(1)
        input_pos = torch.tensor([0])

        minimum_prefix = model.forward_generate(
            token,
            input_pos=input_pos,
            kv_len=1,
        )
        self.assertTrue(torch.isfinite(minimum_prefix.token_logits).all())
        self.assertTrue(torch.isfinite(minimum_prefix.codebook_logits).all())

        for invalid_kv_len in (0, CACHE_CAPACITY + 1):
            with self.subTest(kv_len=invalid_kv_len):
                with self.assertRaisesRegex(
                    ValueError,
                    f"kv_len must be between 1 and {CACHE_CAPACITY}",
                ):
                    model.forward_generate(
                        token,
                        input_pos=input_pos,
                        kv_len=invalid_kv_len,
                    )

    def test_prefill_first_decode_and_maximum_prefix(self) -> None:
        model = self._model()
        original_sdpa = F.scaled_dot_product_attention
        observed_k_lengths = []

        def capture_sdpa(query, key, value, **kwargs):
            observed_k_lengths.append(
                (key.shape[-2], value.shape[-2], kwargs["attn_mask"].shape[-1])
            )
            return original_sdpa(query, key, value, **kwargs)

        with patch(
            "fish_speech.models.text2semantic.llama.F.scaled_dot_product_attention",
            side_effect=capture_sdpa,
        ):
            prefill = model.forward_generate(
                self._tokens(PROMPT_LENGTH),
                input_pos=torch.arange(PROMPT_LENGTH),
                kv_len=PROMPT_LENGTH,
            )
            first_decode = model.forward_generate(
                self._tokens(1),
                input_pos=torch.tensor([PROMPT_LENGTH]),
                kv_len=PROMPT_LENGTH + 1,
            )
            maximum_prefix = model.forward_generate(
                self._tokens(1),
                input_pos=torch.tensor([CACHE_CAPACITY - 1]),
                kv_len=CACHE_CAPACITY,
            )

        self.assertEqual(
            observed_k_lengths,
            [
                (PROMPT_LENGTH, PROMPT_LENGTH, PROMPT_LENGTH),
                (PROMPT_LENGTH + 1, PROMPT_LENGTH + 1, PROMPT_LENGTH + 1),
                (CACHE_CAPACITY, CACHE_CAPACITY, CACHE_CAPACITY),
            ],
        )
        for result in (prefill, first_decode, maximum_prefix):
            self.assertTrue(torch.isfinite(result.token_logits).all())
            self.assertTrue(torch.isfinite(result.codebook_logits).all())


class ActiveKVAutoregressiveDecodeTest(unittest.TestCase):
    @staticmethod
    def _model() -> DualARTransformer:
        torch.manual_seed(4)
        config = DualARModelArgs(
            **_base_args(
                n_fast_layer=1,
                fast_dim=8,
                fast_n_head=2,
                fast_n_local_heads=1,
                fast_head_dim=4,
                fast_intermediate_size=16,
            )
        )
        model = DualARTransformer(config).eval()
        model.tokenizer = _TokenizerStub()
        return model

    @staticmethod
    def _prompt() -> torch.Tensor:
        return torch.tensor(
            [
                [1, 2, 3],
                [0, 1, 2],
                [3, 4, 5],
            ],
            dtype=torch.long,
        )

    @staticmethod
    def _run_generation(model, *, use_active_prefix):
        slow_logits = []
        fast_logits = []
        observed_kv_lengths = []
        original_forward = model.forward_generate
        original_fast_forward = model.forward_generate_fast

        def capture_forward(*args, **kwargs):
            observed_kv_lengths.append(kwargs.get("kv_len"))
            if not use_active_prefix:
                kwargs["kv_len"] = None
            result = original_forward(*args, **kwargs)
            result.logits[..., model.tokenizer.eos_token_id] = -1e9
            slow_logits.append(result.logits.detach().clone())
            return result

        def capture_fast_forward(*args, **kwargs):
            logits = original_fast_forward(*args, **kwargs)
            fast_logits.append(logits.detach().clone())
            return logits

        with (
            patch.object(model, "forward_generate", side_effect=capture_forward),
            patch.object(
                model,
                "forward_generate_fast",
                side_effect=capture_fast_forward,
            ),
            patch(
                "fish_speech.models.text2semantic.inference.tqdm",
                side_effect=lambda values, **_kwargs: values,
            ),
        ):
            output = generate(
                model=model,
                prompt=ActiveKVAutoregressiveDecodeTest._prompt(),
                max_new_tokens=3,
                audio_masks=None,
                audio_parts=None,
                temperature=0.8,
                top_p=0.8,
                top_k=1,
            )

        return output, slow_logits, fast_logits, observed_kv_lengths

    def test_real_decode_chain_preserves_greedy_tokens_and_distributions(self) -> None:
        active_model = self._model()
        full_cache_model = copy.deepcopy(active_model)

        torch.manual_seed(5)
        active = self._run_generation(active_model, use_active_prefix=True)
        torch.manual_seed(5)
        full = self._run_generation(full_cache_model, use_active_prefix=False)
        active_output, active_slow, active_fast, active_kv_lengths = active
        full_output, full_slow, full_fast, full_kv_lengths = full

        expected_shape = (1 + active_model.config.num_codebooks, PROMPT_LENGTH + 3)
        self.assertEqual(active_output.shape, expected_shape)
        self.assertEqual(full_output.shape, expected_shape)
        self.assertEqual(active_kv_lengths, [PROMPT_LENGTH, 4, 5])
        self.assertEqual(full_kv_lengths, active_kv_lengths)

        self.assertEqual(len(active_slow), len(full_slow))
        self.assertEqual(len(active_fast), len(full_fast))
        for active_logits, full_logits in zip(
            active_slow + active_fast,
            full_slow + full_fast,
        ):
            self.assertTrue(torch.isfinite(active_logits).all())
            self.assertTrue(torch.isfinite(full_logits).all())
            active_probs = torch.softmax(active_logits, dim=-1)
            full_probs = torch.softmax(full_logits, dim=-1)
            torch.testing.assert_close(active_probs, full_probs, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                active_probs.argmax(dim=-1),
                full_probs.argmax(dim=-1),
                rtol=0,
                atol=0,
            )

        torch.testing.assert_close(active_output, full_output, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
