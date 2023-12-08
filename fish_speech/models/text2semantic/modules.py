import math
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    return torch.view_as_real(x_ * freqs_cis).flatten(3).type_as(x)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, is_cross_attention=False):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.is_cross_attention = is_cross_attention

        # Auto fuse linear projection
        if is_cross_attention:
            self.q_proj = nn.Linear(d_model, d_model)
            self.kv_proj = nn.Linear(d_model, d_model * 2)
        else:
            self.qkv_proj = nn.Linear(d_model, d_model * 3)

        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        freqs_cis_q,
        kv=None,
        freqs_cis_kv=None,
        attn_mask=None,
        input_pos=None,
        kv_cache=None,
    ):
        if self.is_cross_attention:
            q = self.q_proj(q)
            if kv is None:
                assert self.kv_cache is not None, "kv_cache should be initialized"
                k, v = None
            else:
                # Using kv cache
                kv = self.kv_proj(kv)
                k, v = torch.chunk(kv, 2, dim=-1)
        else:
            assert kv is None, f"kv should be None for self attention"
            assert (
                freqs_cis_kv is None
            ), f"freqs_cis_kv should be None for self attention"
            q, k, v = torch.chunk(self.qkv_proj(q), 3, dim=-1)

        # max_batch_size, max_seq_length, n_heads, head_dim
        q = rearrange(q, "b t (h d) -> b t h d", h=self.nhead, d=self.head_dim)
        q = apply_rotary_emb(q, freqs_cis_q)

        if freqs_cis_kv is None:
            freqs_cis_kv = freqs_cis_q

        # Only do when self attention or cross attention without kv cache
        if k is not None:
            assert v is not None, "v should not be None when k is not None"
            k = rearrange(k, "b t (h d) -> b t h d", h=self.nhead, d=self.head_dim)
            v = rearrange(v, "b t (h d) -> b t h d", h=self.nhead, d=self.head_dim)
            k = apply_rotary_emb(k, freqs_cis_kv)

        if kv_cache is not None:
            if k is None:
                assert v is None, "v should be None when k is None"
                k, v = kv_cache[0], kv_cache[1]
            else:
                k = torch.cat([kv_cache[0], k], dim=1)
                v = torch.cat([kv_cache[1], v], dim=1)
                kv_cache = (k, v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        value = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0,
        )

        value = rearrange(value, "b h t d -> b t (h d)")
        return self.o_proj(value), kv_cache


class GluMLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=None, activation=nn.SiLU):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size * (11 / 3)
            intermediate_size = round(intermediate_size / 8) * 8

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = activation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=None, nhead=16, dropout=0.1):
        super().__init__()

        self.attention = MultiheadAttention(hidden_size, nhead, dropout=dropout)
        self.ffn = GluMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.attention_norm = RMSNorm(hidden_size, eps=1e-6)
        self.ffn_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        x,
        freqs_cis,
        attn_mask=None,
        input_pos=None,
    ):
        x = (
            x
            + self.attention(
                q=self.attention_norm(x),
                freqs_cis_q=freqs_cis,
                attn_mask=attn_mask,
                input_pos=input_pos,
            )[0]
        )

        return x + self.ffn(self.ffn_norm(x))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=None, nhead=16, dropout=0.1):
        super().__init__()

        self.self_attention = MultiheadAttention(hidden_size, nhead, dropout=dropout)
        self.cross_attention = MultiheadAttention(
            hidden_size, nhead, dropout=dropout, is_cross_attention=True
        )
        self.ffn = GluMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

        self.self_attention_norm = RMSNorm(hidden_size, eps=1e-6)
        self.cross_attention_norm = RMSNorm(hidden_size, eps=1e-6)
        self.ffn_norm = RMSNorm(hidden_size, eps=1e-6)

    def forward(
        self,
        x,
        context,
        freqs_cis_q,
        freqs_cis_kv,
        self_attn_mask=None,
        cross_attn_mask=None,
        input_pos=None,
    ):
        x = x + self.self_attention(
            q=self.self_attention_norm(x),
            freqs_cis_q=freqs_cis_q,
            attn_mask=self_attn_mask,
            input_pos=input_pos,
        )

        x = x + self.cross_attention(
            q=self.cross_attention_norm(x),
            kv=context,
            freqs_cis_q=freqs_cis_q,
            freqs_cis_kv=freqs_cis_kv,
            attn_mask=cross_attn_mask,
            input_pos=input_pos,
        )

        return x + self.ffn(self.ffn_norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        codebook_size,
        num_codebooks,
        hidden_size=1024,
        intermediate_size=None,
        nhead=16,
        num_encoder_layers=12,
        num_decoder_layers=12,
        dropout=0.1,
        max_position=4096,
    ):
        super().__init__()

        self.encoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks)]
        )
        self.decoder_head = nn.Linear(hidden_size, codebook_size * num_codebooks)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.nhead = nhead

        self.encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(hidden_size // nhead, max_position, theta=10000.0),
        )

        causual_mask = torch.triu(
            torch.ones(max_position, max_position), diagonal=1
        ).bool()
        causual_mask = torch.zeros(max_position, max_position).masked_fill(
            causual_mask, float("-inf")
        )

        self.register_buffer("causual_mask", causual_mask)

        # The following are reserved for kv cache
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_kv_caches(self, max_batch_size, max_seq_length):
        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return

        if max_seq_length % 8 != 0:
            max_seq_length = max_seq_length + (8 - max_seq_length % 8)

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.decoder:
            b.self_attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                b.self_attention.nhead,
                b.self_attention.head_dim,
            ).to(b.self_attention_norm.weight.device)

            b.cross_attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                b.cross_attention.nhead,
                b.cross_attention.head_dim,
            ).to(b.cross_attention_norm.weight.device)

    def get_key_padding_mask(self, key_padding_mask, q_size=None):
        # inputs: (B, T) bool ->
        assert key_padding_mask.dtype == torch.bool and key_padding_mask.ndim == 2

        key_padding_mask = (
            key_padding_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.nhead, -1, -1)
        )

        key_padding_mask = key_padding_mask.reshape(
            key_padding_mask.shape[0], self.nhead, 1, key_padding_mask.shape[1]
        )

        if q_size is not None:
            key_padding_mask = key_padding_mask.expand(-1, -1, q_size, -1)

        new_mask = torch.zeros(
            *key_padding_mask.shape, dtype=torch.float, device=key_padding_mask.device
        )
        new_mask = new_mask.masked_fill(key_padding_mask, float("-inf"))

        return new_mask

    def forward_encoder(
        self, inputs, input_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # inputs: (B, T)
        # input_mask: (B, T), bool mask
        inputs = self.encoder_embedding(inputs)

        # Calculate mask
        if input_mask is None:
            # Assume no padding
            input_mask = torch.zeros(
                inputs.shape[0], inputs.shape[1], dtype=torch.bool, device=inputs.device
            )

        input_mask = self.get_key_padding_mask(input_mask, q_size=None).to(inputs.dtype)

        freqs_cis = self.freqs_cis[: inputs.shape[1]]
        input_mask_self = input_mask.expand(-1, -1, inputs.shape[1], -1)

        for layer in self.encoder:
            inputs = layer(inputs, freqs_cis=freqs_cis, attn_mask=input_mask_self)

        return inputs, input_mask

    def forward_decoder(
        self, codes, inputs, input_mask, codes_mask=None, input_pos=None
    ):
        # codes: (B, C, T)
        # inputs: (B, T, N)

        print(f"Codes: {codes.shape}, Inputs: {inputs.shape}")
        codes = rearrange(codes, "b c t -> c b t")
        codes = torch.stack(
            [emb(code) for emb, code in zip(self.decoder_embeddings, codes)], dim=0
        )
        codes = torch.mean(codes, dim=0)  # (B, T)

        # If kv cache is enabled
        input_mask = input_mask.expand(-1, -1, codes.shape[1], -1)

        # Calculate mask
        if input_pos is not None:
            attn_mask = self.causual_mask[: codes.shape[1], : codes.shape[1]]
        else:
            attn_mask = None

        # if codes_mask is not None:
        #     codes_mask = self.get_key_padding_mask(codes_mask)
        #     attn_mask = attn_mask + codes_mask

        # For kv cache
        if input_pos is not None:
            freqs_cis_q = self.freqs_cis[input_pos]
        else:
            freqs_cis_q = self.freqs_cis[: codes.shape[1]]

        freqs_cis_kv = self.freqs_cis[: inputs.shape[1]]

        for layer in self.decoder:
            codes = layer(
                codes,
                inputs,
                freqs_cis_q=freqs_cis_q,
                freqs_cis_kv=freqs_cis_kv,
                self_attn_mask=attn_mask,
                cross_attn_mask=input_mask,
                input_pos=input_pos,
            )

        codes = self.decoder_head(codes)
        codes = rearrange(
            codes, "b t (c d) -> b c t d", c=self.num_codebooks, d=self.codebook_size
        )

        return codes

    def forward(
        self,
        inputs,
        codes,
        input_mask=None,
        codes_mask=None,
        input_pos=None,
    ):
        # inputs: (B, T)
        # codes: (B, C, T)
        # input_mask: (B, T), bool mask
        # codes_mask: (B, T), bool mask
        # input_pos: (B, T), int mask

        inputs, input_mask = self.forward_encoder(inputs, input_mask)
        codes = self.forward_decoder(codes, inputs, input_mask, codes_mask, input_pos)

        return codes


if __name__ == "__main__":
    mha = MultiheadAttention(512, 8, dropout=0, is_cross_attention=True)
    mha.eval()
    mha.cuda()

    q, kv = torch.randn(2, 10, 16, 512)
    q, kv = q.cuda(), kv.cuda()

    mha.bfloat16()
    q, kv = q.bfloat16(), kv.bfloat16()
    freqs_cis = precompute_freqs_cis(512 // 8, 4096 * 2).cuda()[:16]

    # Causual mask
    attn_mask = torch.triu(torch.ones(16, 16), diagonal=1).bool().cuda()
    o = mha(q, freqs_cis, kv=kv, attn_mask=attn_mask)

    trans = (
        Transformer(
            vocab_size=30000,
            codebook_size=120,
            num_codebooks=4,
            hidden_size=1024,
            intermediate_size=None,
            nhead=16,
            num_encoder_layers=12,
            num_decoder_layers=12,
        )
        .bfloat16()
        .cuda()
    )
    trans.eval()

    # Print n param
    print("Total params:", sum(i.numel() for i in trans.parameters()) / 1024 / 1024)
    inputs = torch.randint(0, 1000, (2, 16)).cuda()
    codes = torch.randint(0, 120, (2, 4, 128)).cuda()
    x = trans(inputs, codes)
    x1 = trans(inputs, codes)

    assert torch.allclose(x, x1, atol=1e-4, rtol=1e-3), "Model is not deterministic"
    print("Model is deterministic")

    # Test kv cache
    trans.setup_kv_caches(2, 1024)
    inputs, inputs_mask = trans.forward_encoder(inputs)

    outputs = []

    for i in range(128):
        code = codes[..., i].unsqueeze(-1)
        code_mask = torch.tensor([[1], [1]], dtype=torch.bool, device=code.device)
        input_pos = torch.tensor([i], dtype=torch.long, device=code.device)
        outputs.append(
            trans.forward_decoder(
                code, inputs, inputs_mask, code_mask, input_pos=input_pos
            )
        )

    outputs = torch.cat(outputs, dim=2)
    print(x.shape, outputs.shape)
    assert torch.allclose(x, outputs, atol=1e-4, rtol=1e-3), "KV cache is not working"
