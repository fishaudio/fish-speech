import math

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

try:
    from xformers.ops import memory_efficient_attention
except ImportError as e:
    memory_efficient_attention = None
# memory_efficient_attention = None


class AlibiPostionEmbedding:
    def __init__(self, nheads, maxpos):
        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]
        relative_position = memory_position - context_position
        relative_position = (
            torch.abs(relative_position).unsqueeze(0).expand(nheads, -1, -1)
        )
        self.slopes = torch.Tensor(self.get_slopes(nheads)) * -1
        self.alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
        self.alibi = self.alibi.view(nheads, maxpos, maxpos)

    @staticmethod
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    def get_slopes(self, n):
        if math.log2(n).is_integer():
            return self.get_slopes_power_of_2(n)

        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            self.get_slopes_power_of_2(closest_power_of_2)
            + self.get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )

    def __call__(self, x):
        # N, T, C
        return self.alibi[:, : x.size(1), : x.size(1)].to(x.device)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        past_kv=None,
        return_weights=False,
    ):
        # (B, T, C)
        batch_size = q.size(0)
        q_length = q.size(1)
        k_length = k.size(1)

        if past_kv is not None:
            k, v = torch.cat([past_kv, k], 1), torch.cat([past_kv, v], 1)

        if attn_bias is not None:
            assert attn_bias.size() == (
                self.nhead,
                q_length,
                k_length,
            ), f"Should be {(self.nhead, q_length, k_length)}. Got {attn_bias.size()}"

            attn_bias = attn_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        if attn_mask is not None:
            assert attn_mask.size() == (
                q_length,
                k_length,
            ), f"Should be {(q_length, k_length)}. Got {attn_mask.size()}"
            assert attn_mask.dtype == torch.bool
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size * self.nhead, -1, -1)

        if key_padding_mask is not None:
            assert key_padding_mask.size() == (
                batch_size,
                k_length,
            ), f"Should be {(batch_size, k_length)}. Got {key_padding_mask.size()}"
            assert key_padding_mask.dtype == torch.bool
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, self.nhead, -1, -1)
            )
            key_padding_mask = key_padding_mask.reshape(
                batch_size * self.nhead, 1, k_length
            )
            if attn_mask is None:
                attn_mask = key_padding_mask.expand(-1, q.size(1), -1)
            else:
                attn_mask = attn_mask.logical_or(key_padding_mask)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        if (
            return_weights is False
            and memory_efficient_attention is not None
            and q.device.type == "cuda"
        ):
            # (-> b, t,. n, d)
            q = rearrange(q, "b t (n d) -> b t n d", n=self.nhead)
            k = rearrange(k, "b t (n d) -> b t n d", n=self.nhead)
            v = rearrange(v, "b t (n d) -> b t n d", n=self.nhead)

            if attn_mask is not None:
                attn_mask = rearrange(attn_mask, "(b n) q k -> b n q k", n=self.nhead)
                attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))

            attn_output = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                scale=self.head_dim**-0.5,
                p=self.dropout.p,
            )
            attn_output = rearrange(attn_output, "b t n d -> b t (n d)", n=self.nhead)

            returned_weights = None
        else:
            q = rearrange(q, "b t (n d) -> (b n) t d", n=self.nhead)
            k = rearrange(k, "b t (n d) -> (b n) t d", n=self.nhead)
            v = rearrange(v, "b t (n d) -> (b n) t d", n=self.nhead)

            attn_weights = torch.bmm(q, k.mT) * (self.head_dim**-0.5)
            assert attn_weights.size() == (
                batch_size * self.nhead,
                q.size(1),
                k.size(1),
            )

            if attn_bias is not None:
                attn_bias = rearrange(attn_bias, "b n q k -> (b n) q k")
                attn_weights = attn_weights + attn_bias

            if attn_mask is not None:
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=attn_weights.dtype)
            returned_weights = attn_weights.view(
                batch_size, self.nhead, q.size(1), k.size(1)
            )

            attn_probs = self.dropout(attn_weights)
            attn_output = torch.bmm(attn_probs, v)
            attn_output = rearrange(attn_output, "(b n) t d -> b t (n d)", n=self.nhead)

        attn_output = self.out_proj(attn_output)
        return attn_output, returned_weights


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


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=None, dropout=0.1):
        super().__init__()

        self.attn = MultiheadAttention(hidden_size, 1, dropout=dropout)
        self.mlp = GluMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm_q = RMSNorm(hidden_size, eps=1e-6)
        self.input_layernorm_kv = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, tgt, memory, memory_key_padding_mask=None):
        residual = tgt
        tgt, memory = self.input_layernorm_q(tgt), self.input_layernorm_kv(memory)
        x, attn_weights = self.attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            return_weights=True,
        )
        residual = x + residual

        x = self.post_attention_layernorm(residual)
        x = self.mlp(x)
        x = x + residual

        return x, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=None, nhead=16, dropout=0.1):
        super().__init__()

        self.attn = MultiheadAttention(hidden_size, nhead, dropout=dropout)
        self.mlp = GluMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

    def forward(self, x, attn_bias=None, key_padding_mask=None, tgt_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x, _ = self.attn(
            x,
            x,
            x,
            attn_bias=attn_bias,
            key_padding_mask=key_padding_mask,
            attn_mask=tgt_mask,
            return_weights=False,
        )
        residual = x + residual

        x = self.post_attention_layernorm(residual)
        x = self.mlp(x)
        x = x + residual

        return x


class FishSpeechTransformer(nn.Module):
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
    ):
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size * num_codebooks)


if __name__ == "__main__":
    mha = MultiheadAttention(512, 8, dropout=0)
    mha.eval()
    mha.cuda()

    q, k, v = torch.randn(3, 10, 16, 512)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    alibi = AlibiPostionEmbedding(8, 1024)

    mha.bfloat16()
    q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    bias = alibi(q).bfloat16()

    # Causual mask
    attn_mask = torch.triu(torch.ones(16, 16), diagonal=1).bool().cuda()
    o, w = mha(q, k, v, return_weights=True, attn_bias=bias, attn_mask=attn_mask)

    print(o.size())
    print(w.size())

    o1, w = mha(q, k, v, return_weights=False, attn_bias=bias, attn_mask=attn_mask)
    print(o1.size())

    print(o[0], o1.float()[0])

    assert torch.allclose(o.float(), o1.float(), atol=1e-2, rtol=1e-2)
    print("ok")

    cross = CrossAttentionLayer(512, 1024, dropout=0)
    cross.eval()
    cross.cuda()

    tgt = torch.randn(3, 10, 512).cuda()
    memory = torch.randn(3, 20, 512).cuda()
    o, w = cross(tgt, memory)

    print(o.size())
    print(w.size())

    ten = TransformerEncoderLayer(512, 1024, 8, dropout=0)
    ten.eval()
    ten.cuda()

    tgt = torch.randn(3, 10, 512).cuda()
    o = ten(tgt)
    print(o.size())
