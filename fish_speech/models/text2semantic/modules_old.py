import math
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

try:
    from xformers.ops import memory_efficient_attention
except ImportError as e:
    memory_efficient_attention = None


class AlibiPostionEmbedding(nn.Module):
    def __init__(self, nheads, maxpos):
        super().__init__()

        context_position = torch.arange(maxpos)[:, None]
        memory_position = torch.arange(maxpos)[None, :]
        relative_position = memory_position - context_position
        relative_position = (
            torch.abs(relative_position).unsqueeze(0).expand(nheads, -1, -1)
        )
        self.slopes = torch.Tensor(self.get_slopes(nheads)) * -1
        alibi = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.view(nheads, maxpos, maxpos)

        self.register_buffer("alibi", alibi)

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


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, max_seq_length, n_heads * head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos is not None, "input_pos should not be None"

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, input_pos] = k_val
        v_out[:, input_pos] = v_val

        return k_out, v_out


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
        self.kv_cache = None

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        return_weights=False,
        input_pos=None,
    ):
        # (B, T, C)
        batch_size = q.size(0)
        q_length = q.size(1)

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k_length = k.size(1)

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

                if attn_bias is None:
                    attn_bias = torch.zeros_like(
                        attn_mask, dtype=q.dtype, device=q.device
                    )
                attn_bias = attn_bias.masked_fill(attn_mask, float("-inf"))

            if attn_bias is not None:
                attn_bias = attn_bias.to(q.dtype)

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

    def forward(
        self,
        tgt,
        memory,
        memory_key_padding_mask=None,
        input_pos=None,
    ):
        residual = tgt
        tgt, memory = self.input_layernorm_q(tgt), self.input_layernorm_kv(memory)
        x, attn_weights = self.attn(
            tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            return_weights=True,
            input_pos=input_pos,
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

    def forward(
        self, x, attn_bias=None, key_padding_mask=None, tgt_mask=None, input_pos=None
    ):
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
            input_pos=input_pos,
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
        alignment_position=-2,
        max_position=8192,
    ):
        super().__init__()

        self.encoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, hidden_size) for _ in range(num_codebooks)]
        )
        self.decoder_head = nn.Linear(hidden_size, codebook_size * num_codebooks)
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

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

        self.alignment = CrossAttentionLayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        if alignment_position < 0:
            alignment_position = num_decoder_layers + alignment_position

        self.alignment_position = alignment_position
        assert 0 <= alignment_position < num_decoder_layers

        self.decoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    nhead=nhead,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.alibi = AlibiPostionEmbedding(nhead, max_position)
        self.register_buffer(
            "causual_mask",
            torch.triu(torch.ones(max_position, max_position), diagonal=1).bool(),
        )

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
            b.attn.kv_cache = KVCache(
                max_batch_size, max_seq_length, b.attn.nhead, b.attn.head_dim
            )

    def forward(self, inputs, codes, input_mask=None, codes_mask=None):
        # x: (B, T)
        # y: (B, C, T)
        inputs = self.encoder_embedding(inputs)
        codes = rearrange(codes, "b c t -> c b t")
        codes = torch.stack(
            [emb(code) for emb, code in zip(self.decoder_embeddings, codes)], dim=0
        )
        codes = torch.mean(codes, dim=0)  # (B, T)

        attn_bias = self.alibi(inputs)
        for layer in self.encoder:
            inputs = layer(inputs, attn_bias=attn_bias, key_padding_mask=input_mask)

        attn_bias = self.alibi(codes)
        causual_mask = self.causual_mask[: codes.shape[1], : codes.shape[1]]

        for idx, layer in enumerate(self.decoder):
            if idx == self.alignment_position:
                codes, _ = self.alignment(
                    codes, inputs, memory_key_padding_mask=input_mask
                )

            codes = layer(
                codes,
                attn_bias=attn_bias,
                key_padding_mask=codes_mask,
                tgt_mask=causual_mask,
            )

        codes = self.decoder_head(codes)
        codes = rearrange(
            codes, "b t (c d) -> b c t d", c=self.num_codebooks, d=self.codebook_size
        )

        return codes

    def sample_decoder(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        input_pos: torch.Tensor,
        **sampling_kwargs,
    ):
        attn_bias = self.alibi.alibi[:, input_pos, : self.max_seq_length]
        causual_mask = self.causual_mask[input_pos, : self.max_seq_length]

        x = rearrange(x, "b c t -> c b t")
        x = torch.stack(
            [emb(code) for emb, code in zip(self.decoder_embeddings, x)], dim=0
        )
        x = torch.mean(x, dim=0)  # (B, T)

        for idx, layer in enumerate(self.decoder):
            if idx == self.alignment_position:
                x, _ = self.alignment(x, context)

            x = layer(
                x, attn_bias=attn_bias, input_pos=input_pos, tgt_mask=causual_mask
            )

        x = self.decoder_head(x)
        x = rearrange(
            x, "b t (c d) -> b c t d", c=self.num_codebooks, d=self.codebook_size
        )

        # Never predict EOS or BOS for sub-codebooks
        x[:, 1:, :2] = -float("Inf")

        next_token, probs = [], []
        for i in range(self.num_codebooks):
            next_token_i, probs_i = self.sample(x[:, i], **sampling_kwargs)
            next_token.append(next_token_i)
            probs.append(probs_i)

        return torch.stack(next_token, dim=0), torch.stack(probs, dim=0)

    @staticmethod
    def multinomial_sample_one_no_sync(
        probs_sort,
    ):  # Does multinomial sampling without a cuda synchronization
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    @staticmethod
    def logits_to_probs(
        logits,
        temperature: float = 1.0,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[0] = False  # keep at least one option
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=0, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, -float("Inf"))

        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def sample(
        self,
        logits,
        temperature: float = 1.0,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        probs = self.logits_to_probs(logits[0, -1], temperature, top_p, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def decode_n_tokens(
        self,
        cur_token: torch.Tensor,
        context: torch.Tensor,
        input_pos: torch.Tensor,
        num_new_tokens: int,
        callback=lambda _: _,
        **sampling_kwargs,
    ):
        new_tokens, new_probs = [], []
        # Sliding context window
        batch_size = 1
        back_map = torch.zeros(
            [batch_size, 1], device=cur_token.device, dtype=torch.long
        )

        for i in range(num_new_tokens):
            next_token, next_prob = self.sample_decoder(
                cur_token, context, input_pos, **sampling_kwargs
            )

            # index_map = torch.arange(6, device=cur_token.device)
            # index_map = back_map[:, -1:] + index_map.repeat(batch_size, 1)
            # add = torch.arange(batch_size, device=index_map.device).unsqueeze(1) #N, 1
            # index_map = index_map + add * t_length

            input_pos += 1
            new_tokens.append(next_token.clone())
            callback(new_tokens[-1])
            new_probs.append(next_prob.clone())

            if next_token[0, 0] == 1:
                break

            cur_token = next_token.view(1, self.num_codebooks, -1)

        return new_tokens, new_probs

    def compile(self):
        self.sampler_decoder = torch.compile(
            self.sample_decoder, mode="reduce-overhead", fullgraph=True
        )

    @torch.no_grad()
    def inference(self, inputs, prompt=None, max_new_tokens=1024, **sampling_kwargs):
        # inputs: (B, T)
        # prompt: (B, C, T)

        assert inputs.size(0) == 1, "Only support batch size 1 for now"

        if prompt is None:
            prompt = torch.tensor(
                [[[0]] * self.num_codebooks], device=inputs.device, dtype=torch.long
            )

        T = prompt.size(2)
        T_new = T + max_new_tokens

        # Encode Features
        inputs = self.encoder_embedding(inputs)
        attn_bias = self.alibi(inputs)
        for layer in self.encoder:
            inputs = layer(inputs, attn_bias=attn_bias)

        device, dtype = inputs.device, inputs.dtype

        # Decode
        with torch.device(inputs.device):
            self.setup_kv_caches(max_batch_size=1, max_seq_length=T_new)

        # create an empty tensor of the expected final shape and fill in the current tokens
        empty = torch.empty(
            (1, self.num_codebooks, T_new), dtype=torch.long, device=device
        )
        empty[:, :, :T] = prompt
        seq = empty
        input_pos = torch.arange(0, T, device=device)

        # prefill
        next_token, _ = self.sample_decoder(
            prompt.view(1, self.num_codebooks, -1), inputs, input_pos, **sampling_kwargs
        )
        seq[:, :, T] = next_token

        # create an empty tensor of the expected final shape and fill in the current tokens
        input_pos = torch.tensor([T], device=device, dtype=torch.long)
        generated_tokens, _ = self.decode_n_tokens(
            next_token.view(1, self.num_codebooks, -1),
            context=inputs,
            input_pos=input_pos,
            num_new_tokens=max_new_tokens - 1,
            **sampling_kwargs,
        )

        generated_tokens = torch.stack(generated_tokens, dim=-1)
        seq = seq[:, :, : T + 1 + generated_tokens.size(-1)]
        seq[:, :, T + 1 :] = generated_tokens

        return seq


if __name__ == "__main__":
    # mha = MultiheadAttention(512, 8, dropout=0)
    # mha.eval()
    # mha.cuda()

    # q, k, v = torch.randn(3, 10, 16, 512)
    # q, k, v = q.cuda(), k.cuda(), v.cuda()
    # alibi = AlibiPostionEmbedding(8, 1024)

    # mha.bfloat16()
    # q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
    # bias = alibi(q).bfloat16()

    # # Causual mask
    # attn_mask = torch.triu(torch.ones(16, 16), diagonal=1).bool().cuda()
    # o, w = mha(q, k, v, return_weights=True, attn_bias=bias, attn_mask=attn_mask)

    # print(o.size())
    # print(w.size())

    # o1, w = mha(q, k, v, return_weights=False, attn_bias=bias, attn_mask=attn_mask)
    # print(o1.size())

    # print(o[0], o1.float()[0])

    # assert torch.allclose(o.float(), o1.float(), atol=1e-2, rtol=1e-2)
    # print("ok")

    # cross = CrossAttentionLayer(512, 1024, dropout=0)
    # cross.eval()
    # cross.cuda()

    # tgt = torch.randn(3, 10, 512).cuda()
    # memory = torch.randn(3, 20, 512).cuda()
    # o, w = cross(tgt, memory)

    # print(o.size())
    # print(w.size())

    # ten = TransformerEncoderLayer(512, 1024, 8, dropout=0)
    # ten.eval()
    # ten.cuda()

    # tgt = torch.randn(3, 10, 512).cuda()
    # o = ten(tgt)
    # print(o.size())

    trans = (
        FishSpeechTransformer(
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
    # Print n param
    print("Total params:", sum(i.numel() for i in trans.parameters()) / 1024 / 1024)
    inputs = torch.randint(0, 1000, (1, 16)).cuda()
    codes = torch.randint(0, 120, (1, 4, 128)).cuda()
    print(trans(inputs, codes).size())

    r = trans.inference(inputs, max_new_tokens=1024, top_k=5, temperature=0.3)
    print(r)
