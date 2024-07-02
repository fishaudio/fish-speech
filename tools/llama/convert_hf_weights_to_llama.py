import torch
from transformers import LlamaForCausalLM

from fish_speech.models.text2semantic.llama import BaseModelArgs, BaseTransformer

# Load the HF model
hf_model = LlamaForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

model = BaseTransformer(
    BaseModelArgs(
        vocab_size=hf_model.config.vocab_size + 8,
        n_layer=hf_model.config.num_hidden_layers,
        n_head=hf_model.config.num_attention_heads,
        n_local_heads=hf_model.config.num_key_value_heads,
        dim=hf_model.config.hidden_size,
        head_dim=hf_model.config.hidden_size // hf_model.config.num_attention_heads,
        num_codebooks=2,
        codebook_size=1032,
    )
)
print(model.config)

hf_state_dict = hf_model.state_dict()
model_state_dict = model.state_dict()

# print(hf_state_dict.keys())
# print(model_state_dict.keys())

new_state_dict = {}

# Handle embeddings
new_state_dict["embeddings.weight"] = model_state_dict.pop("embeddings.weight")
hf_embed_tokens = hf_state_dict.pop("model.embed_tokens.weight")
new_state_dict["embeddings.weight"][: hf_embed_tokens.shape[0]] = hf_embed_tokens

# Restore layers
for layer_idx in range(hf_model.config.num_hidden_layers):
    # Handle attention
    q_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.self_attn.q_proj.weight")
    k_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.self_attn.k_proj.weight")
    v_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.self_attn.v_proj.weight")
    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
    new_state_dict[f"layers.{layer_idx}.attention.wqkv.weight"] = qkv_weight
    model_state_dict.pop(f"layers.{layer_idx}.attention.wqkv.weight")

    o_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
    new_state_dict[f"layers.{layer_idx}.attention.wo.weight"] = o_weight
    model_state_dict.pop(f"layers.{layer_idx}.attention.wo.weight")

    # Handle feed forward
    up_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.mlp.up_proj.weight")
    down_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.mlp.down_proj.weight")
    gate_weight = hf_state_dict.pop(f"model.layers.{layer_idx}.mlp.gate_proj.weight")

    new_state_dict[f"layers.{layer_idx}.feed_forward.w1.weight"] = gate_weight
    new_state_dict[f"layers.{layer_idx}.feed_forward.w2.weight"] = down_weight
    new_state_dict[f"layers.{layer_idx}.feed_forward.w3.weight"] = up_weight

    model_state_dict.pop(f"layers.{layer_idx}.feed_forward.w1.weight")
    model_state_dict.pop(f"layers.{layer_idx}.feed_forward.w2.weight")
    model_state_dict.pop(f"layers.{layer_idx}.feed_forward.w3.weight")

    # Handle layer norms
    input_layernorm_weight = hf_state_dict.pop(
        f"model.layers.{layer_idx}.input_layernorm.weight"
    )
    post_attention_layernorm_weight = hf_state_dict.pop(
        f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    )

    new_state_dict[f"layers.{layer_idx}.ffn_norm.weight"] = (
        post_attention_layernorm_weight
    )
    new_state_dict[f"layers.{layer_idx}.attention_norm.weight"] = input_layernorm_weight

    model_state_dict.pop(f"layers.{layer_idx}.ffn_norm.weight")
    model_state_dict.pop(f"layers.{layer_idx}.attention_norm.weight")

# Handle final layer norm
new_state_dict["norm.weight"] = hf_state_dict.pop("model.norm.weight")
model_state_dict.pop("norm.weight")

# Handle output layer
w = hf_state_dict.pop("lm_head.weight")
new_state_dict["output.weight"] = model_state_dict.pop("output.weight")
new_state_dict["output.weight"][: w.shape[0]] = w

print(hf_state_dict.keys(), len(hf_state_dict))
print(model_state_dict.keys(), len(model_state_dict))

print(model.load_state_dict(new_state_dict, strict=True))

model = model.bfloat16()

new_state_dict = {f"model.{k}": v for k, v in model.state_dict().items()}
torch.save(
    new_state_dict,
    "checkpoints/fish-speech-agent-1/TinyLlama-1.1B-intermediate-step-1431k-3T.pth",
)
