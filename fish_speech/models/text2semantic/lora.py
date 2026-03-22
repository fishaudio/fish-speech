from dataclasses import dataclass, field

import loralib as lora


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0
    # Valid values: "attention", "mlp", "embeddings", "output",
    #               "fast_attention", "fast_mlp", "fast_embeddings", "fast_output"
    # Unprefixed names target the slow transformer (and fast too for backwards compat).
    # "fast_*" names target only the fast transformer.
    target_modules: list = field(
        default_factory=lambda: ["attention", "mlp", "embeddings", "output"]
    )


def _replace_embedding(old_embed, lora_config):
    new_embed = lora.Embedding(
        num_embeddings=old_embed.num_embeddings,
        embedding_dim=old_embed.embedding_dim,
        padding_idx=old_embed.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )
    new_embed.weight.data.copy_(old_embed.weight.data)
    return new_embed


def setup_lora(model, lora_config):
    targets = set(lora_config.target_modules)
    linears = []

    # Slow transformer: targeted by unprefixed names (e.g. "attention")
    slow_attention = "attention" in targets
    slow_mlp = "mlp" in targets
    slow_embeddings = "embeddings" in targets
    slow_output = "output" in targets

    # Fast transformer: targeted by unprefixed names (backwards compat) OR "fast_*"
    fast_attention = slow_attention or "fast_attention" in targets
    fast_mlp = slow_mlp or "fast_mlp" in targets
    fast_embeddings = slow_embeddings or "fast_embeddings" in targets
    fast_output = slow_output or "fast_output" in targets

    if slow_embeddings:
        model.embeddings = _replace_embedding(model.embeddings, lora_config)
        model.codebook_embeddings = _replace_embedding(
            model.codebook_embeddings, lora_config
        )

    if slow_output and hasattr(model, "output"):
        linears.append((model, "output"))

    for layer in model.layers:
        if slow_attention:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        if slow_mlp:
            linears.extend(
                [
                    (layer.feed_forward, "w1"),
                    (layer.feed_forward, "w2"),
                    (layer.feed_forward, "w3"),
                ]
            )

    if hasattr(model, "fast_layers"):
        if fast_embeddings:
            model.fast_embeddings = _replace_embedding(
                model.fast_embeddings, lora_config
            )
        if fast_output:
            linears.append((model, "fast_output"))

        for layer in model.fast_layers:
            if fast_attention:
                linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
            if fast_mlp:
                linears.extend(
                    [
                        (layer.feed_forward, "w1"),
                        (layer.feed_forward, "w2"),
                        (layer.feed_forward, "w3"),
                    ]
                )

    for module, layer_name in linears:
        old_linear = getattr(module, layer_name)
        updated_linear = lora.Linear(
            in_features=old_linear.in_features,
            out_features=old_linear.out_features,
            bias=old_linear.bias is not None,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
        )
        updated_linear.weight.data.copy_(old_linear.weight.data)
        if old_linear.bias is not None:
            updated_linear.bias.data.copy_(old_linear.bias.data)
        setattr(module, layer_name, updated_linear)

    # Mark only the LoRA layers as trainable
    lora.mark_only_lora_as_trainable(model, bias="none")


def get_merged_state_dict(model):
    # This line will merge the state dict of the model and the LoRA parameters
    model.eval()

    # Then we need to remove the LoRA parameters from the state dict
    state_dict = model.state_dict()
    for name in list(state_dict.keys()):
        if "lora" in name:
            state_dict.pop(name)

    return state_dict
