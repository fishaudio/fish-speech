from dataclasses import dataclass

import loralib as lora


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0


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
    # Replace the embedding layer with a LoRA layer, preserving pretrained weights
    model.embeddings = _replace_embedding(model.embeddings, lora_config)
    model.codebook_embeddings = _replace_embedding(
        model.codebook_embeddings, lora_config
    )

    # Replace output layer with a LoRA layer (only exists when tie_word_embeddings=False)
    linears = []
    if hasattr(model, "output"):
        linears.append((model, "output"))

    # Replace all linear layers with LoRA layers
    for layer in model.layers:
        linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
        linears.extend(
            [
                (layer.feed_forward, "w1"),
                (layer.feed_forward, "w2"),
                (layer.feed_forward, "w3"),
            ]
        )

    if hasattr(model, "fast_layers"):
        model.fast_embeddings = _replace_embedding(model.fast_embeddings, lora_config)

        # Dual-AR model
        linears.append((model, "fast_output"))

        for layer in model.fast_layers:
            linears.extend([(layer.attention, "wqkv"), (layer.attention, "wo")])
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
