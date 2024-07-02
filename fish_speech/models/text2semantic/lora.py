from dataclasses import dataclass

import loralib as lora


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0


def setup_lora(model, lora_config):
    # Replace the embedding layer with a LoRA layer
    model.embeddings = lora.Embedding(
        num_embeddings=model.embeddings.num_embeddings,
        embedding_dim=model.embeddings.embedding_dim,
        padding_idx=model.embeddings.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )

    model.codebook_embeddings = lora.Embedding(
        num_embeddings=model.codebook_embeddings.num_embeddings,
        embedding_dim=model.codebook_embeddings.embedding_dim,
        padding_idx=model.codebook_embeddings.padding_idx,
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
    )

    # Replace output layer with a LoRA layer
    linears = [(model, "output")]

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
        model.fast_embeddings = lora.Embedding(
            num_embeddings=model.fast_embeddings.num_embeddings,
            embedding_dim=model.fast_embeddings.embedding_dim,
            padding_idx=model.fast_embeddings.padding_idx,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
        )

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

    for module, layer in linears:
        updated_linear = lora.Linear(
            in_features=getattr(module, layer).in_features,
            out_features=getattr(module, layer).out_features,
            bias=getattr(module, layer).bias,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
        )
        setattr(module, layer, updated_linear)

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
