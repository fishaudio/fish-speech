from typing import Optional
from transformers import HubertModel
from torch import nn
import torch
from encodec.quantization.core_vq import VectorQuantization


class HubertVQ(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "facebook/hubert-large-ls960-ft",
        vq_layer: int = -4,  # the layer to extract the quantized features
        codebook_size: int = 1024,
        trainable_layers_before_vq: int = 2,
        trainable_layers_after_vq: int = 2,
    ):
        super().__init__()

        self.hubert = HubertModel.from_pretrained(model_name_or_path)
        self.vq_layer = (
            (self.hubert.config.num_hidden_layers + vq_layer)
            if vq_layer < 0
            else vq_layer
        )
        self.trainable_layers_before_vq = trainable_layers_before_vq
        self.trainable_layers_after_vq = trainable_layers_after_vq

        assert (
            self.vq_layer >= trainable_layers_before_vq
            and self.vq_layer
            < self.hubert.config.num_hidden_layers - trainable_layers_after_vq
        ), "vq_layer must be between trainable_layers_before_vq and num_hidden_layers - trainable_layers_after_vq"

        # Freeze both feature extractor & lm head
        for param in self.hubert.parameters():
            param.requires_grad = False

        # Unfreeze layers between vq_layer - trainable_layers_before_vq and vq_layer + trainable_layers_after_vq
        for param in self.hubert.encoder.layers[
            self.vq_layer
            - trainable_layers_before_vq : self.vq_layer
            + trainable_layers_after_vq
        ].parameters():
            param.requires_grad = True

        # Quantization
        self.quantizer = VectorQuantization(
            codebook_size=codebook_size,
            dim=self.hubert.config.hidden_size,
            kmeans_init=False,
        )

    @torch.no_grad()
    def _get_attention_mask(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # compute reduced attention_mask corresponding to feature vectors
        attention_mask = self.hubert._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask
        )

        # make sure padded tokens are not attended to
        expand_attention_mask = attention_mask.unsqueeze(-1).repeat(
            1, 1, hidden_states.shape[2]
        )
        hidden_states[~expand_attention_mask] = 0

        # extend attention_mask
        attention_mask = 1.0 - attention_mask[:, None, None, :].to(
            dtype=hidden_states.dtype
        )
        attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        attention_mask = attention_mask.expand(
            attention_mask.shape[0],
            1,
            attention_mask.shape[-1],
            attention_mask.shape[-1],
        )

        return hidden_states, attention_mask

    def encode(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            # Extract features
            extract_features = self.hubert.feature_extractor(input_values)
            extract_features = extract_features.transpose(1, 2)

            hidden_states = self.hubert.feature_projection(extract_features)
            hidden_states = self.hubert._mask_hidden_states(
                hidden_states, mask_time_indices=mask_time_indices
            )

            position_embeddings = self.hubert.encoder.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings

            if attention_mask is not None:
                # compute reduced attention_mask corresponding to feature vectors
                hidden_states, attention_mask = self._get_attention_mask(
                    hidden_states, attention_mask
                )

            # Only do layer norm if do_stable_layer_norm is False
            if self.hubert.config.do_stable_layer_norm is False:
                hidden_states = self.hubert.encoder.layer_norm(hidden_states)

            hidden_states = self.hubert.encoder.dropout(hidden_states)

        # Execute transformer
        for idx, layer_module in enumerate(self.hubert.encoder.layers[: self.vq_layer]):
            if idx < self.vq_layer - self.trainable_layers_before_vq:
                with torch.no_grad():
                    hidden_states = layer_module(hidden_states, attention_mask)[0]
            else:
                hidden_states = layer_module(hidden_states, attention_mask)[0]

        return hidden_states

    @torch.no_grad()
    def decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            _, attention_mask = self._get_attention_mask(
                hidden_states.clone(), attention_mask
            )

        # Execute transformer
        for idx, layer_module in enumerate(self.hubert.encoder.layers[self.vq_layer :]):
            if idx >= self.trainable_layers_after_vq:
                with torch.no_grad():
                    hidden_states = layer_module(hidden_states, attention_mask)[0]
            else:
                hidden_states = layer_module(hidden_states, attention_mask)[0]

        with torch.no_grad():
            # Only do layer norm if do_stable_layer_norm is False
            if self.hubert.config.do_stable_layer_norm is False:
                hidden_states = self.hubert.encoder.last_layer_norm(hidden_states)
            else:
                hidden_states = self.hubert.encoder.layer_norm(hidden_states)

        return hidden_states

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
    ):
        hidden_states = self.encode(
            input_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
        )

        # Quantize
        quantize, _, vq_loss = self.quantizer(hidden_states.transpose(1, 2))
        quantize = quantize.transpose(1, 2)

        # Inject position embeddings
        with torch.no_grad():
            position_embeddings = self.hubert.encoder.pos_conv_embed(hidden_states)

        quantize = quantize + position_embeddings

        # Decode
        hidden_states = self.decode(quantize, attention_mask=attention_mask)

        return hidden_states, vq_loss


# class HubertVQ
if __name__ == "__main__":
    from transformers import Wav2Vec2Tokenizer
    from datasets import load_dataset

    processor = Wav2Vec2Tokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertVQ()
    model.train()
    print("Loaded model")

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    gt_hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    gt_hubert.train()
    print("Loaded ground truth model")

    ds = load_dataset(
        "patrickvonplaten/librispeech_asr_dummy", "clean", split="validation"
    )
    print("Loaded dataset")

    input_values = processor(
        ds[0]["audio"]["array"], return_tensors="pt"
    )  # Batch size 1

    optim.zero_grad()
    # hidden_states = model.decode(model.encode(**input_values))
    hidden_states, vq_loss = model(**input_values)
    print(hidden_states, vq_loss)

    gt = gt_hubert(**input_values).last_hidden_state

    loss = torch.nn.functional.mse_loss(hidden_states, gt)
    print(loss)

    total_loss = loss + vq_loss
    total_loss.backward()
    optim.step()

    print("Backward pass done")
