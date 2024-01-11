import platform
from dataclasses import dataclass
from typing import Any, Dict, Optional

import lightning as L
import loralib as lora
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils
from fish_speech.models.text2semantic.llama import Transformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0


class TextToSemantic(L.LightningModule):
    def __init__(
        self,
        model: Transformer,
        optimizer: Any,
        lr_scheduler: Any,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler
        self.lora_config = lora_config

        if self.lora_config is not None:
            self.setup_lora()

    def setup_lora(self):
        # Replace the embedding layer with a LoRA layer
        self.model.embeddings = lora.Embedding(
            num_embeddings=self.model.embeddings.num_embeddings,
            embedding_dim=self.model.embeddings.embedding_dim,
            padding_idx=self.model.embeddings.padding_idx,
            r=self.lora_config.r,
            lora_alpha=self.lora_config.lora_alpha,
        )

        # Replace output layer with a LoRA layer
        linears = [(self.model, "output")]

        # Replace all linear layers with LoRA layers
        for layer in self.model.layers:
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
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
            )
            setattr(module, layer, updated_linear)

        # Mark only the LoRA layers as trainable
        lora.mark_only_lora_as_trainable(self.model, bias="lora_only")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_builder(self.parameters())
        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def _step(self, batch, batch_idx, stage: str):
        outputs = self.model(
            x=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
        )

        # Generate labels
        labels = batch["labels"]
        loss = F.cross_entropy(
            outputs.token_logits.reshape(-1, outputs.token_logits.size(-1)),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )

        # If we have a codebook, add the loss
        if self.model.config.num_codebooks != 0:
            codebook_labels = labels[:, 1:].mT
            semantic_loss = F.cross_entropy(
                outputs.codebook_logits.reshape(-1, outputs.codebook_logits.size(-1)),
                codebook_labels.reshape(-1),
                ignore_index=-100,
            )

            loss = loss + semantic_loss

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Top-5 accuracy
        if self.model.config.num_codebooks == 0:
            _, indices = outputs.token_logits.topk(5, dim=-1)
            correct = indices.eq(labels[:, 0].unsqueeze(-1))
            correct[labels[:, 0] == -100] = 0
            correct = correct.sum()
            accuracy = correct / (labels[:, 0] != -100).sum()
        else:
            _, indices = outputs.codebook_logits.topk(5, dim=-1)
            correct = indices.eq(codebook_labels.unsqueeze(-1))
            correct[codebook_labels == -100] = 0
            correct = correct.sum()
            accuracy = correct / (codebook_labels != -100).sum()

        self.log(
            f"{stage}/top_5_accuracy",
            accuracy,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")
