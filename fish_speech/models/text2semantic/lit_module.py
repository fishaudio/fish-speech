import platform
from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils

log = utils.RankedLogger(__name__, rank_zero_only=True)


class TextToSemantic(L.LightningModule):
    def __init__(self, model, optimizer: Any, lr_scheduler: Any):
        super().__init__()

        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

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
