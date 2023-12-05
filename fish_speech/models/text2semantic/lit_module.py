from typing import Any

import lightning as L
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from transformers import LlamaForCausalLM


class TextToSemantic(L.LightningModule):
    def __init__(self, model: LlamaForCausalLM, optimizer: Any, lr_scheduler: Any):
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
        logits = self.model(
            inputs=batch["inputs"],
            input_mask=batch["input_mask"],
            codes=batch["codes"][..., :-1],
            codes_mask=batch["codes_mask"][..., :-1],
        )

        # Generate labels
        labels = batch["codes"][..., 1:].contiguous()
        label_mask = batch["codes_mask"][..., 1:]
        label_mask = label_mask[:, None, :]
        label_mask = label_mask.expand(-1, labels.size(1), -1)
        labels = labels.masked_fill(label_mask, -100)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )

        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Top-5 accuracy
        _, indices = logits.topk(5, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1)).sum()
        accuracy = correct / labels.numel()
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
