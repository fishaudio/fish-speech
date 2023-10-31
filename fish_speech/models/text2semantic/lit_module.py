from typing import Any

import lightning as L
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
        result = self.model(**batch)
        loss = result.loss
        logits = result.logits

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
        correct = indices.eq(batch["labels"].unsqueeze(-1)).sum()
        accuracy = correct / batch["labels"].numel()
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
