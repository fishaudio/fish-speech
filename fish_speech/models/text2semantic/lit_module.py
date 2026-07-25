from typing import Any, Optional

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

import fish_speech.utils as utils

CODEBOOK_PAD_TOKEN_ID = 0
from fish_speech.models.text2semantic.llama import BaseTransformer, NaiveTransformer

log = utils.RankedLogger(__name__, rank_zero_only=True)


class TextToSemantic(L.LightningModule):

    def __init__(
        self,
        model: BaseTransformer,
        optimizer: Any,
        lr_scheduler: Any,
        base_weight: float = 1.0,
        base_vq_weight: float = 0.5,
        decode_semantic_token_weight: float = 0.5,
        semantic_weights: Optional[list[float]] = None,
    ):
        super().__init__()

        self.model = model
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        self.base_weight = base_weight
        self.base_vq_weight = base_vq_weight
        self.decode_semantic_token_weight = decode_semantic_token_weight
        self.semantic_weights = semantic_weights

    def forward(self, x):
        return self.model(x)

    def on_save_checkpoint(self, checkpoint):
        # Save only LoRA parameters
        state_dict = checkpoint["state_dict"]
        use_lora = any("lora" in name for name in state_dict.keys())
        if not use_lora:
            return

        for name in list(state_dict.keys()):
            if "lora" not in name:
                state_dict.pop(name)
        checkpoint.pop("optimizer_states", None)
        checkpoint.pop("lr_schedulers", None)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Get weight decay parameters
        weight_decay_parameters, other_parameters = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if ".bias" in name or "norm.weight" in name or ".embeddings." in name:
                other_parameters.append(param)
            else:
                weight_decay_parameters.append(param)

        optimizer = self.optimizer_builder(
            [
                {"params": weight_decay_parameters},
                {"params": other_parameters, "weight_decay": 0.0},
            ]
        )

        # Print the parameters and their weight decay
        for i, group in enumerate(optimizer.param_groups):
            log.info(
                f"Set weight_decay={group.get('weight_decay', 0.0)} for "
                f"{len(group['params'])} parameters (group {i})"
            )

        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    # Copied from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py#L90
    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:

        assert logits.shape[:-1] == labels.shape

        labels = labels.clone()
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _step(self, batch, batch_idx, stage: str):
        is_train = stage == "train"

        if is_train:
            # Key part to make lora work
            # Otherwise the parameters are merged, which lead to incorrect gradients
            self.model.train()

        labels = batch["labels"]  # (B, C+1, T)
        outputs = self.model(
            inp=batch["inputs"],
            key_padding_mask=batch["attention_masks"],
            labels=batch["labels"],
        )
        token_logits = outputs.token_logits  # (B, T, vocab_size)
        codebook_logits = outputs.codebook_logits  # (num_semantic, num_codebooks, codebook_size) for Dual-AR

        token_labels = labels[:, 0]  # (B, T)

        semantic_begin = self.model.config.semantic_begin_id
        semantic_end = self.model.config.semantic_end_id

        valid = token_labels != -100
        is_semantic = (token_labels >= semantic_begin) & (token_labels <= semantic_end)
        text_mask = valid & ~is_semantic
        vq_mask = valid & is_semantic

        base_loss_text = _masked_cross_entropy(token_logits, token_labels, text_mask)
        base_loss_vq = _masked_cross_entropy(token_logits, token_labels, vq_mask)
        base_loss = base_loss_text + base_loss_vq * self.base_vq_weight

        all_codebook_labels = labels[:, 1 : 1 + self.model.config.num_codebooks]

        all_codebook_labels = all_codebook_labels.permute(0, 2, 1)
        filtered_codebook_labels = all_codebook_labels[is_semantic]

        semantic_loss = self._codebook_loss(
            codebook_logits, filtered_codebook_labels
        )

        loss = (
            base_loss * self.base_weight
            + semantic_loss * self.decode_semantic_token_weight
        )

        self.log(
            f"{stage}/loss",
            loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )
        self.log(
            f"{stage}/base_loss",
            base_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )
        self.log(
            f"{stage}/base_loss_text",
            base_loss_text,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )
        self.log(
            f"{stage}/base_loss_vq",
            base_loss_vq,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )
        self.log(
            f"{stage}/semantic_loss",
            semantic_loss,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=False,
            logger=True,
            sync_dist=not is_train,
        )

        # Top-5 accuracy
        accuracy = self.get_accuracy(codebook_logits, filtered_codebook_labels)
        self.log(
            f"{stage}/top_5_accuracy",
            accuracy,
            on_step=is_train,
            on_epoch=not is_train,
            prog_bar=True,
            logger=True,
            sync_dist=not is_train,
        )

        return loss

    def _codebook_loss(
        self,
        codebook_logits: torch.Tensor,
        codebook_labels: torch.Tensor,
    ) -> torch.Tensor:

        if codebook_logits.numel() == 0 or codebook_labels.shape[0] == 0:
            return torch.tensor(
                0.0,
                device=codebook_logits.device,
                dtype=codebook_logits.dtype,
            )

        V = codebook_logits.size(-1)

        if self.semantic_weights is not None:
            assert len(self.semantic_weights) == codebook_logits.size(1), (
                f"semantic_weights length {len(self.semantic_weights)} must "
                f"match num_codebooks {codebook_logits.size(1)}"
            )
            total_loss = torch.tensor(
                0.0, device=codebook_logits.device, dtype=torch.float32
            )
            total_weight = 0.0
            for cb_idx, w in enumerate(self.semantic_weights):
                if w == 0.0:
                    continue
                loss_i = F.cross_entropy(
                    codebook_logits[:, cb_idx, :].reshape(-1, V),
                    codebook_labels[:, cb_idx].reshape(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + loss_i * w
                total_weight += w
            if total_weight == 0.0:
                return torch.tensor(
                    0.0,
                    device=codebook_logits.device,
                    dtype=codebook_logits.dtype,
                )
            return total_loss / total_weight

        return F.cross_entropy(
            codebook_logits.reshape(-1, V),
            codebook_labels.reshape(-1),
            ignore_index=-100,
        )

    def get_accuracy(self, logits, labels, topk: int = 5):
        mask = (labels != -100) & (labels != CODEBOOK_PAD_TOKEN_ID)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        _, indices = logits.topk(topk, dim=-1)
        correct = indices.eq(labels.unsqueeze(-1))
        correct[~mask] = 0
        correct = correct.sum()
        accuracy = correct / mask.sum()

        return accuracy

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")


def _masked_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:

    if not mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    sel_logits = logits[mask]  # (N, V)
    sel_labels = labels[mask]  # (N,)
    return F.cross_entropy(sel_logits, sel_labels, ignore_index=-100)
