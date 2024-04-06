import itertools
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

from fish_speech.models.vqgan.utils import plot_mel, sequence_mask, slice_segments


@dataclass
class VQEncodeResult:
    features: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    feature_lengths: torch.Tensor


@dataclass
class VQDecodeResult:
    mels: torch.Tensor
    audios: Optional[torch.Tensor] = None


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        encoder: nn.Module,
        quantizer: nn.Module,
        aux_decoder: nn.Module,
        reflow: nn.Module,
        vocoder: nn.Module,
        mel_transform: nn.Module,
        weight_reflow: float = 1.0,
        weight_vq: float = 1.0,
        weight_aux_mel: float = 1.0,
        sampling_rate: int = 44100,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Modules
        self.encoder = encoder
        self.quantizer = quantizer
        self.aux_decoder = aux_decoder
        self.reflow = reflow
        self.mel_transform = mel_transform
        self.vocoder = vocoder

        # Freeze vocoder
        for param in self.vocoder.parameters():
            param.requires_grad = False

        # Loss weights
        self.weight_reflow = weight_reflow
        self.weight_vq = weight_vq
        self.weight_aux_mel = weight_aux_mel

        self.spec_min = -12
        self.spec_max = 3
        self.sampling_rate = sampling_rate
        self.strict_loading = False

    def on_save_checkpoint(self, checkpoint):
        # Do not save vocoder
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "vocoder" in name:
                state_dict.pop(name)

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer = self.optimizer_builder(self.parameters())
        lr_scheduler = self.lr_scheduler_builder(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min

    # @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def training_step(self, batch, batch_idx):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios)

        mel_lengths = audio_lengths // self.mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()

        # Encode
        encoded_features = self.encoder(gt_mels) * mel_masks_float_conv

        # Quantize
        vq_result = self.quantizer(encoded_features)
        loss_vq = getattr("vq_result", "loss", 0.0)
        vq_recon_features = vq_result.z * mel_masks_float_conv

        # VQ Decode
        aux_mel = self.aux_decoder(vq_recon_features)
        loss_aux_mel = F.l1_loss(
            aux_mel * mel_masks_float_conv, gt_mels * mel_masks_float_conv
        )

        # Reflow
        x_1 = self.norm_spec(gt_mels)
        t = torch.rand(gt_mels.shape[0], device=gt_mels.device)
        x_0 = torch.randn_like(x_1)

        # X_t = t * X_1 + (1 - t) * X_0
        x_t = x_0 + t[:, None, None] * (x_1 - x_0)

        v_pred = self.reflow(
            x_t,
            1000 * t,
            vq_recon_features,  # .detach()
            x_masks=mel_masks_float_conv,
            cond_masks=mel_masks_float_conv,
        )

        # Log L2 loss with
        weights = 0.398942 / t / (1 - t) * torch.exp(-0.5 * torch.log(t / (1 - t)) ** 2)
        loss_reflow = weights[:, None, None] * F.mse_loss(
            x_1 - x_0, v_pred, reduction="none"
        )
        loss_reflow = (loss_reflow * mel_masks_float_conv).mean()

        # Total loss
        loss = (
            self.weight_vq * loss_vq
            + self.weight_aux_mel * loss_aux_mel
            + self.weight_reflow * loss_reflow
        )

        # Log losses
        self.log(
            "train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        self.log(
            "train/loss_vq",
            loss_vq,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/loss_aux_mel",
            loss_aux_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/loss_reflow",
            loss_reflow,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        gt_mels = self.mel_transform(audios)
        mel_lengths = audio_lengths // self.mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()

        # Encode
        encoded_features = self.encoder(gt_mels) * mel_masks_float_conv

        # Quantize
        vq_result = self.quantizer(encoded_features)

        # VQ Decode
        aux_mels = self.aux_decoder(vq_result.z)
        loss_aux_mel = F.l1_loss(
            aux_mels * mel_masks_float_conv, gt_mels * mel_masks_float_conv
        )

        self.log(
            "val/loss_aux_mel",
            loss_aux_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Reflow inference
        t_start = 0.0
        infer_step = 10

        x_1 = self.norm_spec(aux_mels)
        x_0 = torch.randn_like(x_1)
        gen_mels = (1 - t_start) * x_0 + t_start * x_1

        t = torch.zeros(gt_mels.shape[0], device=gt_mels.device)
        dt = (1.0 - t_start) / infer_step

        for _ in range(infer_step):
            gen_mels += (
                self.reflow(
                    gen_mels,
                    1000 * t,
                    vq_result.z,
                    x_masks=mel_masks_float_conv,
                    cond_masks=mel_masks_float_conv,
                )
                * dt
            )
            t += dt

        gen_mels = self.denorm_spec(gen_mels)
        loss_recon_reflow = F.l1_loss(
            gen_mels * mel_masks_float_conv, gt_mels * mel_masks_float_conv
        )

        self.log(
            "val/loss_recon_reflow",
            loss_recon_reflow,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        gen_audios = self.vocoder(gen_mels)
        recon_audios = self.vocoder(gt_mels)
        aux_audios = self.vocoder(aux_mels)

        # only log the first batch
        if batch_idx != 0:
            return

        for idx, (
            gt_mel,
            reflow_mel,
            aux_mel,
            audio,
            reflow_audio,
            aux_audio,
            recon_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                gen_mels,
                aux_mels,
                audios.float(),
                gen_audios.float(),
                aux_audios.float(),
                recon_audios.float(),
                audio_lengths,
            )
        ):
            mel_len = audio_len // self.mel_transform.hop_length

            image_mels = plot_mel(
                [
                    gt_mel[:, :mel_len],
                    reflow_mel[:, :mel_len],
                    aux_mel[:, :mel_len],
                ],
                [
                    "Ground-Truth",
                    "Reflow",
                    "Aux",
                ],
            )

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(
                    {
                        "reconstruction_mel": wandb.Image(image_mels, caption="mels"),
                        "wavs": [
                            wandb.Audio(
                                audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="gt",
                            ),
                            wandb.Audio(
                                reflow_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="reflow",
                            ),
                            wandb.Audio(
                                aux_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="aux",
                            ),
                            wandb.Audio(
                                recon_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="recon",
                            ),
                        ],
                    },
                )

            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_figure(
                    f"sample-{idx}/mels",
                    image_mels,
                    global_step=self.global_step,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/gt",
                    audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/reflow",
                    reflow_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/aux",
                    aux_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/recon",
                    recon_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)
