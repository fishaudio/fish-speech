import math
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

from fish_speech.models.vqgan.modules.wavenet import WaveNet
from fish_speech.models.vqgan.utils import plot_mel, sequence_mask


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        encoder: WaveNet,
        quantizer: nn.Module,
        decoder: WaveNet,
        # reflow: nn.Module,
        vocoder: nn.Module,
        mel_transform: nn.Module,
        weight_reflow: float = 1.0,
        weight_vq: float = 1.0,
        weight_mel: float = 1.0,
        sampling_rate: int = 44100,
        freeze_encoder: bool = False,
        reflow_inference_steps: int = 10,
        reflow_inference_start_t: float = 0.5,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Modules
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.vocoder = vocoder
        # self.reflow = reflow
        self.mel_transform = mel_transform

        # Freeze vocoder
        for param in self.vocoder.parameters():
            param.requires_grad = False

        # Loss weights
        self.weight_reflow = weight_reflow
        self.weight_vq = weight_vq
        self.weight_mel = weight_mel

        # Other parameters
        self.spec_min = -12
        self.spec_max = 3
        self.sampling_rate = sampling_rate
        self.reflow_inference_steps = reflow_inference_steps
        self.reflow_inference_start_t = reflow_inference_start_t

        # Disable strict loading
        self.strict_loading = False

        # If encoder is frozen
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.quantizer.parameters():
                param.requires_grad = False

    def on_save_checkpoint(self, checkpoint):
        # Do not save vocoder
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "vocoder" in name:
                state_dict.pop(name)

    def configure_optimizers(self):
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

    def training_step(self, batch, batch_idx):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios)

        mel_lengths = audio_lengths // self.mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        gt_mels = gt_mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(gt_mels) * mel_masks_float_conv

        # Quantize
        vq_result = self.quantizer(encoded_features)
        loss_vq = getattr("vq_result", "loss", 0.0)
        vq_recon_features = vq_result.z * mel_masks_float_conv

        # VQ Decode
        gen_mel = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features,
            )
            * mel_masks_float_conv
        )

        # Mel Loss
        loss_mel = (gen_mel - gt_mels).abs().mean(
            dim=1, keepdim=True
        ).sum() / mel_masks_float_conv.sum()

        # Total loss
        loss = self.weight_vq * loss_vq + self.weight_mel * loss_mel

        # Log losses
        self.log(
            "train/generator/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train/generator/loss_vq",
            loss_vq,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train/generator/loss_mel",
            loss_mel,
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
        gt_mels = gt_mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(gt_mels) * mel_masks_float_conv

        # Quantize
        vq_recon_features = self.quantizer(encoded_features).z * mel_masks_float_conv

        # VQ Decode
        gen_aux_mels = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features,
            )
            * mel_masks_float_conv
        )
        loss_mel = (gen_aux_mels - gt_mels).abs().mean(
            dim=1, keepdim=True
        ).sum() / mel_masks_float_conv.sum()

        self.log(
            "val/loss_mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        recon_audios = self.vocoder(gt_mels)
        gen_aux_audios = self.vocoder(gen_aux_mels)

        # only log the first batch
        if batch_idx != 0:
            return

        for idx, (
            gt_mel,
            gen_aux_mel,
            audio,
            gen_aux_audio,
            recon_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                gen_aux_mels,
                audios.cpu().float(),
                gen_aux_audios.cpu().float(),
                recon_audios.cpu().float(),
                audio_lengths,
            )
        ):
            if idx > 4:
                break

            mel_len = audio_len // self.mel_transform.hop_length

            image_mels = plot_mel(
                [
                    gt_mel[:, :mel_len],
                    gen_aux_mel[:, :mel_len],
                ],
                [
                    "Ground-Truth",
                    "Auxiliary",
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
                                gen_aux_audio[0, :audio_len],
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
                    f"sample-{idx}/wavs/gen",
                    gen_aux_audio[0, :audio_len],
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

    def encode(self, audios, audio_lengths):
        audios = audios.float()

        gt_mels = self.mel_transform(audios)
        mel_lengths = audio_lengths // self.mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        gt_mels = gt_mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(gt_mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // math.prod(self.quantizer.downsample_factor)

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths, return_audios=False):
        factor = math.prod(self.quantizer.downsample_factor)
        mel_masks = sequence_mask(feature_lengths * factor, indices.shape[2] * factor)
        mel_masks_float_conv = mel_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        gen_mel = (
            self.decoder(
                torch.randn_like(z) * mel_masks_float_conv,
                condition=z,
            )
            * mel_masks_float_conv
        )

        if return_audios:
            return self.vocoder(gen_mel)

        return gen_mel
