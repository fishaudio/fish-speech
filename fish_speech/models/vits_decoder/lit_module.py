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

from fish_speech.models.vits_decoder.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from fish_speech.models.vqgan.utils import (
    avg_with_mask,
    plot_mel,
    sequence_mask,
    slice_segments,
)


class VITSDecoder(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: nn.Module,
        discriminator: nn.Module,
        mel_transform: nn.Module,
        spec_transform: nn.Module,
        hop_length: int = 512,
        sample_rate: int = 44100,
        freeze_discriminator: bool = False,
        weight_mel: float = 45,
        weight_kl: float = 0.1,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminator
        self.generator = generator
        self.discriminator = discriminator
        self.mel_transform = mel_transform
        self.spec_transform = spec_transform
        self.freeze_discriminator = freeze_discriminator

        # Loss weights
        self.weight_mel = weight_mel
        self.weight_kl = weight_kl

        # Other parameters
        self.hop_length = hop_length
        self.sampling_rate = sample_rate

        # Disable automatic optimization
        self.automatic_optimization = False

        if self.freeze_discriminator:
            for p in self.discriminator.parameters():
                p.requires_grad = False

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())
        optimizer_discriminator = self.optimizer_builder(
            self.discriminator.parameters()
        )

        lr_scheduler_generator = self.lr_scheduler_builder(optimizer_generator)
        lr_scheduler_discriminator = self.lr_scheduler_builder(optimizer_discriminator)

        return (
            {
                "optimizer": optimizer_generator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_generator,
                    "interval": "step",
                    "name": "optimizer/generator",
                },
            },
            {
                "optimizer": optimizer_discriminator,
                "lr_scheduler": {
                    "scheduler": lr_scheduler_discriminator,
                    "interval": "step",
                    "name": "optimizer/discriminator",
                },
            },
        )

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios)
            gt_specs = self.spec_transform(audios)

        spec_lengths = audio_lengths // self.hop_length
        spec_masks = torch.unsqueeze(
            sequence_mask(spec_lengths, gt_mels.shape[2]), 1
        ).to(gt_mels.dtype)

        (
            fake_audios,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        ) = self.generator(
            audios,
            audio_lengths,
            gt_specs,
            spec_lengths,
            texts,
            text_lengths,
        )

        gt_mels = slice_segments(gt_mels, ids_slice, self.generator.segment_size)
        spec_masks = slice_segments(spec_masks, ids_slice, self.generator.segment_size)
        audios = slice_segments(
            audios,
            ids_slice * self.hop_length,
            self.generator.segment_size * self.hop_length,
        )
        fake_mels = self.mel_transform(fake_audios.squeeze(1))

        assert (
            audios.shape == fake_audios.shape
        ), f"{audios.shape} != {fake_audios.shape}"

        # Discriminator
        if self.freeze_discriminator is False:
            y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(
                audios, fake_audios.detach()
            )

            with torch.autocast(device_type=audios.device.type, enabled=False):
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            self.log(
                f"train/discriminator/loss",
                loss_disc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            optim_d.zero_grad()
            self.manual_backward(loss_disc)
            self.clip_gradients(
                optim_d, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
            )
            optim_d.step()

        # Adv Loss
        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(audios, fake_audios)

        # Adversarial Loss
        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_adv, _ = generator_loss(y_d_hat_g)

        self.log(
            f"train/generator/adv",
            loss_adv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_fm = feature_loss(y_d_hat_r, y_d_hat_g)

        self.log(
            f"train/generator/adv_fm",
            loss_fm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_mel = avg_with_mask(
                F.l1_loss(gt_mels, fake_mels, reduction="none"), spec_masks
            )

        self.log(
            "train/generator/loss_mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, y_mask)

        self.log(
            "train/generator/loss_kl",
            loss_kl,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        loss = (
            loss_mel * self.weight_mel + loss_kl * self.weight_kl + loss_adv + loss_fm
        )
        self.log(
            "train/generator/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Backward
        optim_g.zero_grad()

        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch: Any, batch_idx: int):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        texts, text_lengths = batch["texts"], batch["text_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        gt_mels = self.mel_transform(audios)
        gt_specs = self.spec_transform(audios)
        spec_lengths = audio_lengths // self.hop_length
        spec_masks = torch.unsqueeze(
            sequence_mask(spec_lengths, gt_mels.shape[2]), 1
        ).to(gt_mels.dtype)

        prior_audios = self.generator.infer(
            audios, audio_lengths, gt_specs, spec_lengths, texts, text_lengths
        )
        posterior_audios = self.generator.infer_posterior(gt_specs, spec_lengths)
        prior_mels = self.mel_transform(prior_audios.squeeze(1))
        posterior_mels = self.mel_transform(posterior_audios.squeeze(1))

        min_mel_length = min(
            gt_mels.shape[-1], prior_mels.shape[-1], posterior_mels.shape[-1]
        )
        gt_mels = gt_mels[:, :, :min_mel_length]
        prior_mels = prior_mels[:, :, :min_mel_length]
        posterior_mels = posterior_mels[:, :, :min_mel_length]

        prior_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, prior_mels, reduction="none"), spec_masks
        )
        posterior_mel_loss = avg_with_mask(
            F.l1_loss(gt_mels, posterior_mels, reduction="none"), spec_masks
        )

        self.log(
            "val/prior_mel_loss",
            prior_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "val/posterior_mel_loss",
            posterior_mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # only log the first batch
        if batch_idx != 0:
            return

        for idx, (
            mel,
            prior_mel,
            posterior_mel,
            audio,
            prior_audio,
            posterior_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                prior_mels,
                posterior_mels,
                audios.detach().float(),
                prior_audios.detach().float(),
                posterior_audios.detach().float(),
                audio_lengths,
            )
        ):
            mel_len = audio_len // self.hop_length

            image_mels = plot_mel(
                [
                    prior_mel[:, :mel_len],
                    posterior_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Prior (VQ)",
                    "Posterior (Reconstruction)",
                    "Ground-Truth",
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
                                prior_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="prior",
                            ),
                            wandb.Audio(
                                posterior_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="posterior",
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
                    f"sample-{idx}/wavs/prior",
                    prior_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/posterior",
                    posterior_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)
