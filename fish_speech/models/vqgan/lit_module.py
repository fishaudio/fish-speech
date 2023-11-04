import itertools
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
from vector_quantize_pytorch import ResidualLFQ

from fish_speech.models.vqgan.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss_normal,
)
from fish_speech.models.vqgan.modules.discriminator import EnsembleDiscriminator
from fish_speech.models.vqgan.modules.models import SynthesizerTrn
from fish_speech.models.vqgan.utils import plot_mel, sequence_mask, slice_segments


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        generator: SynthesizerTrn,
        discriminator: EnsembleDiscriminator,
        mel_transform: nn.Module,
        segment_size: int = 20480,
        hop_length: int = 640,
        sample_rate: int = 32000,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminators
        self.generator = generator
        self.discriminator = discriminator
        self.mel_transform = mel_transform

        # Crop length for saving memory
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.sampling_rate = sample_rate

        # Disable automatic optimization
        self.automatic_optimization = False

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
        features, feature_lengths = batch["features"], batch["feature_lengths"]
        audios = audios[:, None, :]

        audios = audios.float()
        features = features.float()

        with torch.no_grad():
            gt_mels = self.mel_transform(audios)
            assert (
                gt_mels.shape[2] == features.shape[1]
            ), f"Shapes do not match: {gt_mels.shape}, {features.shape}"

        (
            y_hat,
            ids_slice,
            x_mask,
            y_mask,
            (z_q_audio, z_p),
            (m_p_text, logs_p_text),
            (m_q, logs_q),
        ) = self.generator(features, feature_lengths, gt_mels)

        y_hat_mel = self.mel_transform(y_hat.squeeze(1))
        y_mel = slice_segments(gt_mels, ids_slice, self.segment_size // self.hop_length)
        y = slice_segments(audios, ids_slice * self.hop_length, self.segment_size)

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(y, y_hat.detach())

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_disc_all, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

        self.log(
            "train/discriminator/loss",
            loss_disc_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        optim_d.zero_grad()
        self.manual_backward(loss_disc_all)
        self.clip_gradients(
            optim_d, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_d.step()

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(y, y_hat)

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            loss_adv, _ = generator_loss(y_d_hat_g)
            loss_fm = feature_loss(fmap_r, fmap_g)
            # x_mask,
            # y_mask,
            # (z_q_audio, z_p),
            # (m_p_text, logs_p_text),
            # (m_q, logs_q),
            loss_kl = kl_loss_normal(
                m_q,
                logs_q,
                m_p_text,
                logs_p_text,
                x_mask,
            )

            loss_gen_all = loss_mel * 45 + loss_fm + loss_adv + loss_kl * 0.05

        self.log(
            "train/generator/loss",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
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
        self.log(
            "train/generator/loss_fm",
            loss_fm,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/loss_adv",
            loss_adv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/loss_kl",
            loss_kl,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        # self.log(
        #     "train/generator/loss_vq",
        #     prior.loss,
        #     on_step=True,
        #     on_epoch=False,
        #     prog_bar=False,
        #     logger=True,
        #     sync_dist=True,
        # )

        optim_g.zero_grad()
        self.manual_backward(loss_gen_all)
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
        features, feature_lengths = batch["features"], batch["feature_lengths"]

        audios = audios.float()
        features = features.float()
        audios = audios[:, None, :]

        gt_mels = self.mel_transform(audios)
        assert (
            gt_mels.shape[2] == features.shape[1]
        ), f"Shapes do not match: {gt_mels.shape}, {features.shape}"

        fake_audios = self.generator.infer(features, feature_lengths, gt_mels)
        posterior_audios = self.generator.reconstruct(gt_mels, feature_lengths)

        fake_mels = self.mel_transform(fake_audios.squeeze(1))
        posterior_mels = self.mel_transform(posterior_audios.squeeze(1))

        min_mel_length = min(gt_mels.shape[-1], fake_mels.shape[-1])
        gt_mels = gt_mels[:, :, :min_mel_length]
        fake_mels = fake_mels[:, :, :min_mel_length]
        posterior_mels = posterior_mels[:, :, :min_mel_length]

        mel_loss = F.l1_loss(gt_mels, fake_mels)
        self.log(
            "val/mel_loss",
            mel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for idx, (
            mel,
            gen_mel,
            post_mel,
            audio,
            gen_audio,
            post_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                fake_mels,
                posterior_mels,
                audios,
                fake_audios,
                posterior_audios,
                audio_lengths,
            )
        ):
            mel_len = audio_len // self.hop_length

            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    post_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Generated Spectrogram",
                    "Posterior Spectrogram",
                    "Ground-Truth Spectrogram",
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
                                gen_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="prediction",
                            ),
                            wandb.Audio(
                                post_audio[0, :audio_len],
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
                    f"sample-{idx}/wavs/prediction",
                    gen_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )
                self.logger.experiment.add_audio(
                    f"sample-{idx}/wavs/posterior",
                    post_audio[0, :audio_len],
                    self.global_step,
                    sample_rate=self.sampling_rate,
                )

            plt.close(image_mels)
