import itertools
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

from fish_speech.models.vqgan.modules import (
    EnsembleDiscriminator,
    Generator,
    PosteriorEncoder,
    SemanticEncoder,
    SpeakerEncoder,
)
from fish_speech.models.vqgan.utils import plot_mel, sequence_mask


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        semantic_encoder: SemanticEncoder,
        posterior_encoder: PosteriorEncoder,
        speaker_encoder: SpeakerEncoder,
        # flow: nn.Module,
        generator: Generator,
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
        # Compile generator so that snake can save memory
        self.semantic_encoder = semantic_encoder
        self.posterior_encoder = posterior_encoder
        self.speaker_encoder = speaker_encoder
        # self.flow = flow
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
        optimizer_generator = self.optimizer_builder(
            itertools.chain(
                self.semantic_encoder.parameters(),
                self.posterior_encoder.parameters(),
                self.speaker_encoder.parameters(),
                self.generator.parameters(),
                # self.flow.parameters(),
            )
        )
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

    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2

    @staticmethod
    def kl_loss(m_q, logs_q, m_p, logs_p, z_mask):
        """
        m_q, logs_q: [b, h, t_t]
        m_p, logs_p: [b, h, t_t]
        """
        m_q = m_q.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        z_mask = z_mask.float()

        kl = 0.5 * (
            (m_q - m_p) ** 2 / torch.exp(logs_p)
            + torch.exp(logs_q) / torch.exp(logs_p)
            - 1
            - logs_q
            + logs_p
        )

        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)

        return l

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        features, feature_lengths = batch["features"], batch["feature_lengths"]

        audios = audios.float()
        features = features.float()

        with torch.no_grad():
            gt_mels, gt_specs = self.mel_transform(audios, return_linear=True)
            gt_mels = gt_mels.transpose(1, 2)
            key_padding_mask = sequence_mask(feature_lengths)
            mels_key_padding_mask = sequence_mask(audio_lengths // self.hop_length)
            audio_masks = sequence_mask(audio_lengths)[:, None]

            assert abs(gt_mels.shape[1] - mels_key_padding_mask.shape[1]) <= 1
            gt_mel_length = min(gt_mels.shape[1], mels_key_padding_mask.shape[1])
            gt_mels = gt_mels[:, :gt_mel_length]
            gt_specs = gt_specs[:, :, :gt_mel_length]
            mels_key_padding_mask = mels_key_padding_mask[:, :gt_mel_length]

            assert abs(features.shape[1] - key_padding_mask.shape[1]) <= 1
            gt_feature_length = min(features.shape[1], key_padding_mask.shape[1])
            features = features[:, :gt_feature_length]
            key_padding_mask = key_padding_mask[:, :gt_feature_length]

        audios = audios[:, None, :]

        speaker = self.speaker_encoder(gt_mels, mels_key_padding_mask)[:, :, None]
        prior = self.semantic_encoder(
            x=features,
            key_padding_mask=key_padding_mask,
            g=speaker,
        )

        posterior_key_padding_mask = (~mels_key_padding_mask).float()[:, None]
        posterior = self.posterior_encoder(
            gt_specs, posterior_key_padding_mask, g=speaker
        )
        # z_p = self.flow(posterior.mean, posterior_key_padding_mask, g=speaker)
        fake_audios = self.generator(posterior.z, g=speaker)

        min_audio_length = min(audios.shape[-1], fake_audios.shape[-1])
        audios = audios[:, :, :min_audio_length]
        fake_audios = fake_audios[:, :, :min_audio_length]
        audio_masks = audio_masks[:, :, :min_audio_length]

        audio = torch.masked_fill(audios, audio_masks, 0.0)
        fake_audios = torch.masked_fill(fake_audios, audio_masks, 0.0)
        assert fake_audios.shape == audio.shape

        # Discriminator
        y_d_hat_r, y_d_hat_g, _, _ = self.discriminator(audio, fake_audios.detach())

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_disc_all, _, _ = self.discriminator_loss(y_d_hat_r, y_d_hat_g)

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

        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.discriminator(audios, fake_audios)
        fake_mels = self.mel_transform(fake_audios.squeeze(1)).transpose(1, 2)

        # Min mel length
        min_mel_length = min(gt_mels.shape[1], fake_mels.shape[1])
        gt_mels = gt_mels[:, :min_mel_length]
        fake_mels = fake_mels[:, :min_mel_length]
        mels_key_padding_mask = mels_key_padding_mask[:, :min_mel_length]

        # Fill mel mask
        fake_mels = torch.masked_fill(fake_mels, mels_key_padding_mask[:, :, None], 0.0)
        gt_mels = torch.masked_fill(gt_mels, mels_key_padding_mask[:, :, None], 0.0)

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_mel = F.l1_loss(gt_mels, fake_mels)
            loss_adv, _ = self.generator_loss(y_d_hat_g)
            loss_fm = self.feature_loss(fmap_r, fmap_g)
            loss_kl = self.kl_loss(
                posterior.mean,
                posterior.logs,
                prior.mean,
                prior.logs,
                posterior_key_padding_mask,
            )

            loss_gen_all = loss_mel * 45 + loss_fm + loss_adv + prior.loss + loss_kl

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
        self.log(
            "train/generator/loss_vq",
            prior.loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

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

        with torch.no_grad():
            gt_mels, gt_specs = self.mel_transform(audios, return_linear=True)
            gt_mels = gt_mels.transpose(1, 2)
            key_padding_mask = sequence_mask(feature_lengths)
            mels_key_padding_mask = sequence_mask(audio_lengths // self.hop_length)

            assert abs(gt_mels.shape[1] - mels_key_padding_mask.shape[1]) <= 1
            gt_mel_length = min(gt_mels.shape[1], mels_key_padding_mask.shape[1])
            gt_mels = gt_mels[:, :gt_mel_length]
            gt_specs = gt_specs[:, :, :gt_mel_length]
            mels_key_padding_mask = mels_key_padding_mask[:, :gt_mel_length]

            assert abs(features.shape[1] - key_padding_mask.shape[1]) <= 1
            gt_feature_length = min(features.shape[1], key_padding_mask.shape[1])
            features = features[:, :gt_feature_length]
            key_padding_mask = key_padding_mask[:, :gt_feature_length]

        # Generator
        # speaker: (B, C, 1)
        speaker = self.speaker_encoder(gt_mels, mels_key_padding_mask)[:, :, None]
        posterior_key_padding_mask = (~mels_key_padding_mask).float()[:, None]

        z_gen = self.semantic_encoder(
            x=features,
            key_padding_mask=key_padding_mask,
            g=speaker,
        ).z

        # z_gen = self.flow(z_gen, posterior_key_padding_mask, g=speaker, reverse=True)

        z_posterior = self.posterior_encoder(
            gt_specs, posterior_key_padding_mask, g=speaker
        ).mean

        audios = audios[:, None, :]
        fake_audios = self.generator(z_gen, g=speaker)
        posterior_audios = self.generator(z_posterior)
        min_audio_length = min(
            audios.shape[-1], fake_audios.shape[-1], posterior_audios.shape[-1]
        )

        audios = audios[:, :, :min_audio_length]
        fake_audios = fake_audios[:, :, :min_audio_length]
        posterior_audios = posterior_audios[:, :, :min_audio_length]
        assert fake_audios.shape == audios.shape == posterior_audios.shape

        fake_mels = self.mel_transform(fake_audios.squeeze(1)).transpose(1, 2)
        posterior_mels = self.mel_transform(posterior_audios.squeeze(1)).transpose(1, 2)

        min_mel_length = min(gt_mels.shape[1], fake_mels.shape[1])
        gt_mels = gt_mels[:, :min_mel_length]
        fake_mels = fake_mels[:, :min_mel_length]
        posterior_mels = posterior_mels[:, :min_mel_length]

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
                gt_mels.transpose(1, 2),
                fake_mels.transpose(1, 2),
                posterior_mels.transpose(1, 2),
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
