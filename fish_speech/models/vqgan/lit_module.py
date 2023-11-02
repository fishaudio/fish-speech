from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as gradient_checkpointing


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        encoder: nn.Module,
        generator: nn.Module,
        discriminator: nn.Module,
        mel_transform: nn.Module,
        segment_size: int = 20480,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminators
        # Compile generator so that snake can save memory
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.mel_transform = mel_transform

        # Crop length for saving memory
        self.segment_size = segment_size

        # Disable automatic optimization
        self.automatic_optimization = False

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(self.generator.parameters())
        optimizer_discriminator = self.optimizer_builder(
            self.discriminators.parameters()
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

    def training_generator(self, audio, audio_mask):
        # fake_audio, base_loss = self.forward(audio, audio_mask)

        assert fake_audio.shape == audio.shape

        # Apply mask
        audio = audio * audio_mask
        fake_audio = fake_audio * audio_mask

        # Multi-Resolution STFT Loss
        sc_loss, mag_loss = self.multi_resolution_stft_loss(
            fake_audio.squeeze(1), audio.squeeze(1)
        )
        loss_stft = sc_loss + mag_loss

        self.log(
            "train/generator/stft",
            loss_stft,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # L1 Mel-Spectrogram Loss
        # This is not used in backpropagation currently
        audio_mel = self.mel_transforms.loss(audio.squeeze(1))
        fake_audio_mel = self.mel_transforms.loss(fake_audio.squeeze(1))
        loss_mel = F.l1_loss(audio_mel, fake_audio_mel)

        self.log(
            "train/generator/mel",
            loss_mel,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Now, we need to reduce the length of the audio to save memory
        if self.crop_length is not None and audio.shape[2] > self.crop_length:
            slice_idx = torch.randint(0, audio.shape[-1] - self.crop_length, (1,))

            audio = audio[..., slice_idx : slice_idx + self.crop_length]
            fake_audio = fake_audio[..., slice_idx : slice_idx + self.crop_length]
            audio_mask = audio_mask[..., slice_idx : slice_idx + self.crop_length]

            assert audio.shape == fake_audio.shape == audio_mask.shape

        # Adv Loss
        loss_adv_all = 0

        for key, disc in self.discriminators.items():
            score_fakes, feat_fake = disc(fake_audio)

            # Adversarial Loss
            score_fakes = torch.cat(score_fakes, dim=1)
            loss_fake = torch.mean((1 - score_fakes) ** 2)

            self.log(
                f"train/generator/adv_{key}",
                loss_fake,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_adv_all += loss_fake

            if self.feature_matching is False:
                continue

            # Feature Matching Loss
            _, feat_real = disc(audio)
            loss_fm = 0
            for dr, dg in zip(feat_real, feat_fake):
                for rl, gl in zip(dr, dg):
                    loss_fm += F.l1_loss(rl, gl)

            loss_fm /= len(feat_real)

            self.log(
                f"train/generator/adv_fm_{key}",
                loss_fm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_adv_all += loss_fm

        loss_adv_all /= len(self.discriminators)
        loss_gen_all = base_loss + loss_stft * 2.5 + loss_mel * 45 + loss_adv_all

        self.log(
            "train/generator/all",
            loss_gen_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss_gen_all, audio, fake_audio

    def training_discriminator(self, audio, fake_audio):
        loss_disc_all = 0

        for key, disc in self.discriminators.items():
            if self.training and self.checkpointing:
                scores, _ = gradient_checkpointing(disc, audio, use_reentrant=False)
                score_fakes, _ = gradient_checkpointing(
                    disc, fake_audio.detach(), use_reentrant=False
                )
            else:
                scores, _ = disc(audio)
                score_fakes, _ = disc(fake_audio.detach())

            scores = torch.cat(scores, dim=1)
            score_fakes = torch.cat(score_fakes, dim=1)
            loss_disc = torch.mean((scores - 1) ** 2) + torch.mean((score_fakes) ** 2)

            self.log(
                f"train/discriminator/{key}",
                loss_disc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_disc_all += loss_disc

        loss_disc_all /= len(self.discriminators)

        self.log(
            "train/discriminator/all",
            loss_disc_all,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss_disc_all

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audio, lengths = batch["audio"], batch["lengths"]
        audio_mask = sequence_mask(lengths)[:, None, :].to(audio.device, torch.float32)

        # Generator
        optim_g.zero_grad()
        loss_gen_all, audio, fake_audio = self.training_generator(audio, audio_mask)
        self.manual_backward(loss_gen_all)

        self.log(
            "train/generator/grad_norm",
            grad_norm(self.generator.parameters()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        self.clip_gradients(
            optim_g, gradient_clip_val=1000, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        # Discriminator
        assert fake_audio.shape == audio.shape

        optim_d.zero_grad()
        loss_disc_all = self.training_discriminator(audio, fake_audio)
        self.manual_backward(loss_disc_all)

        for key, disc in self.discriminators.items():
            self.log(
                f"train/discriminator/grad_norm_{key}",
                grad_norm(disc.parameters()),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        self.clip_gradients(
            optim_d, gradient_clip_val=1000, gradient_clip_algorithm="norm"
        )
        optim_d.step()

        # Manual LR Scheduler
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch: Any, batch_idx: int):
        audio, lengths = batch["audio"], batch["lengths"]
        audio_mask = sequence_mask(lengths)[:, None, :].to(audio.device, torch.float32)

        # Generator
        fake_audio, _ = self.forward(audio, audio_mask)
        assert fake_audio.shape == audio.shape

        # Apply mask
        audio = audio * audio_mask
        fake_audio = fake_audio * audio_mask

        # L1 Mel-Spectrogram Loss
        audio_mel = self.mel_transforms.loss(audio.squeeze(1))
        fake_audio_mel = self.mel_transforms.loss(fake_audio.squeeze(1))
        loss_mel = F.l1_loss(audio_mel, fake_audio_mel)

        self.log(
            "val/metrics/mel",
            loss_mel,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Report other metrics
        self.report_val_metrics(fake_audio, audio, lengths)
