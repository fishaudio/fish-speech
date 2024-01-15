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
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from fish_speech.models.vqgan.losses import (
    MultiResolutionSTFTLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from fish_speech.models.vqgan.modules.convnext import ConvNeXt
from fish_speech.models.vqgan.modules.encoders import VQEncoder
from fish_speech.models.vqgan.utils import plot_mel, sequence_mask


@dataclass
class VQEncodeResult:
    features: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    feature_lengths: torch.Tensor


@dataclass
class VQDecodeResult:
    mels: torch.Tensor


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        encoder: ConvNeXt,
        vq: VQEncoder,
        decoder: ConvNeXt,
        generator: nn.Module,
        discriminator: ConvNeXt,
        mel_transform: nn.Module,
        hop_length: int = 640,
        sample_rate: int = 32000,
        freeze_discriminator: bool = False,
    ):
        super().__init__()

        # pretrain: vq use gt mel as target, hifigan use gt mel as input
        # finetune: end-to-end training, use gt mel as hifi gan target but freeze vq

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminator
        self.encoder = encoder
        self.vq = vq
        self.decoder = decoder
        self.generator = generator
        self.discriminator = discriminator
        self.mel_transform = mel_transform
        self.freeze_discriminator = freeze_discriminator

        # Crop length for saving memory
        self.hop_length = hop_length
        self.sampling_rate = sample_rate

        # Disable automatic optimization
        self.automatic_optimization = False

        if self.freeze_discriminator:
            for p in self.discriminator.parameters():
                p.requires_grad = False

        # Freeze generator
        for p in self.generator.parameters():
            p.requires_grad = False

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        optimizer_generator = self.optimizer_builder(
            itertools.chain(
                self.encoder.parameters(),
                self.vq.parameters(),
                self.decoder.parameters(),
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

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios, sample_rate=self.sampling_rate)

        mel_lengths = audio_lengths // self.hop_length
        mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
            gt_mels.dtype
        )

        vq_result = self.encode(audios, audio_lengths)
        loss_vq = vq_result.loss

        if loss_vq.ndim > 1:
            loss_vq = loss_vq.mean()

        decoded_mels = self.decode(
            indices=None,
            features=vq_result.features,
            audio_lengths=audio_lengths,
        ).mels

        with torch.no_grad():
            with torch.autocast(device_type=audios.device.type, enabled=False):
                fake_audios = self.generator(decoded_mels.float())

        assert (
            audios.shape == fake_audios.shape
        ), f"{audios.shape} != {fake_audios.shape}"

        # Discriminator
        if self.freeze_discriminator is False:
            scores = self.discriminator(gt_mels)
            score_fakes = self.discriminator(decoded_mels.detach())

            with torch.autocast(device_type=audios.device.type, enabled=False):
                loss_disc, _, _ = discriminator_loss([scores], [score_fakes])

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
        score_fakes = self.discriminator(decoded_mels)

        # Adversarial Loss
        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_adv, _ = generator_loss([score_fakes])

        self.log(
            f"train/generator/adv",
            loss_adv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # Feature Matching Loss
        score_gts = self.discriminator(gt_mels)

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_fm = feature_loss([score_gts], [score_fakes])

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
            loss_mel = F.l1_loss(gt_mels * mel_masks, decoded_mels * mel_masks)

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
            "train/generator/loss_vq",
            loss_vq,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        loss = loss_mel * 20 + loss_vq + loss_adv + loss_fm
        self.log(
            "train/generator/loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

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

        audios = audios.float()
        audios = audios[:, None, :]

        gt_mels = self.mel_transform(audios, sample_rate=self.sampling_rate)
        mel_lengths = audio_lengths // self.hop_length
        mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
            gt_mels.dtype
        )

        vq_result = self.encode(audios, audio_lengths)
        decoded_mels = self.decode(
            indices=vq_result.indices,
            audio_lengths=audio_lengths,
        ).mels
        fake_audios = self.generator(decoded_mels)

        fake_mels = self.mel_transform(fake_audios.squeeze(1))

        min_mel_length = min(
            decoded_mels.shape[-1], gt_mels.shape[-1], fake_mels.shape[-1]
        )
        decoded_mels = decoded_mels[:, :, :min_mel_length]
        gt_mels = gt_mels[:, :, :min_mel_length]
        fake_mels = fake_mels[:, :, :min_mel_length]

        mel_loss = F.l1_loss(gt_mels * mel_masks, fake_mels * mel_masks)
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
            decode_mel,
            audio,
            gen_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                fake_mels,
                decoded_mels,
                audios.detach().float(),
                fake_audios.detach().float(),
                audio_lengths,
            )
        ):
            mel_len = audio_len // self.hop_length

            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    decode_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Generated",
                    "Decoded",
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
                                gen_audio[0, :audio_len],
                                sample_rate=self.sampling_rate,
                                caption="prediction",
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

            plt.close(image_mels)

    def encode(self, audios, audio_lengths=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor(
                [audios.shape[-1]] * audios.shape[0],
                device=audios.device,
                dtype=torch.long,
            )

        with torch.no_grad():
            features = self.mel_transform(audios, sample_rate=self.sampling_rate)

        feature_lengths = (
            audio_lengths
            / self.hop_length
            # / self.vq.downsample
        ).long()

        # print(features.shape, feature_lengths.shape, torch.max(feature_lengths))

        feature_masks = torch.unsqueeze(
            sequence_mask(feature_lengths, features.shape[2]), 1
        ).to(features.dtype)

        features = (
            gradient_checkpoint(
                self.encoder, features, feature_masks, use_reentrant=False
            )
            * feature_masks
        )
        vq_features, indices, loss = self.vq(features, feature_masks)

        return VQEncodeResult(
            features=vq_features,
            indices=indices,
            loss=loss,
            feature_lengths=feature_lengths,
        )

    def calculate_audio_lengths(self, feature_lengths):
        return feature_lengths * self.hop_length * self.vq.downsample

    def decode(
        self,
        indices=None,
        features=None,
        audio_lengths=None,
        feature_lengths=None,
    ):
        assert (
            indices is not None or features is not None
        ), "indices or features must be provided"
        assert (
            feature_lengths is not None or audio_lengths is not None
        ), "feature_lengths or audio_lengths must be provided"

        if audio_lengths is None:
            audio_lengths = self.calculate_audio_lengths(feature_lengths)

        mel_lengths = audio_lengths // self.hop_length
        mel_masks = torch.unsqueeze(
            sequence_mask(mel_lengths, torch.max(mel_lengths)), 1
        ).float()

        if indices is not None:
            features = self.vq.decode(indices)

        # Sample mels
        decoded = gradient_checkpoint(self.decoder, features, use_reentrant=False)

        return VQDecodeResult(
            mels=decoded,
        )
