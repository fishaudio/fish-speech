import itertools
from dataclasses import dataclass
from typing import Any, Callable, Literal

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

from fish_speech.models.vqgan.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from fish_speech.models.vqgan.modules.balancer import Balancer
from fish_speech.models.vqgan.modules.decoder import Generator
from fish_speech.models.vqgan.modules.encoders import (
    ConvDownSampler,
    TextEncoder,
    VQEncoder,
)
from fish_speech.models.vqgan.utils import (
    plot_mel,
    rand_slice_segments,
    sequence_mask,
    slice_segments,
)


@dataclass
class VQEncodeResult:
    features: torch.Tensor
    indices: torch.Tensor
    loss: torch.Tensor
    feature_lengths: torch.Tensor


@dataclass
class VQDecodeResult:
    audios: torch.Tensor
    mels: torch.Tensor
    mel_lengths: torch.Tensor


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        downsample: ConvDownSampler,
        vq_encoder: VQEncoder,
        mel_encoder: TextEncoder,
        decoder: TextEncoder,
        generator: Generator,
        discriminators: nn.ModuleDict,
        mel_transform: nn.Module,
        segment_size: int = 20480,
        hop_length: int = 640,
        sample_rate: int = 32000,
        mode: Literal["pretrain", "finetune"] = "finetune",
    ):
        super().__init__()

        # pretrain: vq use gt mel as target, hifigan use gt mel as input
        # finetune: end-to-end training, use gt mel as hifi gan target but freeze vq

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminators
        self.downsample = downsample
        self.vq_encoder = vq_encoder
        self.mel_encoder = mel_encoder
        self.decoder = decoder
        self.generator = generator
        self.discriminators = discriminators
        self.mel_transform = mel_transform

        # Crop length for saving memory
        self.segment_size = segment_size
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.mode = mode

        # Disable automatic optimization
        self.automatic_optimization = False

        # Finetune: Train the VQ only
        if self.mode == "finetune":
            for p in self.vq_encoder.parameters():
                p.requires_grad = False

            for p in self.mel_encoder.parameters():
                p.requires_grad = False

            for p in self.downsample.parameters():
                p.requires_grad = False

        self.balancer = Balancer(
            {
                "mel": 1,
                "adv": 1,
                "fm": 1,
            }
        )

    def configure_optimizers(self):
        # Need two optimizers and two schedulers
        components = [
            self.downsample.parameters(),
            self.vq_encoder.parameters(),
            self.mel_encoder.parameters(),
        ]

        if self.decoder is not None:
            components.append(self.decoder.parameters())

        components.append(self.generator.parameters())
        optimizer_generator = self.optimizer_builder(itertools.chain(*components))
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

    def training_step(self, batch, batch_idx):
        optim_g, optim_d = self.optimizers()

        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios, sample_rate=self.sampling_rate)

        if self.mode == "finetune":
            # Disable gradient computation for VQ
            torch.set_grad_enabled(False)
            self.vq_encoder.eval()
            self.mel_encoder.eval()
            self.downsample.eval()

        mel_lengths = audio_lengths // self.hop_length
        mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
            gt_mels.dtype
        )

        vq_result = self.encode(audios, audio_lengths)
        loss_vq = vq_result.loss

        if loss_vq.ndim > 1:
            loss_vq = loss_vq.mean()

        if self.mode == "finetune":
            # Enable gradient computation
            torch.set_grad_enabled(True)

        decoded = self.decode(
            indices=vq_result.indices if self.mode == "finetune" else None,
            features=vq_result.features if self.mode == "pretrain" else None,
            audio_lengths=audio_lengths,
            mel_only=True,
        )
        decoded_mels = decoded.mels
        input_mels = gt_mels if self.mode == "pretrain" else decoded_mels

        if self.segment_size is not None:
            audios, ids_slice = rand_slice_segments(
                audios, audio_lengths, self.segment_size
            )
            input_mels = slice_segments(
                input_mels,
                ids_slice // self.hop_length,
                self.segment_size // self.hop_length,
            )
            sliced_gt_mels = slice_segments(
                gt_mels,
                ids_slice // self.hop_length,
                self.segment_size // self.hop_length,
            )
            gen_mel_masks = slice_segments(
                mel_masks,
                ids_slice // self.hop_length,
                self.segment_size // self.hop_length,
            )
        else:
            sliced_gt_mels = gt_mels
            gen_mel_masks = mel_masks

        fake_audios = self.generator(input_mels)
        fake_audio_mels = self.mel_transform(fake_audios.squeeze(1))
        assert (
            audios.shape == fake_audios.shape
        ), f"{audios.shape} != {fake_audios.shape}"

        # Discriminator
        loss_disc_all = []

        for key, disc in self.discriminators.items():
            scores, _ = disc(audios)
            score_fakes, _ = disc(fake_audios.detach())

            with torch.autocast(device_type=audios.device.type, enabled=False):
                loss_disc, _, _ = discriminator_loss(scores, score_fakes)

            self.log(
                f"train/discriminator/{key}",
                loss_disc,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_disc_all.append(loss_disc)

        loss_disc_all = torch.stack(loss_disc_all).mean()

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

        # Adv Loss
        loss_adv_all = []
        loss_fm_all = []

        for key, disc in self.discriminators.items():
            score_fakes, feat_fake = disc(fake_audios)

            # Adversarial Loss
            with torch.autocast(device_type=audios.device.type, enabled=False):
                loss_fake, _ = generator_loss(score_fakes)

            self.log(
                f"train/generator/adv_{key}",
                loss_fake,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_adv_all.append(loss_fake)

            # Feature Matching Loss
            _, feat_real = disc(audios)

            with torch.autocast(device_type=audios.device.type, enabled=False):
                loss_fm = feature_loss(feat_real, feat_fake)

            self.log(
                f"train/generator/adv_fm_{key}",
                loss_fm,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            loss_fm_all.append(loss_fm)

        loss_adv_all = torch.stack(loss_adv_all).mean()
        loss_fm_all = torch.stack(loss_fm_all).mean()

        with torch.autocast(device_type=audios.device.type, enabled=False):
            loss_decoded_mel = F.l1_loss(gt_mels * mel_masks, decoded_mels * mel_masks)
            loss_mel = F.l1_loss(
                sliced_gt_mels * gen_mel_masks, fake_audio_mels * gen_mel_masks
            )

            generator_out_grad = self.balancer.compute(
                {
                    "mel": loss_mel,
                    "adv": loss_adv_all,
                    "fm": loss_fm_all,
                },
                fake_audios,
            )

            if self.mode == "pretrain":
                loss_vq_all = loss_decoded_mel + loss_vq

        # Loss vq and loss decoded mel are only used in pretrain stage
        if self.mode == "pretrain":
            self.log(
                "train/generator/loss_vq",
                loss_vq,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            self.log(
                "train/generator/loss_decoded_mel",
                loss_decoded_mel,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
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
            "train/generator/loss_fm_all",
            loss_fm_all,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/generator/loss_adv_all",
            loss_adv_all,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        optim_g.zero_grad()

        # Only backpropagate loss_vq_all in pretrain stage
        if self.mode == "pretrain":
            self.manual_backward(loss_vq_all, retain_graph=True)

        self.manual_backward(fake_audios, gradient=generator_out_grad)
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
        decoded = self.decode(
            indices=vq_result.indices,
            audio_lengths=audio_lengths,
            mel_only=self.mode == "pretrain",
        )

        decoded_mels = decoded.mels

        # Use gt mel as input for pretrain
        if self.mode == "pretrain":
            fake_audios = self.generator(gt_mels)
        else:
            fake_audios = decoded.audios

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

        if self.downsample is not None:
            features = self.downsample(features)

        feature_lengths = (
            audio_lengths
            / self.hop_length
            / (self.downsample.total_strides if self.downsample is not None else 1)
        ).long()

        feature_masks = torch.unsqueeze(
            sequence_mask(feature_lengths, features.shape[2]), 1
        ).to(features.dtype)

        text_features = self.mel_encoder(features, feature_masks)
        vq_features, indices, loss = self.vq_encoder(text_features, feature_masks)

        return VQEncodeResult(
            features=vq_features,
            indices=indices,
            loss=loss,
            feature_lengths=feature_lengths,
        )

    def calculate_audio_lengths(self, feature_lengths):
        return (
            feature_lengths
            * self.hop_length
            * (self.downsample.total_strides if self.downsample is not None else 1)
        )

    def decode(
        self,
        indices=None,
        features=None,
        audio_lengths=None,
        mel_only=False,
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
            features = self.vq_encoder.decode(indices)

        features = F.interpolate(features, size=mel_masks.shape[2], mode="nearest")

        # Sample mels
        if self.decoder is not None:
            decoded_mels = self.decoder(features, mel_masks)
        else:
            decoded_mels = features

        if mel_only:
            return VQDecodeResult(
                audios=None,
                mels=decoded_mels,
                mel_lengths=mel_lengths,
            )

        fake_audios = self.generator(decoded_mels)
        return VQDecodeResult(
            audios=fake_audios,
            mels=decoded_mels,
            mel_lengths=mel_lengths,
        )
