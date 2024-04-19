import itertools
import math
from typing import Any, Callable

import lightning as L
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn

from fish_speech.models.vqgan.modules.discriminator import Discriminator
from fish_speech.models.vqgan.modules.wavenet import WaveNet
from fish_speech.models.vqgan.utils import avg_with_mask, plot_mel, sequence_mask


class VQGAN(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        encoder: WaveNet,
        quantizer: nn.Module,
        decoder: WaveNet,
        discriminator: Discriminator,
        vocoder: nn.Module,
        encode_mel_transform: nn.Module,
        gt_mel_transform: nn.Module,
        weight_adv: float = 1.0,
        weight_vq: float = 1.0,
        weight_mel: float = 1.0,
        sampling_rate: int = 44100,
        freeze_encoder: bool = False,
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
        self.discriminator = discriminator
        self.encode_mel_transform = encode_mel_transform
        self.gt_mel_transform = gt_mel_transform

        # A simple linear layer to project quality to condition channels
        self.quality_projection = nn.Linear(1, 768)

        # Freeze vocoder
        for param in self.vocoder.parameters():
            param.requires_grad = False

        # Loss weights
        self.weight_adv = weight_adv
        self.weight_vq = weight_vq
        self.weight_mel = weight_mel

        # Other parameters
        self.sampling_rate = sampling_rate

        # Disable strict loading
        self.strict_loading = False

        # If encoder is frozen
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

            for param in self.quantizer.parameters():
                param.requires_grad = False

        self.automatic_optimization = False

    def on_save_checkpoint(self, checkpoint):
        # Do not save vocoder
        state_dict = checkpoint["state_dict"]
        for name in list(state_dict.keys()):
            if "vocoder" in name:
                state_dict.pop(name)

    def configure_optimizers(self):
        optimizer_generator = self.optimizer_builder(
            itertools.chain(
                self.encoder.parameters(),
                self.quantizer.parameters(),
                self.decoder.parameters(),
                self.quality_projection.parameters(),
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
            encoded_mels = self.encode_mel_transform(audios)
            gt_mels = self.gt_mel_transform(audios)
            quality = ((gt_mels.mean(-1) > -8).sum(-1) - 90) / 10
            quality = quality.unsqueeze(-1)

        mel_lengths = audio_lengths // self.gt_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        gt_mels = gt_mels * mel_masks_float_conv
        encoded_mels = encoded_mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(encoded_mels) * mel_masks_float_conv

        # Quantize
        vq_result = self.quantizer(encoded_features)
        loss_vq = getattr("vq_result", "loss", 0.0)
        vq_recon_features = vq_result.z * mel_masks_float_conv
        vq_recon_features = (
            vq_recon_features + self.quality_projection(quality)[:, :, None]
        )

        # VQ Decode
        gen_mel = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features,
            )
            * mel_masks_float_conv
        )

        # Discriminator
        real_logits = self.discriminator(gt_mels)
        fake_logits = self.discriminator(gen_mel.detach())
        d_mask = F.interpolate(
            mel_masks_float_conv, size=(real_logits.shape[2],), mode="nearest"
        )

        loss_real = avg_with_mask((real_logits - 1) ** 2, d_mask)
        loss_fake = avg_with_mask(fake_logits**2, d_mask)

        loss_d = loss_real + loss_fake

        self.log(
            "train/discriminator/loss",
            loss_d,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Discriminator backward
        optim_d.zero_grad()
        self.manual_backward(loss_d)
        self.clip_gradients(
            optim_d, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_d.step()

        # Mel Loss, applying l1, using a weighted sum
        mel_distance = (
            gen_mel - gt_mels
        ).abs()  # * 0.5 + self.ssim(gen_mel, gt_mels) * 0.5
        loss_mel_low_freq = avg_with_mask(mel_distance[:, :40, :], mel_masks_float_conv)
        loss_mel_mid_freq = avg_with_mask(
            mel_distance[:, 40:70, :], mel_masks_float_conv
        )
        loss_mel_high_freq = avg_with_mask(
            mel_distance[:, 70:, :], mel_masks_float_conv
        )
        loss_mel = (
            loss_mel_low_freq * 0.6 + loss_mel_mid_freq * 0.3 + loss_mel_high_freq * 0.1
        )

        # Adversarial Loss
        fake_logits = self.discriminator(gen_mel)
        loss_adv = avg_with_mask((fake_logits - 1) ** 2, d_mask)

        # Total loss
        loss = (
            self.weight_vq * loss_vq
            + self.weight_mel * loss_mel
            + self.weight_adv * loss_adv
        )

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
        self.log(
            "train/generator/loss_adv",
            loss_adv,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # Generator backward
        optim_g.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(
            optim_g, gradient_clip_val=1000.0, gradient_clip_algorithm="norm"
        )
        optim_g.step()

        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch: Any, batch_idx: int):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]

        audios = audios.float()
        audios = audios[:, None, :]

        encoded_mels = self.encode_mel_transform(audios)
        gt_mels = self.gt_mel_transform(audios)

        mel_lengths = audio_lengths // self.gt_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, gt_mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        gt_mels = gt_mels * mel_masks_float_conv
        encoded_mels = encoded_mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(encoded_mels) * mel_masks_float_conv

        # Quantize
        vq_recon_features = self.quantizer(encoded_features).z * mel_masks_float_conv
        vq_recon_features = (
            vq_recon_features
            + self.quality_projection(
                torch.ones(
                    vq_recon_features.shape[0], 1, device=vq_recon_features.device
                )
                * 2
            )[:, :, None]
        )

        # VQ Decode
        gen_aux_mels = (
            self.decoder(
                torch.randn_like(vq_recon_features) * mel_masks_float_conv,
                condition=vq_recon_features,
            )
            * mel_masks_float_conv
        )
        loss_mel = avg_with_mask((gen_aux_mels - gt_mels).abs(), mel_masks_float_conv)

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

            mel_len = audio_len // self.gt_mel_transform.hop_length

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

        mels = self.encode_mel_transform(audios)
        mel_lengths = audio_lengths // self.encode_mel_transform.hop_length
        mel_masks = sequence_mask(mel_lengths, mels.shape[2])
        mel_masks_float_conv = mel_masks[:, None, :].float()
        mels = mels * mel_masks_float_conv

        # Encode
        encoded_features = self.encoder(mels) * mel_masks_float_conv
        feature_lengths = mel_lengths // math.prod(self.quantizer.downsample_factor)

        return self.quantizer.encode(encoded_features), feature_lengths

    def decode(self, indices, feature_lengths, return_audios=False):
        factor = math.prod(self.quantizer.downsample_factor)
        mel_masks = sequence_mask(feature_lengths * factor, indices.shape[2] * factor)
        mel_masks_float_conv = mel_masks[:, None, :].float()

        z = self.quantizer.decode(indices) * mel_masks_float_conv
        z = (
            z
            + self.quality_projection(torch.ones(z.shape[0], 1, device=z.device) * 2)[
                :, :, None
            ]
        )

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
