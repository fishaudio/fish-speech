import itertools
from typing import Any, Callable, Optional

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from diffusers.schedulers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from fish_speech.models.vq_diffusion.convnext_1d import ConvNext1DModel
from fish_speech.models.vqgan.modules.encoders import (
    SpeakerEncoder,
    TextEncoder,
    VQEncoder,
)
from fish_speech.models.vqgan.utils import plot_mel, sequence_mask


class ConvDownSample(nn.Module):
    def __init__(
        self,
        dims: list,
        kernel_sizes: list,
        strides: list,
    ):
        super().__init__()

        self.dims = dims
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.total_strides = np.prod(self.strides)

        self.convs = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            in_channels=self.dims[i],
                            out_channels=self.dims[i + 1],
                            kernel_size=self.kernel_sizes[i],
                            stride=self.strides[i],
                            padding=(self.kernel_sizes[i] - 1) // 2,
                        ),
                        nn.LayerNorm(self.dims[i + 1], elementwise_affine=True),
                        nn.GELU(),
                    ]
                )
                for i in range(len(self.dims) - 1)
            ]
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        for conv, norm, act in self.convs:
            x = conv(x)
            x = norm(x.mT).mT
            x = act(x)

        return x


class VQDiffusion(L.LightningModule):
    def __init__(
        self,
        optimizer: Callable,
        lr_scheduler: Callable,
        mel_transform: nn.Module,
        feature_mel_transform: nn.Module,
        vq_encoder: VQEncoder,
        speaker_encoder: SpeakerEncoder,
        text_encoder: TextEncoder,
        denoiser: ConvNext1DModel,
        vocoder: nn.Module,
        hop_length: int = 640,
        sample_rate: int = 32000,
        speaker_use_feats: bool = False,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()

        # Model parameters
        self.optimizer_builder = optimizer
        self.lr_scheduler_builder = lr_scheduler

        # Generator and discriminators
        self.mel_transform = mel_transform
        self.feature_mel_transform = feature_mel_transform
        self.noise_scheduler_train = DDIMScheduler(num_train_timesteps=1000)
        self.noise_scheduler_infer = UniPCMultistepScheduler(num_train_timesteps=1000)

        # Modules
        self.vq_encoder = vq_encoder
        self.speaker_encoder = speaker_encoder
        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.downsample = downsample

        self.vocoder = vocoder
        self.hop_length = hop_length
        self.sampling_rate = sample_rate
        self.speaker_use_feats = speaker_use_feats

        # Freeze vocoder
        for param in self.vocoder.parameters():
            param.requires_grad = False

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

    def normalize_mels(self, x):
        # x is in range -10.1 to 3.1, normalize to -1 to 1
        x_min, x_max = -10.1, 3.1
        return (x - x_min) / (x_max - x_min) * 2 - 1

    def denormalize_mels(self, x):
        x_min, x_max = -10.1, 3.1
        return (x + 1) / 2 * (x_max - x_min) + x_min

    def training_step(self, batch, batch_idx):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        features, feature_lengths = batch["features"], batch["feature_lengths"]

        audios = audios.float()
        # features = features.float().mT
        audios = audios[:, None, :]

        with torch.no_grad():
            gt_mels = self.mel_transform(audios, sample_rate=self.sampling_rate)
            features = self.feature_mel_transform(
                audios, sample_rate=self.sampling_rate
            )

        if self.downsample is not None:
            features = self.downsample(features)

        mel_lengths = audio_lengths // self.hop_length
        feature_lengths = (
            audio_lengths
            / self.sampling_rate
            * self.feature_mel_transform.sample_rate
            / self.feature_mel_transform.hop_length
            / (self.downsample.total_strides if self.downsample is not None else 1)
        ).long()

        feature_masks = torch.unsqueeze(
            sequence_mask(feature_lengths, features.shape[2]), 1
        ).to(gt_mels.dtype)
        mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
            gt_mels.dtype
        )

        if self.speaker_use_feats:
            speaker_features = self.speaker_encoder(features, feature_masks)
        else:
            speaker_features = self.speaker_encoder(gt_mels, mel_masks)

        # vq_features is 50 hz, need to convert to true mel size
        text_features = self.text_encoder(features, feature_masks)
        text_features, vq_loss = self.vq_encoder(text_features, feature_masks)
        text_features = F.interpolate(
            text_features, size=gt_mels.shape[2], mode="nearest"
        )

        text_features = text_features + speaker_features

        # Sample noise that we'll add to the images
        normalized_gt_mels = self.normalize_mels(gt_mels)
        noise = torch.randn_like(normalized_gt_mels)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.noise_scheduler_train.config.num_train_timesteps,
            (normalized_gt_mels.shape[0],),
            device=normalized_gt_mels.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler_train.add_noise(
            normalized_gt_mels, noise, timesteps
        )

        # Predict
        model_output = self.denoiser(noisy_images, timesteps, mel_masks, text_features)

        # MSE loss without the mask
        noise_loss = (torch.abs(model_output * mel_masks - noise * mel_masks)).sum() / (
            mel_masks.sum() * gt_mels.shape[1]
        )

        self.log(
            "train/noise_loss",
            noise_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "train/vq_loss",
            vq_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return noise_loss + vq_loss

    def validation_step(self, batch: Any, batch_idx: int):
        audios, audio_lengths = batch["audios"], batch["audio_lengths"]
        features, feature_lengths = batch["features"], batch["feature_lengths"]

        audios = audios.float()
        # features = features.float().mT
        audios = audios[:, None, :]
        gt_mels = self.mel_transform(audios, sample_rate=self.sampling_rate)
        features = self.feature_mel_transform(audios, sample_rate=self.sampling_rate)

        if self.downsample is not None:
            features = self.downsample(features)

        mel_lengths = audio_lengths // self.hop_length
        feature_lengths = (
            audio_lengths
            / self.sampling_rate
            * self.feature_mel_transform.sample_rate
            / self.feature_mel_transform.hop_length
            / (self.downsample.total_strides if self.downsample is not None else 1)
        ).long()

        feature_masks = torch.unsqueeze(
            sequence_mask(feature_lengths, features.shape[2]), 1
        ).to(gt_mels.dtype)
        mel_masks = torch.unsqueeze(sequence_mask(mel_lengths, gt_mels.shape[2]), 1).to(
            gt_mels.dtype
        )

        if self.speaker_use_feats:
            speaker_features = self.speaker_encoder(features, feature_masks)
        else:
            speaker_features = self.speaker_encoder(gt_mels, mel_masks)

        # vq_features is 50 hz, need to convert to true mel size
        text_features = self.text_encoder(features, feature_masks)
        text_features, vq_loss = self.vq_encoder(text_features, feature_masks)
        text_features = F.interpolate(
            text_features, size=gt_mels.shape[2], mode="nearest"
        )

        text_features = text_features + speaker_features

        # Begin sampling
        sampled_mels = torch.randn_like(gt_mels)
        self.noise_scheduler_infer.set_timesteps(100)

        for t in tqdm(self.noise_scheduler_infer.timesteps):
            timesteps = torch.tensor([t], device=sampled_mels.device, dtype=torch.long)

            # 1. predict noise model_output
            model_output = self.denoiser(
                sampled_mels, timesteps, mel_masks, text_features
            )

            # 2. compute previous image: x_t -> x_t-1
            sampled_mels = self.noise_scheduler_infer.step(
                model_output, t, sampled_mels
            ).prev_sample

        sampled_mels = self.denormalize_mels(sampled_mels)
        sampled_mels = sampled_mels * mel_masks

        with torch.autocast(device_type=sampled_mels.device.type, enabled=False):
            # Run vocoder on fp32
            fake_audios = self.vocoder.decode(sampled_mels.float())

        mel_loss = F.l1_loss(gt_mels * mel_masks, sampled_mels * mel_masks)
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
            audio,
            gen_audio,
            audio_len,
        ) in enumerate(
            zip(
                gt_mels,
                sampled_mels,
                audios,
                fake_audios,
                audio_lengths,
            )
        ):
            mel_len = audio_len // self.hop_length

            image_mels = plot_mel(
                [
                    gen_mel[:, :mel_len],
                    mel[:, :mel_len],
                ],
                [
                    "Generated Spectrogram",
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
