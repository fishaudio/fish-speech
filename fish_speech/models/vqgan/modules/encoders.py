from typing import Optional

import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

from fish_speech.models.vqgan.modules.modules import WN
from fish_speech.models.vqgan.modules.transformer import RelativePositionTransformer
from fish_speech.models.vqgan.utils import sequence_mask


# * Ready and Tested
class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
        gin_channels=0,
        speaker_cond_layer=0,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            n_heads (int): Number of attention heads for the Transformer layers.
            n_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout (float): Dropout rate for the Transformer layers.
            gin_channels (int, optional): Number of channels for speaker embedding. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # self.emb = nn.Linear(n_vocab, hidden_channels)
        self.emb = nn.Linear(n_vocab, hidden_channels, 1)
        # nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            window_size=4,
            gin_channels=gin_channels,
            speaker_cond_layer=speaker_cond_layer,
        )
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor = None,
        noise_scale: float = 1,
    ):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        # x = self.emb(x).mT * math.sqrt(self.hidden_channels)  # [b, h, t]
        x = self.emb(x).mT  # * math.sqrt(self.hidden_channels)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x, x_mask, g=g)
        stats = self.proj(x.mT).mT * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask * noise_scale
        return z, m, logs, x, x_mask


# * Ready and Tested
class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.out_channels = out_channels

        self.pre = nn.Linear(in_channels, hidden_channels)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Linear(hidden_channels, out_channels * 2)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        g: torch.Tensor,
        noise_scale: float = 1,
    ):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x.mT).mT * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x.mT).mT * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + torch.randn_like(m) * torch.exp(logs) * x_mask * noise_scale
        return z, m, logs, x_mask


# TODO: Ready for testing
class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 192,
        out_channels: int = 512,
        num_heads: int = 2,
        num_layers: int = 4,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 5, padding=2),
            nn.SiLU(),
            nn.Dropout(p_dropout),
        )
        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels,
            n_heads=num_heads,
            n_layers=num_layers,
            kernel_size=5,
            dropout=p_dropout,
            window_size=4,
        )
        self.out_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, mels, mel_lengths: torch.Tensor):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
        """

        x_mask = torch.unsqueeze(sequence_mask(mel_lengths, mels.size(2)), 1).to(
            mels.dtype
        )
        x = self.in_proj(mels) * x_mask
        x = self.encoder(x, x_mask)

        # Avg Pooling
        x = x * x_mask
        x = torch.sum(x, dim=2) / torch.sum(x_mask, dim=2)
        x = self.out_proj(x)[..., None]

        return x


class VQEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1024,
        vq_channels: int = 1024,
        codebook_size: int = 2048,
        downsample: int = 2,
        kmeans_ckpt: Optional[str] = None,
    ):
        super().__init__()

        self.vq = VectorQuantize(
            dim=vq_channels,
            codebook_size=codebook_size,
            threshold_ema_dead_code=2,
            kmeans_init=False,
            channel_last=False,
        )

        self.conv_in = nn.Conv1d(
            in_channels, vq_channels, kernel_size=downsample, stride=downsample
        )
        self.conv_out = nn.Sequential(
            nn.Upsample(scale_factor=downsample, mode="nearest"),
            nn.Conv1d(vq_channels, in_channels, kernel_size=1, stride=1),
        )

        if kmeans_ckpt is not None:
            self.init_weights(kmeans_ckpt)

    def init_weights(self, kmeans_ckpt):
        torch.nn.init.normal_(
            self.conv_in.weight,
            mean=1 / (self.conv_in.weight.shape[0] * self.conv_in.weight.shape[-1]),
            std=1e-2,
        )
        self.conv_in.bias.data.zero_()

        kmeans_ckpt = "results/hubert-vq-pretrain/kmeans.pt"
        kmeans_ckpt = torch.load(kmeans_ckpt, map_location="cpu")

        centroids = kmeans_ckpt["centroids"]
        bins = kmeans_ckpt["bins"]
        state_dict = {
            "_codebook.initted": torch.Tensor([True]),
            "_codebook.cluster_size": bins,
            "_codebook.embed": centroids,
            "_codebook.embed_avg": centroids.clone(),
        }

        self.vq.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        # x: [B, T, C]
        x = self.conv_in(x.mT)
        q, _, loss = self.vq(x)
        x = self.conv_out(q).mT

        return x, loss
