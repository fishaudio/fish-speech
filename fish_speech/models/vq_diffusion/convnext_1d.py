from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        mlp_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
        layer_scale_init_value: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            groups=dim,
            dilation=dilation,
            padding=int(dilation * (7 - 1) / 2),
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value is not None and layer_scale_init_value > 0
            else None
        )

        self.condition_projection = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x

        if condition is not None:
            x = x + self.condition_projection(condition)

        if x_mask is not None:
            x = x * x_mask

        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x

        return x


@dataclass
class ConvNext1DOutput(BaseOutput):
    """
    The output of [`UNet1DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.FloatTensor


class ConvNext1DModel(ModelMixin, ConfigMixin):
    r"""
    A ConvNext model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        in_channels (`int`, *optional*, defaults to 128):
            Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 128):
            Number of channels in the output.
        intermediate_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the intermediate blocks.
        mlp_dim (`int`, *optional*, defaults to 2048):
            Dimensionality of the MLP.
        num_layers (`int`, *optional*, defaults to 20):
            Number of layers in the model.
        dilation_cycle_length (`int`, *optional*, defaults to 4):
            Length of the dilation cycle.
        time_embedding_type (`str`, *optional*, defaults to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, defaults to `None`):
            Optional activation function to use only once on the time embeddings before they are passed to the rest of
            the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        intermediate_dim: int = 512,
        mlp_dim: int = 2048,
        num_layers: int = 20,
        dilation_cycle_length: int = 4,
        time_embedding_type: str = "positional",
    ):
        super().__init__()

        if intermediate_dim % 2 != 0:
            raise ValueError("intermediate_dim must be divisible by 2.")

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                intermediate_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=False,
            )
            timestep_input_dim = intermediate_dim
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(in_channels, False, 0)
            timestep_input_dim = in_channels
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_mlp = TimestepEmbedding(
            timestep_input_dim,
            intermediate_dim,
            act_fn="silu",
            cond_proj_dim=None,  # No conditional projection for now
        )

        # Project to intermediate dim
        self.in_proj = nn.Conv1d(in_channels, intermediate_dim, 1)
        self.out_proj = nn.Conv1d(intermediate_dim, out_channels, 1)

        # Blocks
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=intermediate_dim,
                    intermediate_dim=mlp_dim,
                    dilation=2 ** (i % dilation_cycle_length),
                )
                for i in range(num_layers)
            ]
        )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value: bool = False):
        self.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        sample_mask: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> Union[ConvNext1DOutput, Tuple]:
        r"""
        The [`ConvNext1DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.FloatTensor` or `float` or `int`):
                The number of timesteps to denoise an input.
            sample_mask (`torch.BoolTensor`, *optional*):
                A mask of the same shape as `sample` that indicates which elements are invalid.
                True means the element is invalid and should be masked out.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.ConvNext1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_1d.ConvNext1DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_1d.ConvNext1DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        t_emb = self.time_proj(timestep)
        t_emb = self.time_mlp(t_emb)[..., None]

        # 2. pre-process
        if condition is not None:
            sample = torch.cat([sample, condition], dim=1)

        x = self.in_proj(sample)

        if sample_mask.ndim == 2:
            sample_mask = sample_mask[:, None, :]

        # 3. blocks
        for block in self.blocks:
            if self.training and self.is_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, sample_mask)
            else:
                x = block(x, t_emb, sample_mask)

        # 4. post-process
        return self.out_proj(x)
