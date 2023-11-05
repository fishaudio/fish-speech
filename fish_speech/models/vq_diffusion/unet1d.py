# Refer to https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/model/diffusion.py

import math

import torch
from einops import rearrange
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, init_values=1e-5):
        super().__init__()

        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        self.gamma = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out) * self.gamma.view(1, -1, 1, 1) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Unet1DDenoiser(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        pe_scale=1000,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.pe_scale = pe_scale

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim)
        )
        self.downsample_rate = 2 ** (len(dim_mults) - 1)

        dims = [2, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        LinearAttention(dim_out),
                        nn.Conv2d(dim_out, dim_out, 3, 2, 1)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = LinearAttention(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        LinearAttention(dim_in),
                        nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, t, mask, condition):
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        x = torch.stack([condition, x], 1)
        mask = mask.unsqueeze(1)

        original_len = x.shape[3]
        if x.shape[3] % self.downsample_rate != 0:
            x = nn.functional.pad(
                x, (0, self.downsample_rate - x.shape[3] % self.downsample_rate)
            )
            mask = nn.functional.pad(
                mask, (0, self.downsample_rate - mask.shape[3] % self.downsample_rate)
            )

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        output = (output * mask).squeeze(1)
        return output[:, :, :original_len]


if __name__ == "__main__":
    model = Unet1DDenoiser(128)
    mel = torch.randn(1, 128, 99)
    mask = torch.ones(1, 1, 99)

    print(model(mel, mask, torch.tensor([10], dtype=torch.long), mel).shape)
