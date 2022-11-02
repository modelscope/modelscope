from collections import OrderedDict

import torch
import torch.nn.functional as F
from fairseq.modules import LayerNorm
from torch import nn

from .utils.utils import DropPath

__all__ = [
    'vit_base',
    'vit_large',
    'vit_large_336',
    'vit_huge',
]


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path_rate=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([
                ('c_fc', nn.Linear(d_model, d_model * 4)),
                ('gelu', QuickGELU()),
                ('c_proj', nn.Linear(d_model * 4, d_model)),
            ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path_rate)

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None else None)
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask, drop_path_rate)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.width = width
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(
            width, layers, heads, drop_path_rate=drop_path_rate)

    def forward(self, x: torch.Tensor):
        resolution = x.shape[-2]
        height, width = x.shape[-2] // self.patch_size, x.shape[
            -1] // self.patch_size
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        if resolution != self.input_resolution:
            old_pe = self.positional_embedding[1:]
            patch_num = self.input_resolution // self.patch_size
            old_pe = old_pe.reshape(1, patch_num, patch_num,
                                    -1).permute(0, 3, 1, 2)
            new_pe = F.interpolate(
                old_pe, size=(height, width), mode='bilinear')
            new_pe = new_pe.permute(0, 2, 3, 1).reshape(height * width, -1)
            x = x + new_pe.to(x.dtype)
        else:
            x = x + self.positional_embedding[1:].to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        bz, seq, hidden = x.shape
        x = x.transpose(1, 2).reshape(bz, hidden, height, width)

        return x


def vit_base(drop_path_rate: float = 0.0):
    return VisionTransformer(224, 16, 768, 9, 12, drop_path_rate)


def vit_large(drop_path_rate: float = 0.0):
    return VisionTransformer(224, 14, 1024, 18, 16, drop_path_rate)


def vit_large_336(drop_path_rate: float = 0.0):
    return VisionTransformer(336, 14, 1024, 18, 16, drop_path_rate)


def vit_huge(drop_path_rate: float = 0.0):
    return VisionTransformer(224, 14, 1280, 24, 16, drop_path_rate)
