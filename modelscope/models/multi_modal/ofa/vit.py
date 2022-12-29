# Copyright (c) 2021 OpenAI
#
# This source code is licensed under the MIT license which can be found at
# https://github.com/openai/CLIP/blob/main/LICENSE
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
    r"""
    An activation function module.
    """

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    r"""
    A residual attention block module.

    step 1. Calculate the self attention in input with layer normalization.
    step 2. Add input to the result of self attention's result as I.
    step 3. Calculate the mlp of input I with layer normalization.
    step 4. Add I to the result of mlp.
    """

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path_rate=0.0):
        r"""
        Args:
            d_model (`int`): The embedding dimensions.
            n_head (`int`): The number of heads in self attention block.
            attn_mask (`Tensor`, **optional**, default to None):
                Attention mask using in self attention.
            drop_path_rate (`float`, **optional**, default to 0.0):
                Drop path rate. See more details about drop path from
                https://arxiv.org/pdf/1605.07648v4.pdf.
        """
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
        r"""
        A wrapper of self attention .
        """
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
    r"""
    A transformer module using in `VisionTransformer`.

    Execute a sequential of `ResidualAttentionBlock`.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        drop_path_rate: float = 0.0,
    ):
        r"""
        Args:
            width (`int`): The width of input image.
            layers (`int`): The number of `ResidualAttentionBlock` layers.
            heads (int): The number of self attention heads.
            attn_mask (`Tensor`, **optional**, default to None):
                Attention mask using in self attention.
            drop_path_rate (`float`, **optional**, default to 0.0):
                Drop path rate. See more details about drop path from
                https://arxiv.org/pdf/1605.07648v4.pdf.
        """
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
    r"""
    Vision transformer module.

    step 1. Using conv2d to get the image embedding.
    step 2. If the resolution of input image doesn't equal to the initialized one
        do `bilinear` interpolate to get new patch position embedding.
    step 3. Add position embedding to image embedding to generate final image representation.
    step 4. Do `Transformer` to the image representation.
    """

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        drop_path_rate: float = 0.0,
    ):
        r"""
        Args:
            input_resolution (`int`): The resolution of input image.
            patch_size  (`int`): The resolution of each patch image.
            width (`int`): The dimension of each patch image.
            layers (`int`): The number of `ResidualAttentionBlock` in `Transformer`.
            heads (`int`): The number of heads in self attention block.
            drop_path_rate (`float`, **optional**, default to 0.0):
                Drop path rate. See more details about drop path from
                https://arxiv.org/pdf/1605.07648v4.pdf.
        """
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
    r"""
    An instance of base vision transformer model.

    Args:
        drop_path_rate (`float`, **optional**, default to 0.0):
            Drop path rate. See more details about drop path from
            https://arxiv.org/pdf/1605.07648v4.pdf.
    """
    return VisionTransformer(224, 16, 768, 9, 12, drop_path_rate)


def vit_large(drop_path_rate: float = 0.0):
    r"""
    An instance of large vision transformer model.

    Args:
        drop_path_rate (`float`, **optional**, default to 0.0):
            Drop path rate. See more details about drop path from
            https://arxiv.org/pdf/1605.07648v4.pdf.
    """
    return VisionTransformer(224, 14, 1024, 18, 16, drop_path_rate)


def vit_large_336(drop_path_rate: float = 0.0):
    r"""
    An instance of large vision transformer model with 336 as input image width .

    Args:
        drop_path_rate (`float`, **optional**, default to 0.0):
            Drop path rate. See more details about drop path from
            https://arxiv.org/pdf/1605.07648v4.pdf.
    """
    return VisionTransformer(336, 14, 1024, 18, 16, drop_path_rate)


def vit_huge(drop_path_rate: float = 0.0):
    r"""
    An instance of huge vision transformer model.

    Args:
        drop_path_rate (`float`, **optional**, default to 0.0):
            Drop path rate. See more details about drop path from
            https://arxiv.org/pdf/1605.07648v4.pdf.
    """
    return VisionTransformer(224, 14, 1280, 24, 16, drop_path_rate)
