# The implementation is adopted from CLIP,
# made publicly available under the MIT License at https://github.com/openai/CLIP

import math
import os
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .vim import ViM


class LayerNorm(nn.LayerNorm):

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)),
                         ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.vim_att = ViM()
        self.vim_mlp = ViM()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, task_name: str):
        x_normed_1 = self.ln_1(x)
        x = x + self.attention(x_normed_1)
        x = x + self.vim_att(x_normed_1, task_name)

        x_normed_2 = self.ln_2(x)
        x = x + self.mlp(x_normed_2)
        x = x + self.vim_mlp(x_normed_2, task_name)

        return x


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, **kwargs):
        L, B, D = x.size()
        features = []
        for i, blk in enumerate(self.resblocks):
            x = blk(x, **kwargs)
            features.append(x)
        return features


class VisionTransformer(nn.Module):
    """
    The Vision Transformer (ViT) model
    Args:
        - input_resolution (int): shape of input image
        - patch_width (int): size of patch tokens
        - width (int): feature channels
        - layers (int): number of transformer layers
        - heads (int): number of multi-head attention
        - output_dim (int): output feature channels
    """

    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int = 512):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.patch_per_side = input_resolution // patch_size
        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        P = x.size(2)

        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [*, grid ** 2, width]

        cls_token = self.class_embedding.to(x.dtype).reshape(1, 1, -1).repeat(
            B, 1, 1)
        x = torch.cat([cls_token, x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_per_layer = self.transformer(x, **kwargs)

        x = x_per_layer[-1]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj

        # outputs: [x_1, ..., x_N, last_cls_token], x_i in 2D
        outputs = []
        for output in x_per_layer:
            outputs.append(output[1:, :, :].permute(1, 2,
                                                    0).reshape(B, -1, P, P))
        outputs.append(x)
        return outputs


model_dict = {
    'vit_b16_224':
    dict(input_resolution=224, patch_size=16, width=768, layers=12, heads=12),
    'vit_b32_224':
    dict(input_resolution=224, patch_size=32, width=768, layers=12, heads=12),
}


def build_backbone(arch='vit_b16_224', pretrained=None):
    """ build a ViT + ViM model
        Args:
            arch: name of backbone
            pretrained: weights of pretrained model
    """
    model_args = model_dict[arch]
    model = VisionTransformer(**model_args)
    model.load_state_dict(pretrained)

    return model


if __name__ == '__main__':
    model = build_backbone()
