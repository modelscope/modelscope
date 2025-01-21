# The implementation is adopted from OpenAI-CLIP,
# made publicly available under the MIT License at https://github.com/openai/CLIP

import math
import sys
from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from .utils import convert_weights, load_pretrained


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed
        # after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict([('-1', nn.AvgPool2d(stride)),
                             ('0',
                              nn.Conv2d(
                                  inplanes,
                                  planes * self.expansion,
                                  1,
                                  stride=1,
                                  bias=False)),
                             ('1', nn.BatchNorm2d(planes * self.expansion))]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):

    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1],
                      x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)

        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

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

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype,
            device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, idx):
        features = {}
        x_norm = self.ln_1(x)
        features['layer_{}_pre_attn'.format(idx)] = x_norm.permute(1, 0, 2)
        attn = self.attention(x_norm)
        features['layer_{}_attn'.format(idx)] = attn.permute(1, 0, 2)
        x = x + attn
        mlp = self.mlp(self.ln_2(x))
        features['layer_{}_mlp'.format(idx)] = mlp.permute(1, 0, 2)
        x = x + mlp
        return x, features


class Transformer(nn.Module):

    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        for i in range(layers):
            block = ResidualAttentionBlock(width, heads, attn_mask)
            self.resblocks.append(block)

    def forward(self, x: torch.Tensor):
        features = {}
        for idx, block in enumerate(self.resblocks):
            x, block_feats = block(x, idx)
            features.update(block_feats)
        return x, features


class VisualTransformer(nn.Module):

    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        print(input_resolution, patch_size, width, layers, heads, output_dim)
        self.input_resolution = input_resolution
        self.output_dim = output_dim
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

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_all=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1],
                      -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        zeros = torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        # shape = [*, grid ** 2 + 1, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + zeros, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, features = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if return_all:
            features['pre_logits'] = x
            return features

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPNet(nn.Module):

    def __init__(self, arch_name, pretrained, **kwargs):
        super(CLIPNet, self).__init__()

        if arch_name == 'CLIP_ViTB32':
            self.clip = VisualTransformer(
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                output_dim=512)

        elif arch_name in ('CLIP_ViTB16', 'CLIP_ViTB16_FP16'):
            self.clip = VisualTransformer(
                input_resolution=224,
                patch_size=16,
                width=768,
                layers=12,
                heads=12,
                output_dim=512)

        elif arch_name in ('CLIP_ViTL14', 'CLIP_ViTL14_FP16'):
            self.clip = VisualTransformer(
                input_resolution=224,
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                output_dim=768)

        else:
            raise KeyError(f'Unsupported arch_name for CLIP, {arch_name}')

    def forward(self, input_data):
        output = self.clip(input_data)
        return output


def CLIP(arch_name='CLIP_RN50',
         use_pretrain=False,
         load_from='',
         state_dict=None,
         **kwargs):
    model = CLIPNet(arch_name=arch_name, pretrained=None, **kwargs)
    if use_pretrain:
        if arch_name.endswith('FP16'):
            convert_weights(model.clip)
        load_pretrained(model.clip, state_dict, load_from)
    return model


class ProbingModel(torch.nn.Module):

    def __init__(self, feat_size, num_classes):
        super(ProbingModel, self).__init__()
        self.linear = torch.nn.Linear(feat_size, num_classes)

    def forward(self, x):
        return self.linear(x)
