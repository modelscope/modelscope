# Part of the implementation is borrowed and modified from Next-ViT,
# publicly available at https://github.com/bytedance/Next-ViT
import collections.abc
import itertools
import math
import os
import warnings
from functools import partial
from typing import Dict, Sequence

import torch
import torch.nn as nn
from einops import rearrange
from mmcls.models.backbones.base_backbone import BaseBackbone
from mmcls.models.builder import BACKBONES
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..utils import trunc_normal_

NORM_EPS = 1e-5


class ConvBNReLU(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2),
                                        stride=2,
                                        ceil_mode=True,
                                        count_include_pad=False)
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 out_features=None,
                 mlp_ratio=None,
                 drop=0.,
                 bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(
            in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 path_dropout=0,
                 drop=0,
                 head_dim=32,
                 mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(self,
                 dim,
                 out_dim=None,
                 head_dim=32,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(
                kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merge = False

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads,
                      int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merge:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads,
                          int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NTB(nn.Module):
    """
    Next Transformer Block
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        path_dropout,
        stride=1,
        sr_ratio=1,
        mlp_ratio=2,
        head_dim=32,
        mix_block_ratio=0.75,
        attn_drop=0,
        drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels,
                                      stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, 'b c h w -> b (h w) c')  # b n c
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, 'b (h w) c -> b c h w', h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


@BACKBONES.register_module()
class NextViT(BaseBackbone):
    stem_chs = {
        'x_small': [64, 32, 64],
        'small': [64, 32, 64],
        'base': [64, 32, 64],
        'large': [64, 32, 64],
    }
    depths = {
        'x_small': [1, 1, 5, 1],
        'small': [3, 4, 10, 3],
        'base': [3, 4, 20, 3],
        'large': [3, 4, 30, 3],
    }

    def __init__(self,
                 arch='small',
                 path_dropout=0.2,
                 attn_drop=0,
                 drop=0,
                 strides=[1, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 head_dim=32,
                 mix_block_ratio=0.75,
                 resume='',
                 with_extra_norm=True,
                 norm_eval=False,
                 norm_cfg=None,
                 out_indices=-1,
                 frozen_stages=-1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        stem_chs = self.stem_chs[arch]
        depths = self.depths[arch]

        self.frozen_stages = frozen_stages
        self.with_extra_norm = with_extra_norm
        self.norm_eval = norm_eval
        self.stage1_out_channels = [96] * (depths[0])
        self.stage2_out_channels = [192] * (depths[1] - 1) + [256]
        self.stage3_out_channels = [384, 384, 384, 384, 512] * (depths[2] // 5)
        self.stage4_out_channels = [768] * (depths[3] - 1) + [1024]
        self.stage_out_channels = [
            self.stage1_out_channels, self.stage2_out_channels,
            self.stage3_out_channels, self.stage4_out_channels
        ]

        # Next Hybrid Strategy
        self.stage1_block_types = [NCB] * depths[0]
        self.stage2_block_types = [NCB] * (depths[1] - 1) + [NTB]
        self.stage3_block_types = [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5)
        self.stage4_block_types = [NCB] * (depths[3] - 1) + [NTB]
        self.stage_block_types = [
            self.stage1_block_types, self.stage2_block_types,
            self.stage3_block_types, self.stage4_block_types
        ]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))
               ]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(
                        input_channel,
                        output_channel,
                        stride=stride,
                        path_dropout=dpr[idx + block_id],
                        drop=drop,
                        head_dim=head_dim)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(
                        input_channel,
                        output_channel,
                        path_dropout=dpr[idx + block_id],
                        stride=stride,
                        sr_ratio=sr_ratios[stage_id],
                        head_dim=head_dim,
                        mix_block_ratio=mix_block_ratio,
                        attn_drop=attn_drop,
                        drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = sum(depths) + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.stage_out_idx = out_indices

        if norm_cfg is not None:
            self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def init_weights(self):
        super(NextViT, self).init_weights()
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return

        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d,
                              nn.BatchNorm1d)):  # nn.GroupNorm, nn.LayerNorm,
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = list()
        x = self.stem(x)
        stage_id = 0
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx == self.stage_out_idx[stage_id]:
                if self.with_extra_norm:
                    x = self.norm(x)
                outputs.append(x)
                stage_id += 1
        return tuple(outputs)

    def _freeze_stages(self):
        if self.frozen_stages > 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False
            for idx, layer in enumerate(self.features):
                if idx <= self.stage_out_idx[self.frozen_stages - 1]:
                    layer.eval()
                    for param in layer.parameters():
                        param.requires_grad = False

    def train(self, mode=True):
        super(NextViT, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
