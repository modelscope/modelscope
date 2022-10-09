# Part of the implementation is borrowed and modified from latent-diffusion,
# publicly avaialbe at https://github.com/CompVis/latent-diffusion.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SuperResUNet1024']


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = torch.outer(
        timesteps, torch.pow(10000,
                             -torch.arange(half).to(timesteps).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    if dim % 2 != 0:
        x = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
    return x


class Resample(nn.Module):

    def __init__(self, in_dim, out_dim, scale_factor, use_conv=False):
        assert scale_factor in [0.5, 1.0, 2.0]
        super(Resample, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale_factor = scale_factor
        self.use_conv = use_conv

        # layers
        if scale_factor == 2.0:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_dim, out_dim, 3, padding=1)
                if use_conv else nn.Identity())
        elif scale_factor == 0.5:
            self.resample = nn.Conv2d(
                in_dim, out_dim, 3, stride=2,
                padding=1) if use_conv else nn.AvgPool2d(
                    kernel_size=2, stride=2)
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return self.resample(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 use_scale_shift_norm=True,
                 scale_factor=1.0,
                 dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        self.scale_factor = scale_factor

        # layers
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.resample = Resample(in_dim, in_dim, scale_factor, use_conv=False)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim,
                      out_dim * 2 if use_scale_shift_norm else out_dim))
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, e):
        identity = self.resample(x)
        x = self.layer1[-1](self.resample(self.layer1[:-1](x)))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1).type(x.dtype)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class SuperResUNet1024(nn.Module):

    def __init__(self,
                 in_dim=6,
                 dim=192,
                 out_dim=3,
                 dim_mult=[1, 1, 2, 2, 4, 4],
                 num_res_blocks=2,
                 resblock_resample=True,
                 use_scale_shift_norm=True,
                 dropout=0.0):
        embed_dim = dim * 4
        super(SuperResUNet1024, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.resblock_resample = resblock_resample
        self.use_scale_shift_norm = use_scale_shift_norm

        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))

        # encoder
        self.encoder = nn.ModuleList(
            [nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual block
                block = nn.ModuleList([
                    ResidualBlock(in_dim, embed_dim, out_dim,
                                  use_scale_shift_norm, 1.0, dropout)
                ])
                shortcut_dims.append(out_dim)
                in_dim = out_dim
                self.encoder.append(block)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    if resblock_resample:
                        downsample = ResidualBlock(out_dim, embed_dim, out_dim,
                                                   use_scale_shift_norm, 0.5,
                                                   dropout)
                    else:
                        downsample = Resample(
                            out_dim, out_dim, 0.5, use_conv=True)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.encoder.append(downsample)

        # middle
        self.middle = nn.ModuleList([
            ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm,
                          1.0, dropout),
            ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm,
                          1.0, dropout)
        ])

        # decoder
        self.decoder = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual block
                block = nn.ModuleList([
                    ResidualBlock(in_dim + shortcut_dims.pop(), embed_dim,
                                  out_dim, use_scale_shift_norm, 1.0, dropout)
                ])
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    if resblock_resample:
                        upsample = ResidualBlock(out_dim, embed_dim, out_dim,
                                                 use_scale_shift_norm, 2.0,
                                                 dropout)
                    else:
                        upsample = Resample(
                            out_dim, out_dim, 2.0, use_conv=True)
                    scale *= 2.0
                    block.append(upsample)
                self.decoder.append(block)

        # head
        self.head = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, x, t, concat):
        # embedding
        if concat is not None:
            if concat.shape[-2:] != x.shape[-2:]:
                concat = F.interpolate(
                    concat, x.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, concat], dim=1)
        e = self.time_embedding(sinusoidal_embedding(t, self.dim))

        # encoder
        xs = []
        for block in self.encoder:
            x = self._forward_single(block, x, e)
            xs.append(x)

        # middle
        for block in self.middle:
            x = self._forward_single(block, x, e)

        # decoder
        for block in self.decoder:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, e)

        # head
        x = self.head(x)
        return x

    def _forward_single(self, module, x, e):
        if isinstance(module, ResidualBlock):
            x = module(x, e)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, e)
        else:
            x = module(x)
        return x
