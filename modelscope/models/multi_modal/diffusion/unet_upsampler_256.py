# Part of the implementation is borrowed and modified from latent-diffusion,
# publicly available at https://github.com/CompVis/latent-diffusion.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SuperResUNet256']


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
        self.resample_x = Resample(in_dim, in_dim, scale_factor)
        self.resample_i = Resample(in_dim, in_dim, scale_factor)
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
        identity = self.resample_i(x)
        x = self.layer1[-1](self.resample_x(self.layer1[:-1](x)))
        e = self.embedding(e).unsqueeze(-1).unsqueeze(-1)
        if self.use_scale_shift_norm:
            scale, shift = e.chunk(2, dim=1)
            x = self.layer2[0](x) * (1 + scale) + shift
            x = self.layer2[1:](x)
        else:
            x = x + e
            x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class AttentionBlock(nn.Module):

    def __init__(self, dim, context_dim=None, num_heads=None, head_dim=None):
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)

        # layers
        self.norm = nn.GroupNorm(32, dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        if context_dim is not None:
            self.context_kv = nn.Linear(context_dim, dim * 2)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, context=None, mask=None):
        r"""x:       [B, C, H, W].
            context: [B, L, C] or None.
            mask:    [B, L] or None.
        """
        identity = x
        b, c, h, w, n, d = *x.size(), self.num_heads, self.head_dim

        # compute query, key, value
        x = self.norm(x)
        q, k, v = self.to_qkv(x).view(b, n * 3, d, h * w).chunk(3, dim=1)
        if context is not None:
            ck, cv = self.context_kv(context).reshape(b, -1, n * 2,
                                                      d).permute(0, 2, 3,
                                                                 1).chunk(
                                                                     2, dim=1)
            k = torch.cat([k, ck], dim=-1)
            v = torch.cat([v, cv], dim=-1)

        # compute attention
        attn = torch.einsum('bndi,bndj->bnij', q * self.scale, k * self.scale)
        if mask is not None:
            pad_mask = mask.new_ones((b, 1, 1, h * w))
            mask = torch.cat((pad_mask, mask.unsqueeze(1).unsqueeze(1)),
                             dim=-1)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # gather context
        x = torch.einsum('bnij,bndj->bndi', attn, v)
        x = x.reshape(b, c, h, w)

        # output
        x = self.proj(x)
        return x + identity


class SuperResUNet256(nn.Module):

    def __init__(self,
                 in_dim=6,
                 out_dim=3,
                 dim=256,
                 text_dim=1024,
                 context_dim=512,
                 dim_mult=[1, 2, 2, 3, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=2,
                 attn_scales=[1 / 16],
                 resblock_resample=True,
                 use_conv=True,
                 use_scale_shift_norm=True,
                 dropout=0.1):
        embed_dim = dim * 4
        super(SuperResUNet256, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.resblock_resample = resblock_resample
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        self.noise_time_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        self.pool_embedding = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, embed_dim))
        self.text_embedding = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, context_dim),
            nn.SiLU(), nn.Linear(context_dim, context_dim))

        # encoder
        self.encoder = nn.ModuleList(
            [nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResidualBlock(in_dim, embed_dim, out_dim,
                                  use_scale_shift_norm, 1.0, dropout)
                ])
                if scale in attn_scales:
                    block.append(
                        AttentionBlock(out_dim, context_dim, num_heads,
                                       head_dim))
                shortcut_dims.append(out_dim)
                in_dim = out_dim
                self.encoder.append(block)

            # downsample
            if i != len(dim_mult) - 1:
                if resblock_resample:
                    downsample = ResidualBlock(out_dim, embed_dim, out_dim,
                                               use_scale_shift_norm, 0.5,
                                               dropout)
                else:
                    downsample = Resample(out_dim, out_dim, 0.5, use_conv)
                shortcut_dims.append(out_dim)
                scale /= 2.0
                self.encoder.append(downsample)

        # middle
        self.middle = nn.ModuleList([
            ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm,
                          1.0, dropout),
            AttentionBlock(out_dim, context_dim, num_heads, head_dim),
            ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm,
                          1.0, dropout)
        ])

        # decoder
        self.decoder = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResidualBlock(in_dim + shortcut_dims.pop(), embed_dim,
                                  out_dim, use_scale_shift_norm, 1.0, dropout)
                ])
                if scale in attn_scales:
                    block.append(
                        AttentionBlock(out_dim, context_dim, num_heads,
                                       head_dim))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    if resblock_resample:
                        upsample = ResidualBlock(out_dim, embed_dim, out_dim,
                                                 use_scale_shift_norm, 2.0,
                                                 dropout)
                    else:
                        upsample = Resample(out_dim, out_dim, 2.0, use_conv)
                    scale *= 2.0
                    block.append(upsample)
                self.decoder.append(block)

        # head
        self.head = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, x, t, lx, lt, y, context, mask):
        assert context.shape[:-1] == mask.shape

        # embeddings
        t = self.time_embedding(sinusoidal_embedding(t, self.dim)) \
            + self.noise_time_embedding(sinusoidal_embedding(lt, self.dim)) \
            + self.pool_embedding(y)

        context = self.text_embedding(context)

        if lx.shape[-2:] != x.shape[-2:]:
            lx = F.interpolate(
                lx, x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, lx], dim=1)

        # encoder
        xs = []
        for block in self.encoder:
            x = self._forward_single(block, x, t, context, mask)
            xs.append(x)

        # middle
        for block in self.middle:
            x = self._forward_single(block, x, t, context, mask)

        # decoder
        for block in self.decoder:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, t, context, mask)

        # head
        x = self.head(x)
        return x

    def _forward_single(self, module, x, t, context, mask):
        if isinstance(module, ResidualBlock):
            x = module(x, t)
        elif isinstance(module, AttentionBlock):
            x = module(x, context, mask)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, t, context, mask)
        else:
            x = module(x)
        return x
