# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UNet']


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

    def __init__(self, scale_factor=1.0):
        assert scale_factor in [0.5, 1.0, 2.0]
        super(Resample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor == 2.0:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.scale_factor == 0.5:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, embed_dim, out_dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        # layers
        self.layer1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.SiLU(),
            nn.Conv2d(in_dim, out_dim, 3, padding=1))
        self.embedding = nn.Sequential(nn.SiLU(),
                                       nn.Linear(embed_dim, out_dim))
        self.layer2 = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 3, padding=1))
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Conv2d(
            in_dim, out_dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.layer2[-1].weight)

    def forward(self, x, y):
        identity = x
        x = self.layer1(x)
        x = x + self.embedding(y).unsqueeze(-1).unsqueeze(-1)
        x = self.layer2(x)
        x = x + self.shortcut(identity)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, context_dim=None, num_heads=8, dropout=0.0):
        assert dim % num_heads == 0
        assert context_dim is None or context_dim % num_heads == 0
        context_dim = context_dim or dim
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.pow(self.head_dim, -0.25)

        # layers
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        # check inputs
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        # compute attention
        attn = torch.einsum('binc,bjnc->bnij', q * self.scale, k * self.scale)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # gather context
        x = torch.einsum('bnij,bjnc->binc', attn, v)
        x = x.reshape(b, -1, n * c)

        # output
        x = self.o(x)
        x = self.dropout(x)
        return x


class GLU(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class TransformerBlock(nn.Module):

    def __init__(self, dim, context_dim, num_heads, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # input
        self.norm1 = nn.GroupNorm(32, dim, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(dim, dim, 1)

        # self attention
        self.norm2 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, None, num_heads, dropout)

        # cross attention
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, context_dim, num_heads,
                                             dropout)

        # ffn
        self.norm4 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            GLU(dim, dim * 4), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

        # output
        self.conv2 = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x, context):
        b, c, h, w = x.size()
        identity = x

        # input
        x = self.norm1(x)
        x = self.conv1(x).view(b, c, -1).transpose(1, 2)

        # attention
        x = x + self.self_attn(self.norm2(x))
        x = x + self.cross_attn(self.norm3(x), context)
        x = x + self.ffn(self.norm4(x))

        # output
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.conv2(x)
        return x + identity


class UNet(nn.Module):

    def __init__(self,
                 resolution=64,
                 in_dim=3,
                 dim=192,
                 context_dim=512,
                 out_dim=3,
                 dim_mult=[1, 2, 3, 5],
                 num_heads=1,
                 head_dim=None,
                 num_res_blocks=2,
                 attn_scales=[1 / 2, 1 / 4, 1 / 8],
                 num_classes=1001,
                 dropout=0.0):
        embed_dim = dim * 4
        super(UNet, self).__init__()
        self.resolution = resolution
        self.in_dim = in_dim
        self.dim = dim
        self.context_dim = context_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.num_classes = num_classes

        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        self.label_embedding = nn.Embedding(num_classes, context_dim)

        # encoder
        self.encoder = nn.ModuleList(
            [nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        shortcut_dims.append(dim)
        for i, (in_dim,
                out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                # residual (+attention) blocks
                block = nn.ModuleList(
                    [ResidualBlock(in_dim, embed_dim, out_dim, dropout)])
                if scale in attn_scales:
                    block.append(
                        TransformerBlock(out_dim, context_dim, num_heads))
                in_dim = out_dim
                self.encoder.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    self.encoder.append(
                        nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))
                    shortcut_dims.append(out_dim)
                    scale /= 2.0

        # middle
        self.middle = nn.ModuleList([
            ResidualBlock(out_dim, embed_dim, out_dim, dropout),
            TransformerBlock(out_dim, context_dim, num_heads),
            ResidualBlock(out_dim, embed_dim, out_dim, dropout)
        ])

        # decoder
        self.decoder = nn.ModuleList()
        for i, (in_dim,
                out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                # residual (+attention) blocks
                block = nn.ModuleList([
                    ResidualBlock(in_dim + shortcut_dims.pop(), embed_dim,
                                  out_dim, dropout)
                ])
                if scale in attn_scales:
                    block.append(
                        TransformerBlock(out_dim, context_dim, num_heads,
                                         dropout))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    block.append(
                        nn.Sequential(
                            Resample(scale_factor=2.0),
                            nn.Conv2d(out_dim, out_dim, 3, padding=1)))
                    scale *= 2.0
                self.decoder.append(block)

        # head
        self.head = nn.Sequential(
            nn.GroupNorm(32, out_dim), nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))

        # zero out the last layer params
        nn.init.zeros_(self.head[-1].weight)

    def forward(self, x, t, y, concat=None):
        # embeddings
        if concat is not None:
            x = torch.cat([x, concat], dim=1)
        t = self.time_embedding(sinusoidal_embedding(t, self.dim))
        y = self.label_embedding(y)

        # encoder
        xs = []
        for block in self.encoder:
            x = self._forward_single(block, x, t, y)
            xs.append(x)

        # middle
        for block in self.middle:
            x = self._forward_single(block, x, t, y)

        # decoder
        for block in self.decoder:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, t, y)

        # head
        x = self.head(x)
        return x

    def _forward_single(self, module, x, t, y):
        if isinstance(module, ResidualBlock):
            x = module(x, t)
        elif isinstance(module, TransformerBlock):
            x = module(x, y)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block, x, t, y)
        else:
            x = module(x)
        return x
