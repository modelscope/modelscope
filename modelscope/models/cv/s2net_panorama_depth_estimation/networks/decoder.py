# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

from .util_helper import (precompute_pixelization_maps,
                          precompute_position_encoding)


class SphConv2d(nn.Module):

    def __init__(self, in_dims, out_dims, k, neigh_map, groups=1, bias=True):
        super().__init__()
        self.neighbor_map = neigh_map
        self.conv = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size=k**2,
            stride=k**2,
            groups=groups,
            bias=bias)

    def forward(self, fmap):
        fmap = fmap.squeeze(2)  # remove row-dim
        x = F.pad(fmap, (0, 1, 0, 0, 0, 0), mode='constant', value=0.0)
        vec = x[:, :, self.neighbor_map]  # index_select
        return self.conv(vec).unsqueeze(2)


# classes
class Residual(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class SphPixelization(nn.Module):
    """
        A projection which converts ERP format feature maps to spherical feature maps,
        The indices are precomputed.
    """

    def __init__(self, index):
        super(SphPixelization, self).__init__()

        self.x0, self.y0 = index[0, :].long(), index[1, :].long()
        self.x1, self.y1 = index[2, :].long(), index[3, :].long()
        self.wa, self.wb = index[4, :].float(), index[5, :].float()
        self.wc, self.wd = index[6, :].float(), index[7, :].float()

    def forward(self, x):
        # a projection from ERP feature map to spherical feature maps by bilinear interpolation
        out = self.wa * x[:, :, self.y0, self.x0] + self.wb * x[:, :, self.y1, self.x0] \
            + self.wc * x[:, :, self.y0, self.x1] + self.wd * x[:, :, self.y1, self.x1]
        out = out.unsqueeze(-2)
        return out


class SphPosEmbedding(nn.Module):
    """Spherical positional embedding"""

    def __init__(self, dim, pos_enc):
        super(SphPosEmbedding, self).__init__()
        self.pos_enc = pos_enc.float()
        self.pos_emb = nn.Conv2d(3, dim, 1)

    def forward(self, x):
        b, c, w, h = x.shape
        pos_enc = self.pos_enc.unsqueeze(0).repeat(b, 1, 1, 1)
        x += self.pos_emb(pos_enc)
        return x


class SphInterpolate(nn.Module):
    """Spherical pixels interpolation"""

    def __init__(self, scale_factor):
        super(SphInterpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.repeat_interleave(x, self.scale_factor, dim=-1)


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SphGSA(nn.Module):
    """
        Global self attention on spherical surface.
        To balance memory usage and performance，
        we use a strategy like twins-svt(https://github.com/Meituan-AutoML/Twins)
    """

    def __init__(self, dim_in, dim_out, heads=8, dim_head=64, dropout=0., k=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Conv2d(dim_in, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, inner_dim * 2, k, stride=k, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out, 1), nn.Dropout(dropout))
        self.k = k

    def forward(self, fmap):
        shape, k = fmap.shape, self.k
        [b, n, x, y] = shape
        h = self.heads
        # windows number
        w = y // (k**2)
        q = self.to_q(fmap)
        fmap = rearrange(
            fmap, 'b n x (y p1 p2) -> (b x y) n p1 p2', p1=k, p2=k)
        kv = self.to_kv(fmap)
        kv = rearrange(kv, '(b w) n x y -> b n x (w y)', w=w)
        k, v = kv.chunk(2, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h),
            (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        return self.to_out(out)


def make_skip_connection(in_dim, out_dim, pix_index_maps=None):
    """
        establish skip connections between encoders and decoders
        we call this process as SPP（spherial pixelization projection）
    """
    out_dim1, out_dim2, out_dim3, out_dim4 = out_dim, out_dim, out_dim, out_dim
    skip_con1 = nn.Sequential(
        SphPixelization(pix_index_maps[0]), nn.Conv2d(in_dim[0], out_dim1, 1),
        nn.ReLU(False))
    skip_con2 = nn.Sequential(
        SphPixelization(pix_index_maps[1]), nn.Conv2d(in_dim[1], out_dim2, 1),
        nn.ReLU(False))
    skip_con3 = nn.Sequential(
        SphPixelization(pix_index_maps[2]), nn.Conv2d(in_dim[2], out_dim3, 1),
        nn.ReLU(False))
    skip_con4 = nn.Sequential(
        SphPixelization(pix_index_maps[3]), nn.Conv2d(in_dim[3], out_dim4, 1),
        nn.ReLU(False))
    return skip_con1, skip_con2, skip_con3, skip_con4


class ResidualConvBlock(nn.Module):

    def __init__(self, in_dim, activation, bn, neighbor_map):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = SphConv2d(
            in_dim,
            in_dim,
            k=3,
            bias=not self.bn,
            groups=self.groups,
            neigh_map=neighbor_map)
        self.conv2 = SphConv2d(
            in_dim,
            in_dim,
            k=3,
            bias=not self.bn,
            groups=self.groups,
            neigh_map=neighbor_map)

        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features_in)
            self.bn2 = nn.BatchNorm2d(features_in)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)
        return self.skip_add.add(out, x)


class SphCrossAttFusionBlock(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., k0=4, k1=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q0 = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv0 = nn.Conv2d(dim, inner_dim * 2, k0, stride=k0, bias=False)
        self.to_q1 = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv1 = nn.Conv2d(dim, inner_dim * 2, k1, stride=k1, bias=False)

        self.to_out0 = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))
        self.to_out1 = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1), nn.Dropout(dropout))
        self.add = nn.quantized.FloatFunctional()
        self.k0, self.k1 = k0, k1

    def forward(self, *xs):
        x0, x1 = xs[0], xs[1]
        assert x0.shape == x1.shape
        shape, k0, k1 = x0.shape, self.k0, self.k1
        [b, n, x, y] = shape
        h = self.heads

        # for feature map x0
        w = y // (k0**2)
        q0 = self.to_q0(x0)
        x0_w = rearrange(
            x0, 'b n x (y p1 p2) -> (b x y) n p1 p2', p1=k0, p2=k0)
        kv0 = self.to_kv0(x0_w)
        kv0 = rearrange(kv0, '(b w) n x y -> b n x (w y)', w=w)
        k0, v0 = kv0.chunk(2, dim=1)
        q0, k0, v0 = map(
            lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h),
            (q0, k0, v0))

        # for feature map x1
        w = y // (k1**2)
        q1 = self.to_q1(x1)
        x1_w = rearrange(
            x1, 'b n x (y p1 p2) -> (b x y) n p1 p2', p1=k1, p2=k1)
        kv1 = self.to_kv1(x1_w)
        kv1 = rearrange(kv1, '(b w) n x y -> b n x (w y)', w=w)
        k1, v1 = kv1.chunk(2, dim=1)
        q1, k1, v1 = map(
            lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h),
            (q1, k1, v1))

        # fusion
        dots = einsum('b i d, b j d -> b i j', q0, k1) * self.scale
        attn = dots.softmax(dim=-1)
        out0 = einsum('b i j, b j d -> b i d', attn, v1)
        out0 = rearrange(out0, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        out0 = self.to_out0(out0)

        dots = einsum('b i d, b j d -> b i j', q1, k0) * self.scale
        attn = dots.softmax(dim=-1)
        out1 = einsum('b i j, b j d -> b i d', attn, v0)
        out1 = rearrange(out1, '(b h) (x y) d -> b (h d) x y', h=h, y=y)
        out1 = self.to_out1(out1)
        return 0.5 * self.add.add(x1 + out0, x0 + out1)


class SphCAFBlock(nn.Module):

    def __init__(self, dim, pos_enc, k0=4, k1=4):
        super().__init__()
        self.process0 = nn.Sequential(
            SphPosEmbedding(dim, pos_enc=pos_enc), PreNorm(dim, nn.Identity()))
        self.process1 = nn.Sequential(
            SphPosEmbedding(dim, pos_enc=pos_enc), PreNorm(dim, nn.Identity()))
        self.fusion = SphCrossAttFusionBlock(dim, k0=k0, k1=k1)
        self.feedforward = Residual(PreNorm(dim, FeedForward(dim)))

    def forward(self, *xs):
        x0, x1 = xs[0], xs[1]
        x0, x1 = self.process0(x0), self.process1(x1)
        out = self.feedforward(self.fusion(x0, x1))
        return out


class FeatureFusion(nn.Module):

    def __init__(self,
                 enable_caf,
                 dim,
                 neighbor_map,
                 activation=nn.ReLU(False),
                 bn=False,
                 k0=4,
                 k1=4,
                 pos_enc=None):
        super(FeatureFusion, self).__init__()
        self.enable_CAF = enable_caf
        self.skip_add = None if self.enable_CAF else nn.quantized.FloatFunctional(
        )
        self.resConfUnit2 = ResidualConvBlock(
            dim, activation, bn, neighbor_map=neighbor_map)
        self.resConfUnit1 = ResidualConvBlock(
            dim, activation, bn, neighbor_map=neighbor_map)
        self.fusionBlock = SphCAFBlock(
            dim, pos_enc, k0=k0,
            k1=k1) if self.enable_CAF else self.skip_add.add
        self.up_sample = SphInterpolate(scale_factor=4)

    def forward(self, *xs):
        output = xs[0]
        res = self.resConfUnit1(xs[1])
        output = self.fusionBlock(output, res)
        output = self.resConfUnit2(output)
        output = self.up_sample(output)
        return output


class FeatureFusionBottom(nn.Module):

    def __init__(self,
                 enable_caf,
                 dim,
                 neighbor_map,
                 activation=nn.ReLU(False),
                 bn=False,
                 k=1,
                 pos_enc=None):
        super(FeatureFusionBottom, self).__init__()
        self.enable_CAF = enable_caf
        self.resConfUnit = nn.Sequential(
            SphPosEmbedding(dim, pos_enc=pos_enc),
            Residual(PreNorm(dim, SphGSA(dim, dim, k=k))),
            Residual(PreNorm(dim, FeedForward(dim, mult=4, dropout=0.))),
        ) if self.enable_CAF else ResidualConvBlock(
            dim, activation, bn, neighbor_map=neighbor_map)
        self.up_sample = SphInterpolate(scale_factor=4)

    def forward(self, *xs):
        output = xs[0]
        output = self.resConfUnit(output)
        output = self.up_sample(output)
        return output


class SphDecoder(nn.Module):

    def __init__(self,
                 cfg,
                 neighbor_maps,
                 num_ch_enc,
                 use_bn=False,
                 img_size=(512, 1024)):
        super(SphDecoder, self).__init__()
        self.use_caf = cfg.MODEL.USE_CAF_FUSION
        self.pix_sample_index_maps = precompute_pixelization_maps(
            nsides=[32, 16, 8, 4],
            initial_img_size=[img_size[0] // 4, img_size[1] // 4])

        self.pos_encodings = precompute_position_encoding(
            nsides=[32, 16, 8, 4]) if self.use_caf else None
        self.ffBlock1 = FeatureFusion(
            cfg.MODEL.USE_CAF_FUSION,
            cfg.MODEL.DECODER_DIM,
            neighbor_map=neighbor_maps[32],
            bn=use_bn,
            k0=4,
            k1=4,
            pos_enc=self.pos_encodings[0] if self.use_caf else None)
        self.ffBlock2 = FeatureFusion(
            cfg.MODEL.USE_CAF_FUSION,
            cfg.MODEL.DECODER_DIM,
            neighbor_map=neighbor_maps[16],
            bn=use_bn,
            k0=2,
            k1=2,
            pos_enc=self.pos_encodings[1] if self.use_caf else None)
        self.ffBlock3 = FeatureFusion(
            cfg.MODEL.USE_CAF_FUSION,
            cfg.MODEL.DECODER_DIM,
            neighbor_map=neighbor_maps[8],
            bn=use_bn,
            k0=1,
            k1=1,
            pos_enc=self.pos_encodings[2] if self.use_caf else None)
        self.ffBlock4 = FeatureFusionBottom(
            cfg.MODEL.USE_CAF_FUSION,
            cfg.MODEL.DECODER_DIM,
            neighbor_map=neighbor_maps[4],
            bn=use_bn,
            k=1,
            pos_enc=self.pos_encodings[3] if self.use_caf else None)
        self.head = nn.Sequential(
            SphConv2d(
                cfg.MODEL.DECODER_DIM,
                cfg.MODEL.DECODER_DIM // 4,
                k=3,
                neigh_map=neighbor_maps[64]), nn.ReLU(True),
            SphInterpolate(scale_factor=4),
            SphConv2d(
                cfg.MODEL.DECODER_DIM // 4,
                cfg.MODEL.DECODER_DIM // 8,
                k=3,
                neigh_map=neighbor_maps[128]), nn.ReLU(True),
            nn.Conv2d(
                cfg.MODEL.DECODER_DIM // 8,
                1,
                kernel_size=1,
                stride=1,
                padding=0), nn.ReLU(True))
        self.skip_con1, self.skip_con2, self.skip_con3, self.skip_con4 = \
            make_skip_connection(num_ch_enc, cfg.MODEL.DECODER_DIM, pix_index_maps=self.pix_sample_index_maps)

    def forward(self, enc_feat1, enc_feat2, enc_feat3, enc_feat4):
        sph_skip_feat1 = self.skip_con1(enc_feat1)
        sph_skip_feat2 = self.skip_con2(enc_feat2)
        sph_skip_feat3 = self.skip_con3(enc_feat3)
        sph_skip_feat4 = self.skip_con4(enc_feat4)

        sph_dec_feat4 = self.ffBlock4(sph_skip_feat4)
        sph_dec_feat3 = self.ffBlock3(sph_dec_feat4, sph_skip_feat3)
        sph_dec_feat2 = self.ffBlock2(sph_dec_feat3, sph_skip_feat2)
        sph_dec_feat1 = self.ffBlock1(sph_dec_feat2, sph_skip_feat1)
        out = self.head(sph_dec_feat1)
        return out
