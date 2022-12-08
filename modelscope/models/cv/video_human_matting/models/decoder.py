"""
Part of the implementation is borrowed from paper RVM
paper publicly available at <https://arxiv.org/abs/2108.11515/>
"""
from typing import Optional

import torch
from torch import Tensor, nn


class hswish(nn.Module):

    def forward(self, x):
        return torch.nn.Hardswish(inplace=True)(x)


class scSEblock(nn.Module):

    def __init__(self, out):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(out, int(out / 2), 3, 1, 1),
            nn.GroupNorm(out // 8, int(out / 2)), hswish())
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out / 2), out, 1, 1, 0),
            nn.GroupNorm(out // 4, out),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward_single(self, x):
        b, c, _, _ = x.size()
        x2 = self.avgpool(x).view(b, c, 1, 1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = torch.sigmoid(x2)
        out = x2 * x
        return out

    def forward_time(self, x):
        B, T, _, H, W = x.shape
        x = x.flatten(0, 1)
        out = self.forward_single(x)
        out = out.unflatten(0, (B, T))
        return out

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time(x)
        else:
            return self.forward_single(x)


class RecurrentDecoder(nn.Module):

    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3],
                                       feature_channels[2], 3,
                                       decoder_channels[0])
        self.sc3 = scSEblock(decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0],
                                       feature_channels[1], 3,
                                       decoder_channels[1])
        self.sc2 = scSEblock(decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1],
                                       feature_channels[0], 3,
                                       decoder_channels[2])
        self.sc1 = scSEblock(decoder_channels[2])
        self.out0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

        self.crosslevel1 = crossfeature(feature_channels[3],
                                        feature_channels[1])
        self.crosslevel2 = crossfeature(feature_channels[2],
                                        feature_channels[0])

    def forward(self, s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor,
                f4: Tensor, r1: Optional[Tensor], r2: Optional[Tensor],
                r3: Optional[Tensor], r4: Optional[Tensor]):
        s2, s3, s4 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s4, r3)
        x3 = self.sc3(x3)
        f2 = self.crosslevel1(f4, f2)
        x2, r2 = self.decode2(x3, f2, s3, r2)
        x2 = self.sc2(x2)
        f1 = self.crosslevel2(f3, f1)
        x1, r1 = self.decode1(x2, f1, s2, r1)
        x1 = self.sc1(x1)
        out = self.out0(x1, s0)
        return out, r1, r2, r3, r4


class AvgPool(nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(
            2, 2, count_include_pad=False, ceil_mode=True)

    def forward_single_frame(self, s0):
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3

    def forward_time_series(self, s0):
        B, T = s0.shape[:2]
        s0 = s0.flatten(0, 1)
        s1, s2, s3 = self.forward_single_frame(s0)
        s1 = s1.unflatten(0, (B, T))
        s2 = s2.unflatten(0, (B, T))
        s3 = s3.unflatten(0, (B, T))
        return s1, s2, s3

    def forward(self, s0):
        if s0.ndim == 5:
            return self.forward_time_series(s0)
        else:
            return self.forward_single_frame(s0)


class crossfeature(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward_single_frame(self, x1, x2):
        b, c, _, _ = x1.size()
        x1 = self.avg(x1).view(b, c, 1, 1)
        x1 = self.conv(x1)
        x1 = torch.sigmoid(x1)
        x2 = x1 * x2
        return x2

    def forward_time_series(self, x1, x2):
        b, t = x1.shape[:2]
        x1 = x1.flatten(0, 1)
        x2 = x2.flatten(0, 1)
        x2 = self.forward_single_frame(x1, x2)
        return x2.unflatten(0, (b, t))

    def forward(self, x1, x2):
        if x1.ndim == 5:
            return self.forward_time_series(x1, x2)
        else:
            return self.forward_single_frame(x1, x2)


class BottleneckBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = GRU(channels // 2)

    def forward(self, x, r):
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(skip_channels, in_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(in_channels // 4, in_channels), hswish())
        self.att_skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=False),
            nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + in_channels + src_channels,
                out_channels,
                3,
                1,
                1,
                bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            hswish(),
        )
        self.gru = GRU(out_channels // 2)

    def forward_single_frame(self, x, f, s, r: Optional[Tensor]):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        att = self.att_skip(x)
        f = self.shortcut(f)
        f = att * f
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r

    def forward_time_series(self, x, f, s, r: Optional[Tensor]):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        f = f.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        f = self.shortcut(f)
        att = self.att_skip(x)
        f = att * f
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        a, b = x.split(self.out_channels // 2, dim=2)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=2)
        return x, r

    def forward(self, x, f, s, r: Optional[Tensor]):
        if x.ndim == 5:
            return self.forward_time_series(x, f, s, r)
        else:
            return self.forward_single_frame(x, f, s, r)


class OutputBlock(nn.Module):

    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(out_channels // 2, out_channels),
            hswish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(out_channels // 2, out_channels),
            hswish(),
        )

    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x

    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        x = x.unflatten(0, (B, T))
        return x

    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class Projection(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return self.conv(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)


class GRU(nn.Module):

    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Conv2d(
            channels * 2, channels * 2, kernel_size, padding=padding)
        self.act_ih = nn.Sigmoid()
        self.hh = nn.Conv2d(
            channels * 2, channels, kernel_size, padding=padding)
        self.act_hh = nn.Tanh()

    def forward_single_frame(self, x, pre_fea):
        fea_ih = self.ih(torch.cat([x, pre_fea], dim=1))
        r, z = self.act_ih(fea_ih).split(self.channels, dim=1)
        fea_hh = self.hh(torch.cat([x, r * pre_fea], dim=1))
        c = self.act_hh(fea_hh)
        fea_gru = (1 - z) * pre_fea + z * c
        return fea_gru, fea_gru

    def forward_time_series(self, x, pre_fea):
        o = []
        for xt in x.unbind(dim=1):
            ot, pre_fea = self.forward_single_frame(xt, pre_fea)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, pre_fea

    def forward(self, x, pre_fea):
        if pre_fea is None:
            pre_fea = torch.zeros(
                (x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                device=x.device,
                dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, pre_fea)
        else:
            return self.forward_single_frame(x, pre_fea)
