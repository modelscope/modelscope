# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        return self.triple_conv(x)


class DownConv(nn.Module):
    """Downscaling with avgpool then double/triple conv"""

    def __init__(self, in_channels, out_channels, num_conv=2):
        super().__init__()
        if num_conv == 2:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2), DoubleConv(in_channels, out_channels))
        else:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2), TripleConv(in_channels, out_channels))

    def forward(self, x):
        return self.pool_conv(x)


class UpCatConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(
                scale_factor=2, mode='nearest', align_corners=None)
            self.conv = DoubleConv(in_channels, out_channels)

        self.subpixel = nn.PixelShuffle(2)

    def interpolate(self, x):
        tensor_temp = x
        for i in range(3):
            tensor_temp = torch.cat((tensor_temp, x), 1)
        x = tensor_temp
        x = self.subpixel(x)
        return x

    def forward(self, x1, x2):

        x1 = self.interpolate(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
