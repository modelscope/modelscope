# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn


class GatedConvBNActiv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        images = self.conv(x)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'bn'):
            images = self.bn(images)
        if hasattr(self, 'activation'):
            images = self.activation(images)

        images = images * gates

        return images


class GatedConvBNActiv2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(GatedConvBNActiv2, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=bias)
        elif sample == 'down-5':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=bias)
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)
            self.gate = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias)

        self.conv_skip = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, f_up, f_skip, mask):
        x = torch.cat((f_up, f_skip, mask), dim=1)
        images = self.conv(x)
        images_skip = self.conv_skip(f_skip)
        gates = self.sigmoid(self.gate(x))

        if hasattr(self, 'bn'):
            images = self.bn(images)
            images_skip = self.bn(images_skip)
        if hasattr(self, 'activation'):
            images = self.activation(images)
            images_skip = self.activation(images_skip)

        images = images * gates + images_skip * (1 - gates)

        return images
