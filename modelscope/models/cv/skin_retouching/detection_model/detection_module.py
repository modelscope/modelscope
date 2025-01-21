# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn


class ConvBNActiv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bn=True,
                 sample='none-3',
                 activ='relu',
                 bias=False):
        super(ConvBNActiv, self).__init__()

        if sample == 'down-7':
            self.conv = nn.Conv2d(
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
        elif sample == 'down-3':
            self.conv = nn.Conv2d(
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

        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, images):

        outputs = self.conv(images)
        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)
        if hasattr(self, 'activation'):
            outputs = self.activation(outputs)

        return outputs
