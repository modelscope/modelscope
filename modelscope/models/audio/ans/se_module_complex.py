# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from torch import nn


class SELayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_r = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())
        self.fc_i = nn.Sequential(
            nn.Linear(channel, channel // reduction), nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        x_r = self.avg_pool(x[:, :, :, :, 0]).view(b, c)
        x_i = self.avg_pool(x[:, :, :, :, 1]).view(b, c)
        y_r = self.fc_r(x_r).view(b, c, 1, 1, 1) - self.fc_i(x_i).view(
            b, c, 1, 1, 1)
        y_i = self.fc_r(x_i).view(b, c, 1, 1, 1) + self.fc_i(x_r).view(
            b, c, 1, 1, 1)
        y = torch.cat([y_r, y_i], 4)
        return x * y
