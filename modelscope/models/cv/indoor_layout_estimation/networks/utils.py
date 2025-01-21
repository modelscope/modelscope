# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StripPooling(nn.Module):

    def __init__(self, in_channels, input_size, pool_size, norm_layer):
        super(StripPooling, self).__init__()
        input_size = np.array(input_size)
        output_size = np.array([pool_size[0], pool_size[0]])
        stride = np.floor((input_size / output_size)).astype(int)
        kernel_size = (input_size - (output_size - 1) * stride).astype(int)
        self.pool1 = nn.AvgPool2d(
            kernel_size=(kernel_size[0], kernel_size[1]),
            stride=(stride[0], stride[1]),
            ceil_mode=False)

        output_size = np.array([pool_size[1], pool_size[1]])
        stride = np.floor((input_size / output_size)).astype(int)
        kernel_size = (input_size - (output_size - 1) * stride).astype(int)
        self.pool2 = nn.AvgPool2d(
            kernel_size=(kernel_size[0], kernel_size[1]),
            stride=(stride[0], stride[1]),
            ceil_mode=False)

        output_size = np.array([1, input_size[1]])
        stride = np.floor((input_size / output_size)).astype(int)
        kernel_size = (input_size - (output_size - 1) * stride).astype(int)
        self.pool3 = nn.AvgPool2d(
            kernel_size=(kernel_size[0], kernel_size[1]),
            stride=(stride[0], stride[1]),
            ceil_mode=False)

        output_size = np.array([input_size[0], 1])
        stride = np.floor((input_size / output_size)).astype(int)
        kernel_size = (input_size - (output_size - 1) * stride).astype(int)
        self.pool4 = nn.AvgPool2d(
            kernel_size=(kernel_size[0], kernel_size[1]),
            stride=(stride[0], stride[1]),
            ceil_mode=False)

        inter_channels = in_channels // 4

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.conv2_0 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
            norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
            norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
            norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(
                inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
            norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(
                inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
            norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels * 2, in_channels, 1, bias=False),
            norm_layer(in_channels))

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w))
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w))
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w))
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w))
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


def visualize_a_data(x, y_bon, y_cor):
    x = (x.cpu().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    y_bon = y_bon.round().astype(int)
    gt_cor = np.zeros((30, 1024, 3), np.uint8)
    gt_cor[:] = y_cor[None, :, None] * 255
    img_pad = np.zeros((3, 1024, 3), np.uint8) + 255

    img_bon = (x.copy()).astype(np.uint8)
    img_bon[y_bon[0], np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1], np.arange(len(y_bon[1])), 1] = 255
    img_bon[y_bon[0] - 1, np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1] - 1, np.arange(len(y_bon[1])), 1] = 255
    img_bon[y_bon[0] + 1, np.arange(len(y_bon[0])), 1] = 255
    img_bon[y_bon[1] + 1, np.arange(len(y_bon[1])), 1] = 255
    return np.concatenate([gt_cor, img_pad, img_bon], 0)
