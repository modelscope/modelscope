# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn

FACE_PART_SIZE = 56


class InvertedResidual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 kernel_size,
                 stride,
                 padding,
                 expand_ratio=2,
                 use_connect=False,
                 activation='relu'):
        super(InvertedResidual, self).__init__()

        hid_channels = int(inp * expand_ratio)
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.ReLU(inplace=True),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        elif activation == 'prelu':
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_channels), nn.PReLU(hid_channels),
                nn.Conv2d(
                    hid_channels,
                    hid_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=hid_channels,
                    bias=False), nn.BatchNorm2d(hid_channels),
                nn.PReLU(hid_channels),
                nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup))
        self.use_connect = use_connect

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Residual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 kernel_size,
                 stride,
                 padding,
                 use_connect=False,
                 activation='relu'):
        super(Residual, self).__init__()

        self.use_connect = use_connect

        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        elif activation == 'prelu':
            self.conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size,
                    stride,
                    padding,
                    groups=inp,
                    bias=False), nn.BatchNorm2d(inp), nn.PReLU(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
                nn.PReLU(oup))

    def forward(self, x):
        if self.use_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.PReLU(oup))


def conv_no_relu(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup))


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Softmax(nn.Module):

    def __init__(self, dim):
        super(Softmax, self).__init__()
        self.softmax = nn.Softmax(dim)

    def forward(self, x):
        return self.softmax(x)


class LargeEyeballNet(nn.Module):

    def __init__(self):
        super(LargeEyeballNet, self).__init__()

        # v6/v7/v9
        # iris : -1*2, 3, FACE_PART_SIZE, FACE_PART_SIZE
        self.net = nn.Sequential(
            conv_bn(3, 16, 3, 2, 0),
            InvertedResidual(16, 16, 3, 1, 1, 2, True, activation='prelu'),
            InvertedResidual(16, 32, 3, 2, 0, 2, False, activation='prelu'),
            InvertedResidual(32, 32, 3, 1, 1, 2, True, activation='prelu'),
            InvertedResidual(32, 64, 3, 2, 1, 2, False, activation='prelu'),
            InvertedResidual(64, 64, 3, 1, 1, 2, True, activation='prelu'),
            InvertedResidual(64, 64, 3, 2, 0, 2, False, activation='prelu'),
            InvertedResidual(64, 64, 3, 1, 1, 2, True, activation='prelu'),
            View((-1, 64 * 3 * 3, 1, 1)), conv_bn(64 * 3 * 3, 64, 1, 1, 0),
            conv_no_relu(64, 40, 1, 1, 0), View((-1, 40)))

    def forward(self, x):  # x: -1, 3, FACE_PART_SIZE, FACE_PART_SIZE
        iris = self.net(x)

        return iris
