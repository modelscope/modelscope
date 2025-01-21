# Copyright (c) 2022 Kensho Hara.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

# The implementation here is modified based on 3D-ResNets-PyTorch,
# originally MIT License, Copyright (c) 2022 Kensho Hara,
# and publicly available at https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet2p1d.py
""" ResNet2plus1d Model Architecture."""

import torch
import torch.nn as nn


def conv1x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, dilation, dilation),
        groups=groups,
        bias=False,
        dilation=(1, dilation, dilation))


def conv3x1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 1, 1),
        stride=(stride, 1, 1),
        padding=(dilation, 0, 0),
        groups=groups,
        bias=False,
        dilation=(dilation, 1, 1))


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')

        midplanes1 = (inplanes * planes * 3 * 3 * 3) // (
            inplanes * 3 * 3 + planes * 3)
        self.conv1_s = conv1x3x3(inplanes, midplanes1, stride)
        self.bn1_s = norm_layer(midplanes1)
        self.conv1_t = conv3x1x1(midplanes1, planes, stride)
        self.bn1_t = norm_layer(planes)

        midplanes2 = (planes * planes * 3 * 3 * 3) // (
            planes * 3 * 3 + planes * 3)
        self.conv2_s = conv1x3x3(planes, midplanes2)
        self.bn2_s = norm_layer(midplanes2)
        self.conv2_t = conv3x1x1(midplanes2, planes)
        self.bn2_t = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        midplanes = (width * width * 3 * 3 * 3) // (width * 3 * 3 + width * 3)
        self.conv2_s = conv1x3x3(width, midplanes, stride, groups, dilation)
        self.bn2_s = norm_layer(midplanes)
        self.conv2_t = conv3x1x1(midplanes, width, stride, groups, dilation)
        self.bn2_t = norm_layer(width)

        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2p1d(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=None,
                 zero_init_residual=True,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 dropout=0.5,
                 inplanes=3,
                 first_stride=2,
                 norm_layer=None,
                 last_pool=True):
        super(ResNet2p1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not last_pool and num_classes is not None:
            raise ValueError('num_classes should be None when last_pool=False')
        self._norm_layer = norm_layer
        self.first_stride = first_stride

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        midplanes = (3 * self.inplanes * 3 * 7 * 7) // (3 * 7 * 7
                                                        + self.inplanes * 3)
        self.conv1_s = nn.Conv3d(
            inplanes,
            midplanes,
            kernel_size=(1, 7, 7),
            stride=(1, first_stride, first_stride),
            padding=(0, 3, 3),
            bias=False)
        self.bn1_s = norm_layer(midplanes)
        self.conv1_t = nn.Conv3d(
            midplanes,
            self.inplanes,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False)
        self.bn1_t = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) if last_pool else None
        if num_classes is None:
            self.dropout = None
            self.fc = None
        else:
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_planes = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2_t.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avgpool:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.dropout and self.fc:
                x = self.dropout(x)
                x = self.fc(x)

        return x


def resnet10_2p1d(**kwargs):
    return ResNet2p1d(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18_2p1d(**kwargs):
    return ResNet2p1d(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet26_2p1d(**kwargs):
    return ResNet2p1d(Bottleneck, [2, 2, 2, 2], **kwargs)


def resnet34_2p1d(**kwargs):
    return ResNet2p1d(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_2p1d(**kwargs):
    return ResNet2p1d(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_2p1d(**kwargs):
    return ResNet2p1d(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_2p1d(**kwargs):
    return ResNet2p1d(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet200_2p1d(**kwargs):
    return ResNet2p1d(Bottleneck, [3, 24, 36, 3], **kwargs)
