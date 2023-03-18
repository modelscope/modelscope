# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------

import copy
import math
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=0,
        bias=False)


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.BN_MOMENTUM = 0.1
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.BN_MOMENTUM = 0.1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class LoreDetectModel(nn.Module):
    """
    A key point-based detector with ResNet backbone. In this model, it is trained for table cell detection.
    See details in paper "LORE: Logical Location Regression Network for Table Structure Recognition"
    (https://arxiv.org/abs/2303.03730)
    """

    def __init__(self, **kwargs):
        '''
            Args:
        '''
        self.BN_MOMENTUM = 0.1
        self.inplanes = 64
        self.deconv_with_bias = False
        self.block = BasicBlock
        self.layers = [2, 2, 2, 2]
        self.heads = {
            'hm': 2,
            'st': 8,
            'wh': 8,
            'ax': 256,
            'cr': 256,
            'reg': 2
        }
        self.head_conv = 64

        super(LoreDetectModel, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            self.block, 64, self.layers[0], stride=2)
        self.layer2 = self._make_layer(
            self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(
            self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(
            self.block, 256, self.layers[3], stride=2)

        self.adaption3 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption2 = nn.Conv2d(
            128, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption1 = nn.Conv2d(
            64, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.adaption0 = nn.Conv2d(
            64, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.adaptionU1 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0, bias=False)

        # used for deconv layers
        self.deconv_layers1 = self._make_deconv_layer(
            1,
            [256],
            [4],
        )
        self.deconv_layers2 = self._make_deconv_layer(
            1,
            [256],
            [4],
        )
        self.deconv_layers3 = self._make_deconv_layer(
            1,
            [256],
            [4],
        )
        self.deconv_layers4 = self._make_deconv_layer(
            1,
            [256],
            [4],
        )

        self.hm_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.hm_sigmoid = nn.Sigmoid()
        self.mk_maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.mk_sigmoid = nn.Sigmoid()

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if self.head_conv > 0 and (head == 'reg' or head == 'mk_reg'):
                inchannel = 256
                fc = nn.Sequential(
                    nn.Conv2d(
                        inchannel,
                        self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv,
                        num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0))
            elif self.head_conv > 0:
                inchannel = 256
                fc = nn.Sequential(
                    nn.Conv2d(
                        inchannel,
                        self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv,
                        self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv,
                        self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv,
                        self.head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True), nn.ReLU(inplace=True),
                    nn.Conv2d(
                        self.head_conv,
                        num_output,
                        kernel_size=1,
                        stride=1,
                        padding=0))
            else:
                inchannel = 256
                fc = nn.Conv2d(
                    in_channels=inchannel,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(
                    planes * block.expansion, momentum=self.BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 7:
            padding = 3
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x : Input image, a tensor of [batch_size, channel, w, h].

        Returns:
            ret : A dict of tensors, the keys are corresponding to the keys of head as initialized,
                  and the value is tensors of [batch_size, dim_key ,w, h],
                  where dim_key is different according to different keys. For example,
                  in this implementation, the dim_keys are 2, 8, 8, 256, 256, 2.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x3_ = self.deconv_layers1(x4)
        x3_ = self.adaption3(x3) + x3_

        x2_ = self.deconv_layers2(x3_)
        x2_ = self.adaption2(x2) + x2_

        x1_ = self.deconv_layers3(x2_)
        x1_ = self.adaption1(x1) + x1_

        x0_ = self.deconv_layers4(x1_) + self.adaption0(x0)
        x0_ = self.adaptionU1(x0_)

        ret = {}

        for head in self.heads:
            ret[head] = self.__getattr__(head)(x0_)
        return [ret]
