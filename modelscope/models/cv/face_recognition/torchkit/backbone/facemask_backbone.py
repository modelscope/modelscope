# The implementation is adopted from InsightFace, made pubicly available under the Apache-2.0 license at
# https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/model.py

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import (AdaptiveAvgPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d,
                      Conv2d, Dropout, Dropout2d, Linear, MaxPool2d, Module,
                      Parameter, PReLU, ReLU, Sequential, Sigmoid)


class Flatten(Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels,
            channels // reduction,
            kernel_size=1,
            padding=0,
            bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction,
            channels,
            kernel_size=1,
            padding=0,
            bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BottleneckIR(Module):

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)
            ] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 252:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=6),
            get_block(in_channel=64, depth=128, num_units=21),
            get_block(in_channel=128, depth=256, num_units=66),
            get_block(in_channel=256, depth=512, num_units=6)
        ]
    return blocks


class IResNet(Module):

    def __init__(self,
                 dropout=0,
                 num_features=512,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 fp16=False,
                 with_wcd=False,
                 wrs_M=400,
                 wrs_q=0.9):
        super(IResNet, self).__init__()
        num_layers = 252
        mode = 'ir'
        assert num_layers in [50, 100, 152,
                              252], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.fc_scale = 7 * 7
        num_features = 512
        self.fp16 = fp16
        drop_ratio = 0.0
        self.with_wcd = with_wcd
        if self.with_wcd:
            self.wrs_M = wrs_M
            self.wrs_q = wrs_q
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = BottleneckIR
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64),
            PReLU(64))
        self.bn2 = nn.BatchNorm2d(
            512,
            eps=1e-05,
        )
        self.dropout = nn.Dropout(p=drop_ratio, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.input_layer(x)
            x = self.body(x)
            x = self.bn2(x)
            if self.with_wcd:
                B = x.size()[0]
                C = x.size()[1]
                x_abs = torch.abs(x)
                score = torch.nn.functional.adaptive_avg_pool2d(x_abs,
                                                                1).reshape(
                                                                    (B, C))
                r = torch.rand((B, C), device=x.device)
                key = torch.pow(r, 1. / score)
                _, topidx = torch.topk(key, self.wrs_M, dim=1)
                mask = torch.zeros_like(key, dtype=torch.float32)
                mask.scatter_(1, topidx, 1.)
                maskq = torch.rand((B, C), device=x.device)
                maskq_ones = torch.ones_like(maskq, dtype=torch.float32)
                maskq_zeros = torch.zeros_like(maskq, dtype=torch.float32)
                maskq_m = torch.where(maskq < self.wrs_q, maskq_ones,
                                      maskq_zeros)
                new_mask = mask * maskq_m
                score_sum = torch.sum(score, dim=1, keepdim=True)
                selected_score_sum = torch.sum(
                    new_mask * score, dim=1, keepdim=True)
                alpha = score_sum / (selected_score_sum + 1e-6)
                alpha = alpha.reshape((B, 1, 1, 1))
                new_mask = new_mask.reshape((B, C, 1, 1))
                x = x * new_mask * alpha
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def iresnet286(pretrained=False, progress=True, **kwargs):
    model = IResNet(
        dropout=0,
        num_features=512,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        fp16=False,
        with_wcd=False,
        wrs_M=400,
        wrs_q=0.9)
    return model
