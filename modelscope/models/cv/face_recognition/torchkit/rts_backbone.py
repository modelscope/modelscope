# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import namedtuple
from math import lgamma

import torch
import torch.nn as nn
from torch.nn import (AdaptiveAvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d,
                      Dropout, Linear, MaxPool2d, Module, PReLU, ReLU,
                      Sequential, Sigmoid)
from torch.nn.modules.flatten import Flatten

from modelscope.models import MODELS
from modelscope.models.base import TorchModel
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module('face-recognition', 'rts-backbone')
class RTSBackbone(TorchModel):

    def __init__(self, *args, **kwargs):
        super(RTSBackbone, self).__init__()
        # model initialization
        self.alpha = kwargs.get('alpha')
        self.rts_plus = kwargs.get('rts_plus')
        resnet = Backbone([112, 112], 64, mode='ir_se')

        self.features = nn.Sequential(
            resnet.input_layer, resnet.body,
            Sequential(
                BatchNorm2d(512),
                Dropout(),
                Flatten(),
            ))

        self.features_backbone = nn.Sequential(
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512),
        )

        self.logvar_rts_backbone = nn.Sequential(
            Linear(512 * 7 * 7, 1),
            BatchNorm1d(1),
        )

        self.logvar_rts_plus_backbone = nn.Sequential(
            Linear(512 * 7 * 7, self.alpha),
            BatchNorm1d(self.alpha),
        )

    def forward(self, img):
        x = self.features(img)
        image_features = self.features_backbone(x)
        if not self.rts_plus:
            logvar = self.logvar_rts_backbone(x)
        else:
            logvar = self.logvar_rts_plus_backbone(x)
        return image_features, logvar

    @classmethod
    def _instantiate(cls, **kwargs):
        model_file = kwargs.get('am_model_name', ModelFile.TORCH_MODEL_FILE)
        ckpt_path = os.path.join(kwargs['model_dir'], model_file)
        logger.info(f'loading model from {ckpt_path}')
        model_dir = kwargs.pop('model_dir')
        model = cls(**kwargs)
        ckpt_path = os.path.join(model_dir, model_file)
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        return model


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


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

        nn.init.xavier_uniform_(self.fc1.weight.data)

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


class bottleneck_IR_SE(Module):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
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
            BatchNorm2d(depth), SEModule(depth, 16))

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
    elif num_layers == 64:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=16),
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

    return blocks


class Backbone(Module):

    def __init__(self, input_size, num_layers, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [
            112, 224
        ], 'input_size should be [112, 112] or [224, 224]'
        assert num_layers in [50, 64, 100,
                              152], 'num_layers should be 50, 64, 100 or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64),
            PReLU(64))
        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2d(512), Dropout(), Flatten(),
                Linear(512 * 7 * 7, 512), BatchNorm1d(512))
        else:
            self.output_layer = Sequential(
                BatchNorm2d(512), Dropout(), Flatten(),
                Linear(512 * 14 * 14, 512), BatchNorm1d(512))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)

        return x
