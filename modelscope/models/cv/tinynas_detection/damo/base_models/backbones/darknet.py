# Copyright (c) Megvii Inc. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from torch import nn

from modelscope.models.cv.tinynas_detection.damo.base_models.core.base_ops import (
    BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck)


class CSPDarknet(nn.Module):

    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=('dark3', 'dark4', 'dark5'),
        depthwise=False,
        act='silu',
        reparam=False,
    ):
        super(CSPDarknet, self).__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.stem = Focus(3, base_channels, 3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
                reparam=reparam,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
                reparam=reparam,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
                reparam=reparam,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(
                base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
                reparam=reparam,
            ),
        )

    def init_weights(self, pretrain=None):

        if pretrain is None:
            return
        else:
            pretrained_dict = torch.load(
                pretrain, map_location='cpu')['state_dict']
            new_params = self.state_dict().copy()
            for k, v in pretrained_dict.items():
                ks = k.split('.')
                if ks[0] == 'fc' or ks[-1] == 'total_ops' or ks[
                        -1] == 'total_params':
                    continue
                else:
                    new_params[k] = v

            self.load_state_dict(new_params)
            print(f' load pretrain backbone from {pretrain}')

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs['stem'] = x
        x = self.dark2(x)
        outputs['dark2'] = x
        x = self.dark3(x)
        outputs['dark3'] = x
        x = self.dark4(x)
        outputs['dark4'] = x
        x = self.dark5(x)
        outputs['dark5'] = x
        features_out = [
            outputs['stem'], outputs['dark2'], outputs['dark3'],
            outputs['dark4'], outputs['dark5']
        ]

        return features_out
