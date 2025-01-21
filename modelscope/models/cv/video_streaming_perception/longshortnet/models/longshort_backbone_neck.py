# Copyright (c) 2014-2021 Megvii Inc.
# Copyright (c) 2022-2023 Alibaba, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.stream_yolo.models.darknet import CSPDarknet
from modelscope.models.cv.stream_yolo.models.network_blocks import (BaseConv,
                                                                    CSPLayer,
                                                                    DWConv)


class BACKBONENECK(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=('dark3', 'dark4', 'dark5'),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act='silu',
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width),
            int(in_channels[1] * width),
            1,
            1,
            act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width),
            int(in_channels[0] * width),
            1,
            1,
            act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width),
            int(in_channels[0] * width),
            3,
            2,
            act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width),
            int(in_channels[1] * width),
            3,
            2,
            act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):

        rurrent_out_features = self.backbone(input)
        rurrent_features = [rurrent_out_features[f] for f in self.in_features]
        [rurrent_x2, rurrent_x1, rurrent_x0] = rurrent_features

        rurrent_fpn_out0 = self.lateral_conv0(rurrent_x0)
        rurrent_f_out0 = F.interpolate(
            rurrent_fpn_out0, size=rurrent_x1.shape[2:4], mode='nearest')
        rurrent_f_out0 = torch.cat([rurrent_f_out0, rurrent_x1], 1)
        rurrent_f_out0 = self.C3_p4(rurrent_f_out0)

        rurrent_fpn_out1 = self.reduce_conv1(rurrent_f_out0)
        rurrent_f_out1 = F.interpolate(
            rurrent_fpn_out1, size=rurrent_x2.shape[2:4], mode='nearest')
        rurrent_f_out1 = torch.cat([rurrent_f_out1, rurrent_x2], 1)
        rurrent_pan_out2 = self.C3_p3(rurrent_f_out1)

        rurrent_p_out1 = self.bu_conv2(rurrent_pan_out2)
        rurrent_p_out1 = torch.cat([rurrent_p_out1, rurrent_fpn_out1], 1)
        rurrent_pan_out1 = self.C3_n3(rurrent_p_out1)

        rurrent_p_out0 = self.bu_conv1(rurrent_pan_out1)
        rurrent_p_out0 = torch.cat([rurrent_p_out0, rurrent_fpn_out0], 1)
        rurrent_pan_out0 = self.C3_n4(rurrent_p_out0)

        outputs = (rurrent_pan_out2, rurrent_pan_out1, rurrent_pan_out0)

        return outputs
