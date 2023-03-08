# Copyright (c) 2014-2021 Megvii Inc.
# Copyright (c) 2022-2023 Alibaba, Inc. and its affiliates. All rights reserved.

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.stream_yolo.models.darknet import CSPDarknet
from modelscope.models.cv.stream_yolo.models.network_blocks import (BaseConv,
                                                                    DWConv)


class DFPPAFPNSHORT(nn.Module):

    def __init__(self,
                 depth=1.0,
                 width=1.0,
                 in_features=('dark3', 'dark4', 'dark5'),
                 in_channels=[256, 512, 1024],
                 depthwise=False,
                 act='silu',
                 frame_num=2,
                 with_short_cut=True,
                 out_channels=[
                     ((64, 128, 256), 1),
                 ]):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        self.frame_num = frame_num
        self.with_short_cut = with_short_cut
        self.out_channels = out_channels
        self.conv_group_num = len(out_channels)
        self.conv_group_dict = defaultdict(dict)
        assert self.frame_num == sum([x[1] for x in out_channels])
        Conv = DWConv if depthwise else BaseConv

        for i in range(self.conv_group_num):
            setattr(
                self, f'group_{i}_jian2',
                Conv(
                    in_channels=int(in_channels[0] * width),
                    out_channels=self.out_channels[i][0][0],
                    ksize=1,
                    stride=1,
                    act=act,
                ))

            setattr(
                self, f'group_{i}_jian1',
                Conv(
                    in_channels=int(in_channels[1] * width),
                    out_channels=self.out_channels[i][0][1],
                    ksize=1,
                    stride=1,
                    act=act,
                ))

            setattr(
                self, f'group_{i}_jian0',
                Conv(
                    in_channels=int(in_channels[2] * width),
                    out_channels=self.out_channels[i][0][2],
                    ksize=1,
                    stride=1,
                    act=act,
                ))

    def off_forward(self, input, backbone_neck):

        rurrent_pan_out2, rurrent_pan_out1, rurrent_pan_out0 = backbone_neck(
            torch.split(input, 3, dim=1)[0])

        support_pan_out2s = []
        support_pan_out1s = []
        support_pan_out0s = []
        for i in range(self.frame_num - 1):

            support_pan_out2, support_pan_out1, support_pan_out0 = backbone_neck(
                torch.split(input, 3, dim=1)[i + 1])

            support_pan_out2s.append(support_pan_out2)
            support_pan_out1s.append(support_pan_out1)
            support_pan_out0s.append(support_pan_out0)

        all_pan_out2s = [rurrent_pan_out2] + support_pan_out2s
        all_pan_out1s = [rurrent_pan_out1] + support_pan_out1s
        all_pan_out0s = [rurrent_pan_out0] + support_pan_out0s
        pan_out2s = []
        pan_out1s = []
        pan_out0s = []

        frame_start_id = 0
        for i in range(self.conv_group_num):
            group_frame_num = self.out_channels[i][1]
            for j in range(group_frame_num):
                frame_id = frame_start_id + j
                pan_out2s.append(
                    getattr(self, f'group_{i}_jian2')(all_pan_out2s[frame_id]))
                pan_out1s.append(
                    getattr(self, f'group_{i}_jian1')(all_pan_out1s[frame_id]))
                pan_out0s.append(
                    getattr(self, f'group_{i}_jian0')(all_pan_out0s[frame_id]))
            frame_start_id += group_frame_num

        if self.with_short_cut:
            pan_out2 = torch.cat(pan_out2s, dim=1) + rurrent_pan_out2
            pan_out1 = torch.cat(pan_out1s, dim=1) + rurrent_pan_out1
            pan_out0 = torch.cat(pan_out0s, dim=1) + rurrent_pan_out0
        else:
            pan_out2 = torch.cat(pan_out2s, dim=1)
            pan_out1 = torch.cat(pan_out1s, dim=1)
            pan_out0 = torch.cat(pan_out0s, dim=1)

        outputs = (pan_out2, pan_out1, pan_out0)
        rurrent_pan_outs = (rurrent_pan_out2, rurrent_pan_out1,
                            rurrent_pan_out0)

        return outputs, rurrent_pan_outs

    def forward(self, input, buffer=None, mode='off_pipe', backbone_neck=None):

        if mode == 'off_pipe':
            if input.size()[1] == 3:
                input = torch.cat([input, input], dim=1)
                output = self.off_forward(input, backbone_neck)
            else:
                output = self.off_forward(input, backbone_neck)

            return output

        else:
            raise NotImplementedError
