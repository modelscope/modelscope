# Copyright (c) 2014-2021 Megvii Inc.
# Copyright (c) 2022-2023 Alibaba, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from modelscope.models.cv.stream_yolo.models.network_blocks import BaseConv


class LONGSHORT(nn.Module):

    def __init__(self,
                 long_backbone=None,
                 short_backbone=None,
                 backbone_neck=None,
                 head=None,
                 merge_form='add',
                 in_channels=[256, 512, 1024],
                 width=1.0,
                 act='silu',
                 with_short_cut=False,
                 long_cfg=None,
                 jian_ratio=None):
        super().__init__()

        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.backbone = backbone_neck
        self.head = head
        self.merge_form = merge_form
        self.in_channels = in_channels
        self.with_short_cut = with_short_cut
        if merge_form == 'concat':
            self.jian2 = BaseConv(
                in_channels=int(in_channels[0] * width),
                out_channels=int(in_channels[0] * width)
                // 2 if jian_ratio is None else int(in_channels[0] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian1 = BaseConv(
                in_channels=int(in_channels[1] * width),
                out_channels=int(in_channels[1] * width)
                // 2 if jian_ratio is None else int(in_channels[1] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian0 = BaseConv(
                in_channels=int(in_channels[2] * width),
                out_channels=int(in_channels[2] * width)
                // 2 if jian_ratio is None else int(in_channels[2] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )
        elif merge_form == 'long_fusion':
            assert long_cfg is not None and 'out_channels' in long_cfg
            self.jian2 = BaseConv(
                in_channels=sum(
                    [x[0][0] * x[1] for x in long_cfg['out_channels']]),
                out_channels=int(in_channels[0] * width)
                // 2 if jian_ratio is None else int(in_channels[0] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian1 = BaseConv(
                in_channels=sum(
                    [x[0][1] * x[1] for x in long_cfg['out_channels']]),
                out_channels=int(in_channels[1] * width)
                // 2 if jian_ratio is None else int(in_channels[1] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )

            self.jian0 = BaseConv(
                in_channels=sum(
                    [x[0][2] * x[1] for x in long_cfg['out_channels']]),
                out_channels=int(in_channels[2] * width)
                // 2 if jian_ratio is None else int(in_channels[2] * width
                                                    * jian_ratio),
                ksize=1,
                stride=1,
                act=act,
            )

    def forward(self, x, targets=None, buffer=None, mode='off_pipe'):
        assert mode in ['off_pipe', 'on_pipe']

        if mode == 'off_pipe':
            short_fpn_outs, rurrent_pan_outs = self.short_backbone(
                x[0],
                buffer=buffer,
                mode='off_pipe',
                backbone_neck=self.backbone)
            long_fpn_outs = self.long_backbone(
                x[1],
                buffer=buffer,
                mode='off_pipe',
                backbone_neck=self.backbone
            ) if self.long_backbone is not None else None
            if not self.with_short_cut:
                if self.long_backbone is None:
                    fpn_outs = short_fpn_outs
                else:
                    if self.merge_form == 'add':
                        fpn_outs = [
                            x + y
                            for x, y in zip(short_fpn_outs, long_fpn_outs)
                        ]
                    elif self.merge_form == 'concat':
                        jian2_outs = [
                            self.jian2(short_fpn_outs[0]),
                            self.jian2(long_fpn_outs[0])
                        ]
                        jian1_outs = [
                            self.jian1(short_fpn_outs[1]),
                            self.jian1(long_fpn_outs[1])
                        ]
                        jian0_outs = [
                            self.jian0(short_fpn_outs[2]),
                            self.jian0(long_fpn_outs[2])
                        ]
                        fpn_outs_2 = torch.cat(jian2_outs, dim=1)
                        fpn_outs_1 = torch.cat(jian1_outs, dim=1)
                        fpn_outs_0 = torch.cat(jian0_outs, dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == 'pure_concat':
                        fpn_outs_2 = torch.cat(
                            [short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat(
                            [short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat(
                            [short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == 'long_fusion':
                        fpn_outs_2 = torch.cat(
                            [short_fpn_outs[0],
                             self.jian2(long_fpn_outs[0])],
                            dim=1)
                        fpn_outs_1 = torch.cat(
                            [short_fpn_outs[1],
                             self.jian1(long_fpn_outs[1])],
                            dim=1)
                        fpn_outs_0 = torch.cat(
                            [short_fpn_outs[2],
                             self.jian0(long_fpn_outs[2])],
                            dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    else:
                        raise Exception(
                            'merge_form must be in ["add", "concat"]')
            else:
                if self.long_backbone is None:
                    fpn_outs = [
                        x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)
                    ]
                else:
                    if self.merge_form == 'add':
                        fpn_outs = [
                            x + y + z
                            for x, y, z in zip(short_fpn_outs, long_fpn_outs,
                                               rurrent_pan_outs)
                        ]
                    elif self.merge_form == 'concat':
                        jian2_outs = [
                            self.jian2(short_fpn_outs[0]),
                            self.jian2(long_fpn_outs[0])
                        ]
                        jian1_outs = [
                            self.jian1(short_fpn_outs[1]),
                            self.jian1(long_fpn_outs[1])
                        ]
                        jian0_outs = [
                            self.jian0(short_fpn_outs[2]),
                            self.jian0(long_fpn_outs[2])
                        ]
                        fpn_outs_2 = torch.cat(jian2_outs, dim=1)
                        fpn_outs_1 = torch.cat(jian1_outs, dim=1)
                        fpn_outs_0 = torch.cat(jian0_outs, dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [
                            x + y for x, y in zip(fpn_outs, rurrent_pan_outs)
                        ]
                    elif self.merge_form == 'pure_concat':
                        fpn_outs_2 = torch.cat(
                            [short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat(
                            [short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat(
                            [short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [
                            x + y for x, y in zip(fpn_outs, rurrent_pan_outs)
                        ]
                    elif self.merge_form == 'long_fusion':
                        fpn_outs_2 = torch.cat(
                            [short_fpn_outs[0],
                             self.jian2(long_fpn_outs[0])],
                            dim=1)
                        fpn_outs_1 = torch.cat(
                            [short_fpn_outs[1],
                             self.jian1(long_fpn_outs[1])],
                            dim=1)
                        fpn_outs_0 = torch.cat(
                            [short_fpn_outs[2],
                             self.jian0(long_fpn_outs[2])],
                            dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [
                            x + y for x, y in zip(fpn_outs, rurrent_pan_outs)
                        ]
                    else:
                        raise Exception(
                            'merge_form must be in ["add", "concat"]')

            outputs = self.head(fpn_outs)

            return outputs
        else:
            raise NotImplementedError
