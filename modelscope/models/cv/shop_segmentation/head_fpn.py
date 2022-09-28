# Base modules are adapted from https://github.com/open-mmlab/mmcv/,
# originally Apache 2.0 License, Copyright (c) 2018-2022 OpenMMLab,
# https://github.com/open-mmlab/mmsegmentation/,
# originally Apache 2.0 License, Copyright (c) 2020-2021 OpenMMLab,
# and adapted from https://github.com/raoyongming/DenseCLIP/,
# originally MIT License, Copyright (c) 2022 Rao, Yongming.

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from timm.models.layers import drop, drop_path, trunc_normal_

from .common import Upsample, resize


class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks.
    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.
    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,
                 channels,
                 num_classes,
                 dropout_ratio=0.1,
                 feature_strides=[4, 8, 16, 32],
                 align_corners=False,
                 **kwargs):
        super(FPNHead, self).__init__()
        self.act_cfg = dict(type='ReLU')
        self.channels = channels
        self.conv_cfg = None
        self.norm_cfg = None
        self.norm_cfg = dict(type='BN2d', requires_grad=True)
        self.align_corners = align_corners
        self.dropout_ratio = dropout_ratio
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.in_index = [0, 1, 2, 3]
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.apply(self._init_weights)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        inputs = [inputs[i] for i in self.in_index]
        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
