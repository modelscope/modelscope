# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from modelscope.models.cv.image_instance_segmentation.maskdino.utils import \
    Conv2d


# This is a modified FPN decoder.
class BaseFPN(nn.Module):

    def __init__(
        self,
        input_shape,
        *,
        convs_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        Args:
            input_shape: shapes (channels and stride) of the input features
            convs_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1]['stride'])
        self.in_features = [k for k, v in input_shape
                            ]  # starting from "res3" to "res5"
        feature_channels = [v['channels'] for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ''
        for idx, in_channels in enumerate(feature_channels):
            lateral_norm = nn.GroupNorm(32, convs_dim)
            output_norm = nn.GroupNorm(32, convs_dim)

            lateral_conv = Conv2d(
                in_channels,
                convs_dim,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm)
            output_conv = Conv2d(
                convs_dim,
                convs_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.convs_dim = convs_dim
        self.num_feature_levels = 3  # always use 3 scales

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if idx == 0:
                y = lateral_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(
                    y,
                    size=cur_fpn.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            y = output_conv(y)

            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning(
            'Calling forward() may cause unpredicted behavior of PixelDecoder module.'
        )
        return self.forward_features(features)


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes])
        self.bottleneck = Conv2d(in_channels + len(sizes) * channels,
                                 in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=F.relu_(stage(feats)),
                size=(h, w),
                mode='bilinear',
                align_corners=False) for stage in self.stages
        ] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


class PyramidPoolingModuleFPN(BaseFPN):

    def __init__(
        self,
        input_shape,
        *,
        convs_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            convs_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(
            input_shape, convs_dim=convs_dim, mask_dim=mask_dim, norm=norm)
        self.ppm = PyramidPoolingModule(convs_dim, convs_dim // 4)

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if idx == 0:
                y = self.ppm(lateral_conv(x))
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(
                    y,
                    size=cur_fpn.shape[-2:],
                    mode='bilinear',
                    align_corners=False)
            y = output_conv(y)

            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1

        return None, multi_scale_features
