# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from timm.models.layers import (Swish, create_conv2d, create_pool2d,
                                get_act_layer)

from modelscope.models.cv.tinynas_detection.damo.base_models.core.base_ops import (
    CSPLayer, ShuffleBlock, ShuffleCSPLayer)
from .giraffe_config import get_graph_config

_ACT_LAYER = Swish


class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""

    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x


class ConvBnAct2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 padding='',
                 bias=False,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()

        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding='',
                 bias=False,
                 channel_multiplier=1.0,
                 pw_kernel_size=1,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=_ACT_LAYER):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True)

        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias)

        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def _init_weight(
    m,
    n='',
):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError(
                'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions'
            )
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, SeparableConv2d):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
    elif isinstance(m, ConvBnAct2d):
        if 'box_net' in n or 'class_net' in n:
            m.conv.weight.data.normal_(std=.01)
            if m.conv.bias is not None:
                if 'class_net.predict' in n:
                    m.conv.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv.bias.data.zero_()
        else:
            _glorot_uniform(m.conv.weight)
            if m.conv.bias is not None:
                m.conv.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


def _init_weight_alt(
    m,
    n='',
):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()


class Interpolate2d(nn.Module):
    r"""Resamples a 2d Image

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int], optional):
            output spatial sizes
        scale_factor (float or Tuple[float] or Tuple[float, float] or Tuple[float, float, float], optional):
            multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str, optional): the upsampling algorithm: one of ``'nearest'``,
            ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
            Default: ``'nearest'``
        align_corners (bool, optional): if ``True``, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is
            ``'linear'``, ``'bilinear'``, or ``'trilinear'``. Default: ``False``
    """
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(self,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 mode: str = 'nearest',
                 align_corners: bool = False) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False)


class ResampleFeatureMap(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduction_ratio=1.,
                 pad_type='',
                 downsample=None,
                 upsample=None,
                 norm_layer=nn.BatchNorm2d,
                 apply_bn=False,
                 conv_after_downsample=False,
                 redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = ConvBnAct2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias,
                act_layer=None)

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            if downsample in ('max', 'avg'):
                stride_size = int(reduction_ratio)
                downsample = create_pool2d(
                    downsample,
                    kernel_size=stride_size + 1,
                    stride=stride_size,
                    padding=pad_type)
            else:
                downsample = Interpolate2d(
                    scale_factor=1. / reduction_ratio, mode=downsample)
            self.add_module('downsample', downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module(
                    'upsample',
                    Interpolate2d(scale_factor=scale, mode=upsample))


class GiraffeCombine(nn.Module):

    def __init__(self,
                 feature_info,
                 fpn_config,
                 fpn_channels,
                 inputs_offsets,
                 target_reduction,
                 pad_type='',
                 downsample=None,
                 upsample=None,
                 norm_layer=nn.BatchNorm2d,
                 apply_resample_bn=False,
                 conv_after_downsample=False,
                 redundant_bias=False,
                 weight_method='attn'):
        super(GiraffeCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        reduction_base = feature_info[0]['reduction']

        target_channels_idx = int(
            math.log(target_reduction // reduction_base, 2))
        for idx, offset in enumerate(inputs_offsets):
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset
                input_reduction = fpn_config[node_idx]['reduction']
                # in_channels = fpn_config[node_idx]['num_chs']
                input_channels_idx = int(
                    math.log(input_reduction // reduction_base, 2))
                in_channels = feature_info[input_channels_idx]['num_chs']

            reduction_ratio = target_reduction / input_reduction
            if weight_method == 'concat':
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channels,
                    in_channels,
                    reduction_ratio=reduction_ratio,
                    pad_type=pad_type,
                    downsample=downsample,
                    upsample=upsample,
                    norm_layer=norm_layer,
                    apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample,
                    redundant_bias=redundant_bias)
            else:
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channels,
                    fpn_channels[target_channels_idx],
                    reduction_ratio=reduction_ratio,
                    pad_type=pad_type,
                    downsample=downsample,
                    upsample=upsample,
                    norm_layer=norm_layer,
                    apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample,
                    redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(
                torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        if len(self.inputs_offsets) == 0:
            return None
        for offset, resample in zip(self.inputs_offsets,
                                    self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(
                self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(
                self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            weights_norm = weights_sum + 0.0001
            out = torch.stack([(nodes[i] * edge_weights[i]) / weights_norm
                               for i in range(len(nodes))],
                              dim=-1)

            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'concat':
            out = torch.cat(nodes, dim=1)
        else:
            raise ValueError('unknown weight_method {}'.format(
                self.weight_method))
        return out


class GiraffeNode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(GiraffeNode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        combine_feat = self.combine(x)
        if combine_feat is None:
            return None
        else:
            return self.after_combine(combine_feat)


class GiraffeLayer(nn.Module):

    def __init__(self,
                 feature_info,
                 fpn_config,
                 inner_fpn_channels,
                 outer_fpn_channels,
                 num_levels=5,
                 pad_type='',
                 downsample=None,
                 upsample=None,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=_ACT_LAYER,
                 apply_resample_bn=False,
                 conv_after_downsample=True,
                 conv_bn_relu_pattern=False,
                 separable_conv=True,
                 redundant_bias=False,
                 merge_type='conv'):
        super(GiraffeLayer, self).__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = {}
        for idx, feat in enumerate(feature_info):
            self.feature_info[idx] = feat

        self.fnode = nn.ModuleList()
        reduction_base = feature_info[0]['reduction']
        for i, fnode_cfg in fpn_config.items():

            if fnode_cfg['is_out'] == 1:
                fpn_channels = outer_fpn_channels
            else:
                fpn_channels = inner_fpn_channels

            reduction = fnode_cfg['reduction']
            fpn_channels_idx = int(math.log(reduction // reduction_base, 2))
            combine = GiraffeCombine(
                self.feature_info,
                fpn_config,
                fpn_channels,
                tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()

            in_channels = 0
            out_channels = 0
            for input_offset in fnode_cfg['inputs_offsets']:
                in_channels += self.feature_info[input_offset]['num_chs']

            out_channels = fpn_channels[fpn_channels_idx]

            if merge_type == 'csp':
                after_combine.add_module(
                    'CspLayer',
                    CSPLayer(
                        in_channels,
                        out_channels,
                        2,
                        shortcut=True,
                        depthwise=False,
                        act='silu'))
            elif merge_type == 'shuffle':
                after_combine.add_module(
                    'shuffleBlock', ShuffleBlock(in_channels, in_channels))
                after_combine.add_module(
                    'conv1x1',
                    create_conv2d(in_channels, out_channels, kernel_size=1))
            elif merge_type == 'conv':
                after_combine.add_module(
                    'conv1x1',
                    create_conv2d(in_channels, out_channels, kernel_size=1))
                conv_kwargs = dict(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=pad_type,
                    bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer)
                if not conv_bn_relu_pattern:
                    conv_kwargs['bias'] = redundant_bias
                    conv_kwargs['act_layer'] = None
                    after_combine.add_module('act', act_layer(inplace=True))
                after_combine.add_module(
                    'conv',
                    SeparableConv2d(**conv_kwargs)
                    if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(
                GiraffeNode(combine=combine, after_combine=after_combine))
            self.feature_info[i] = dict(
                num_chs=fpn_channels[fpn_channels_idx], reduction=reduction)

        self.out_feature_info = []
        out_node = list(self.feature_info.keys())[-num_levels::]
        for i in out_node:
            self.out_feature_info.append(self.feature_info[i])

        self.feature_info = self.out_feature_info

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class GiraffeNeck(nn.Module):

    def __init__(self, min_level, max_level, num_levels, norm_layer,
                 norm_kwargs, act_type, fpn_config, fpn_name, fpn_channels,
                 out_fpn_channels, weight_method, depth_multiplier,
                 width_multiplier, with_backslash, with_slash,
                 with_skip_connect, skip_connect_type, separable_conv,
                 feature_info, merge_type, pad_type, downsample_type,
                 upsample_type, apply_resample_bn, conv_after_downsample,
                 redundant_bias, conv_bn_relu_pattern, alternate_init):
        super(GiraffeNeck, self).__init__()

        self.num_levels = num_levels
        self.min_level = min_level
        self.in_features = [0, 1, 2, 3, 4, 5,
                            6][self.min_level - 1:self.min_level - 1
                               + num_levels]
        self.alternate_init = alternate_init
        norm_layer = norm_layer or nn.BatchNorm2d
        if norm_kwargs:
            norm_layer = partial(norm_layer, **norm_kwargs)
        act_layer = get_act_layer(act_type) or _ACT_LAYER
        fpn_config = fpn_config or get_graph_config(
            fpn_name,
            min_level=min_level,
            max_level=max_level,
            weight_method=weight_method,
            depth_multiplier=depth_multiplier,
            with_backslash=with_backslash,
            with_slash=with_slash,
            with_skip_connect=with_skip_connect,
            skip_connect_type=skip_connect_type)

        # width scale
        for i in range(len(fpn_channels)):
            fpn_channels[i] = int(fpn_channels[i] * width_multiplier)

        self.resample = nn.ModuleDict()
        for level in range(num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=feature_info[level - 1]['num_chs'],
                    pad_type=pad_type,
                    downsample=downsample_type,
                    upsample=upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    apply_bn=apply_resample_bn,
                    conv_after_downsample=conv_after_downsample,
                    redundant_bias=redundant_bias,
                )
                in_chs = feature_info[level - 1]['num_chs']
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        giraffe_layer = GiraffeLayer(
            feature_info=feature_info,
            fpn_config=fpn_config,
            inner_fpn_channels=fpn_channels,
            outer_fpn_channels=out_fpn_channels,
            num_levels=num_levels,
            pad_type=pad_type,
            downsample=downsample_type,
            upsample=upsample_type,
            norm_layer=norm_layer,
            act_layer=act_layer,
            separable_conv=separable_conv,
            apply_resample_bn=apply_resample_bn,
            conv_after_downsample=conv_after_downsample,
            conv_bn_relu_pattern=conv_bn_relu_pattern,
            redundant_bias=redundant_bias,
            merge_type=merge_type)
        self.cell.add_module('giraffeNeck', giraffe_layer)
        feature_info = giraffe_layer.feature_info

    def init_weights(self, pretrained=False):
        for n, m in self.named_modules():
            if 'backbone' not in n:
                if self.alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

    def forward(self, x: List[torch.Tensor]):
        if type(x) is tuple:
            x = list(x)
        x = [x[f] for f in self.in_features]
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x
