import re
import sys

import numpy as np
import torch
import torch.nn as nn

from .layers import (IdentityLayer, MBInvertedConvLayer,
                     MobileInvertedResidualBlock, ZeroLayer)
from .mix_ops import MixedEdge, build_candidate_ops, conv_func_by_name


class NasRecBackbone(nn.Module):

    def __init__(self, first_conv, blocks):
        super(NasRecBackbone, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.output_idx = [5, 11, 17, 23]

    def forward(self, x):
        x = self.first_conv(x)
        idx = 0
        out = []
        for block in self.blocks:
            x = block(x)
            if (idx + 1) % (int(len(self.blocks) / 4)) == 0:
                out.append(x)
            idx += 1
        return out[0], out[1], out[2], out[3]

    def get_bn_param(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                return {
                    'momentum': m.momentum,
                    'eps': m.eps,
                }
        return None

    @property
    def config(self):
        return {
            'name': NasRecBackbone.__name__,
            'bn': self.get_bn_param(),
            'first_conv': 'conv_in3_out32_k3_s2_p1',
            'blocks': [block.config for block in self.blocks]
        }

    def set_bn_param(self, momentum, eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.momentum = momentum
                m.eps = eps
        return

    @staticmethod
    def build_from_config(config):
        first_conv_config = config['first_conv']
        match_obj = re.match(r'conv_in(\d+)_out(\d+)_k(\d+)_s(\d+)_p(\d+)',
                             first_conv_config)
        in_channel = int(match_obj.group(1))
        out_channel = int(match_obj.group(2))
        kernel_size = int(match_obj.group(3))
        stride = int(match_obj.group(4))
        padding = int(match_obj.group(5))
        first_conv = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride,
                padding,
                bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for block_config in config['blocks']:
            blocks.append(
                MobileInvertedResidualBlock.build_from_config(block_config))
        net = NasRecBackbone(first_conv, blocks)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)
        return net


class CompactDetBackbone(NasRecBackbone):
    '''
    proxyless nas backbone, 5M.
    '''

    def __init__(self,
                 width_stages,
                 input_channel=None,
                 bn_param=(0.1, 1e-3),
                 **kwargs):

        if input_channel is None:
            input_channel = width_stages[0]
        first_conv = nn.Sequential(
            nn.Conv2d(
                3,
                input_channel,
                kernel_size=(3, 3),
                stride=2,
                padding=1,
                bias=False), nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True))

        conv_candidates = [
            '5x5_MBConv2', '5x5_MBConv4', '3x3_MBConv2', '3x3_MBConv4',
            '13_MixConv2', '13_MixConv4', '35_MixConv2', '35_MixConv4',
            '135_MixConv2', '135_MixConv4', '13_LinMixConv', '35_LinMixConv',
            '135_LinMixConv', '13_RepConv2', '13_RepConv4', '35_RepConv2',
            '35_RepConv4', '135_RepConv2', '135_RepConv4', 'Zero'
        ]

        se_candidates = ['SE_2', 'SE_4', 'SE_8', 'Zero']

        conv_op_ids = [
            15, 17, 17, 17, 17, 0, 16, 16, 18, 18, 16, 2, 16, 18, 16, 18, 18,
            2, 1, 18, 18, 18, 16, 2
        ]

        n_cell_stages = [5, 5, 5, 5]

        stride_stages = [(2, 2), (2, 2), (2, 2), (2, 2)]

        if se_candidates:
            assert len(conv_op_ids) == sum(n_cell_stages) + 4
        else:
            assert len(conv_op_ids) == sum(n_cell_stages)

        blocks = []

        for width, n_cell, s in zip(width_stages, n_cell_stages,
                                    stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = (1, 1)
                block_i = len(blocks)
                conv_op = conv_func_by_name(
                    conv_candidates[conv_op_ids[block_i]])(input_channel,
                                                           width, stride)

                if stride == (1, 1) and input_channel == width:
                    shortcut = IdentityLayer()
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(
                    conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_channel = width

            if se_candidates is not None:
                block_i = len(blocks)
                se_op = conv_func_by_name(se_candidates[conv_op_ids[block_i]])(
                    input_channel, width, stride)
                shortcut = IdentityLayer()
                inverted_residual_block = MobileInvertedResidualBlock(
                    se_op, shortcut)
                blocks.append(inverted_residual_block)

        self.out_channel = input_channel

        super(CompactDetBackbone, self).__init__(first_conv, blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
