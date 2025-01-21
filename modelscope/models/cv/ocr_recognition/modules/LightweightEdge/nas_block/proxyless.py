# Part of the implementation is borrowed and modified from ProxylessNAS,
# publicly available at https://github.com/mit-han-lab/proxylessnas
# paper linking at https://arxiv.org/pdf/1812.00332.pdf
import re
import sys
from queue import Queue

import numpy as np
import torch
import torch.nn as nn

from .layers import IdentityLayer, MobileInvertedResidualBlock
from .mix_ops import conv_func_by_name


class NasRecBackbone(nn.Module):

    def __init__(self, first_conv, blocks):
        super(NasRecBackbone, self).__init__()
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        return x

    def get_flops(self, x):
        expected_flops = 0
        # first conv
        flop = count_conv_flop(self.first_conv[0], x)
        x = self.first_conv(x)
        expected_flops += flop
        # blocks
        for mb_conv in self.blocks:
            assert isinstance(mb_conv, MobileInvertedResidualBlock)
            if mb_conv.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = mb_conv.shortcut.get_flops(x)
            expected_flops += shortcut_flop
            expected_flops += mb_conv.get_flops(x)[0]

            x = mb_conv(x)
        return expected_flops


class CompactRecBackboneMixSE(NasRecBackbone):

    def __init__(self, first_stride, input_channel, stride_stages,
                 n_cell_stages, width_stages, conv_op_ids, conv_candidates,
                 se_candidates):
        input_block_channel = 24
        first_conv = nn.Sequential(
            nn.Conv2d(
                input_channel,
                input_block_channel,
                kernel_size=(3, 3),
                stride=first_stride,
                padding=1,
                bias=False), nn.BatchNorm2d(input_block_channel), nn.PReLU())

        assert len(conv_op_ids) - 4 == sum(n_cell_stages)
        blocks = []
        img_height = 16
        height_flag = 0
        for width, n_cell, s in zip(width_stages, n_cell_stages,
                                    stride_stages):
            for i in range(n_cell):
                if i == 1:
                    img_height = int(img_height / 2)

                if img_height % 2 == 0:
                    height_flag = 1
                else:
                    height_flag = 0

                if i == 0:
                    stride = s
                else:
                    stride = (1, 1)
                block_i = len(blocks)
                conv_op = conv_func_by_name(
                    conv_candidates[conv_op_ids[block_i]])(
                        input_block_channel, width, stride,
                        img_height + height_flag)
                if stride == (1, 1) and input_block_channel == width:
                    shortcut = IdentityLayer()
                else:
                    shortcut = None
                inverted_residual_block = MobileInvertedResidualBlock(
                    conv_op, shortcut)
                blocks.append(inverted_residual_block)
                input_block_channel = width

            block_i = len(blocks)
            se_op = conv_func_by_name(se_candidates[conv_op_ids[block_i]])(
                input_block_channel, width, stride, img_height)

            inverted_residual_block = MobileInvertedResidualBlock(se_op, None)
            blocks.append(inverted_residual_block)
        self.out_channel = input_block_channel

        super(CompactRecBackboneMixSE, self).__init__(first_conv, blocks)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def plnas_linear_mix_se(input_channel, output_channel):
    conv_candidates = [
        '5x5_MBConv2', '5x5_MBConv4', '5x5_MBConv6', '3x3_MBConv2',
        '3x3_MBConv4', '3x3_MBConv6', '13_MixConv2', '13_MixConv4',
        '13_MixConv6', '35_MixConv2', '35_MixConv4', '35_MixConv6',
        '135_MixConv2', '135_MixConv4', '135_MixConv6', '13_LinMixConv',
        '35_LinMixConv', '135_LinMixConv', '13_RepConv2', '13_RepConv4',
        '13_RepConv6', '35_RepConv2', '35_RepConv4', '35_RepConv6',
        '135_RepConv2', '135_RepConv4', '135_RepConv6', 'Zero'
    ]
    se_candidates = ['SE_2', 'SE_4', 'SE_8', 'Zero']

    stride_stages = [(2, 2), (2, 1), (2, 1), (2, 1)]
    n_cell_stages = [5, 5, 5, 5]
    width_stages = [32, 64, 96, 128]
    conv_op_ids = [
        2, 23, 24, 26, 2, 2, 11, 27, 27, 27, 27, 2, 0, 2, 16, 10, 27, 2, 2, 2,
        22, 10, 27, 3
    ]
    net = CompactRecBackboneMixSE(2, input_channel, stride_stages,
                                  n_cell_stages, width_stages, conv_op_ids,
                                  conv_candidates, se_candidates)

    return net
