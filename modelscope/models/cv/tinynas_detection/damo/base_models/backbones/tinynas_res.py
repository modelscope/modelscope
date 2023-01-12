# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn

from modelscope.models.cv.tinynas_detection.damo.base_models.core.ops import (
    Focus, RepConv, SPPBottleneck, get_activation)
from modelscope.utils.file_utils import read_file


class ConvKXBN(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride):
        super(ConvKXBN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_c,
            out_c,
            kernel_size,
            stride, (kernel_size - 1) // 2,
            groups=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn1(self.conv1(x))

    def fuseforward(self, x):
        return self.conv1(x)


class ConvKXBNRELU(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, stride, act='silu'):
        super(ConvKXBNRELU, self).__init__()
        self.conv = ConvKXBN(in_c, out_c, kernel_size, stride)
        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

    def forward(self, x):
        output = self.conv(x)
        return self.activation_function(output)


class ResConvBlock(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(ResConvBlock, self).__init__()
        self.stride = stride
        if block_type == 'k1kx':
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size=1, stride=1)
        else:
            self.conv1 = ConvKXBN(
                in_c, btn_c, kernel_size=kernel_size, stride=1)

        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, kernel_size, stride)
        else:
            self.conv2 = RepConv(
                btn_c, out_c, kernel_size, stride, act='identity')

        self.activation_function = get_activation(act)

        if in_c != out_c and stride != 2:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = None

    def forward(self, x):
        if self.residual_proj is not None:
            reslink = self.residual_proj(x)
        else:
            reslink = x
        x = self.conv1(x)
        x = self.activation_function(x)
        x = self.conv2(x)
        if self.stride != 2:
            x = x + reslink
        x = self.activation_function(x)
        return x


class SuperResStem(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False,
                 block_type='k1kx'):
        super(SuperResStem, self).__init__()
        if act is None:
            self.act = torch.relu
        else:
            self.act = get_activation(act)
        self.block_list = nn.ModuleList()
        for block_id in range(num_blocks):
            if block_id == 0:
                in_channels = in_c
                out_channels = out_c
                this_stride = stride
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                this_kernel_size = kernel_size
            the_block = ResConvBlock(
                in_channels,
                out_channels,
                btn_c,
                this_kernel_size,
                this_stride,
                act=act,
                reparam=reparam,
                block_type=block_type)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class TinyNAS(nn.Module):

    def __init__(self,
                 structure_info=None,
                 out_indices=[2, 4, 5],
                 with_spp=False,
                 use_focus=False,
                 act='silu',
                 reparam=False):
        super(TinyNAS, self).__init__()
        self.out_indices = out_indices
        self.block_list = nn.ModuleList()

        for idx, block_info in enumerate(structure_info):
            the_block_class = block_info['class']
            if the_block_class == 'ConvKXBNRELU':
                if use_focus:
                    the_block = Focus(
                        block_info['in'],
                        block_info['out'],
                        block_info['k'],
                        act=act)
                else:
                    the_block = ConvKXBNRELU(
                        block_info['in'],
                        block_info['out'],
                        block_info['k'],
                        block_info['s'],
                        act=act)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvK1KX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(
                    block_info['in'],
                    block_info['out'],
                    block_info['btn'],
                    block_info['k'],
                    block_info['s'],
                    block_info['L'],
                    spp,
                    act=act,
                    reparam=reparam,
                    block_type='k1kx')
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResStem(
                    block_info['in'],
                    block_info['out'],
                    block_info['btn'],
                    block_info['k'],
                    block_info['s'],
                    block_info['L'],
                    spp,
                    act=act,
                    reparam=reparam,
                    block_type='kxkx')
                self.block_list.append(the_block)
            else:
                raise NotImplementedError

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                stage_feature_list.append(output)
        return stage_feature_list


def load_tinynas_net(backbone_cfg):
    # load masternet model to path
    import ast

    net_structure_str = read_file(backbone_cfg.structure_file)
    struct_str = ''.join([x.strip() for x in net_structure_str])
    struct_info = ast.literal_eval(struct_str)
    for layer in struct_info:
        if 'nbitsA' in layer:
            del layer['nbitsA']
        if 'nbitsW' in layer:
            del layer['nbitsW']

    model = TinyNAS(
        structure_info=struct_info,
        out_indices=backbone_cfg.out_indices,
        with_spp=backbone_cfg.with_spp,
        use_focus=backbone_cfg.use_focus,
        act=backbone_cfg.act,
        reparam=backbone_cfg.reparam)

    return model
