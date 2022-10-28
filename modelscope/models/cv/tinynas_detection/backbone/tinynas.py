# Copyright (c) Alibaba, Inc. and its affiliates.
# The AIRDet implementation is also open-sourced by the authors, and available at https://github.com/tinyvision/AIRDet.

import torch
import torch.nn as nn

from modelscope.utils.file_utils import read_file
from ..core.base_ops import Focus, SPPBottleneck, get_activation
from ..core.repvgg_block import RepVggBlock


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


class ResConvK1KX(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 force_resproj=False,
                 act='silu',
                 reparam=False):
        super(ResConvK1KX, self).__init__()
        self.stride = stride
        self.conv1 = ConvKXBN(in_c, btn_c, 1, 1)
        if not reparam:
            self.conv2 = ConvKXBN(btn_c, out_c, 3, stride)
        else:
            self.conv2 = RepVggBlock(
                btn_c, out_c, kernel_size, stride, act='identity')

        if act is None:
            self.activation_function = torch.relu
        else:
            self.activation_function = get_activation(act)

        if stride == 2:
            self.residual_downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.residual_downsample = nn.Identity()

        if in_c != out_c or force_resproj:
            self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        if self.stride != 2:
            reslink = self.residual_downsample(x)
            reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        output = self.activation_function(output)
        output = self.conv2(output)
        if self.stride != 2:
            output = output + reslink
        output = self.activation_function(output)

        return output


class SuperResConvK1KX(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu',
                 reparam=False):
        super(SuperResConvK1KX, self).__init__()
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
                force_resproj = False  # as a part of CSPLayer, DO NOT need this flag
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                force_resproj = False
                this_kernel_size = kernel_size
            the_block = ResConvK1KX(
                in_channels,
                out_channels,
                btn_c,
                this_kernel_size,
                this_stride,
                force_resproj,
                act=act,
                reparam=reparam)
            self.block_list.append(the_block)
            if block_id == 0 and with_spp:
                self.block_list.append(
                    SPPBottleneck(out_channels, out_channels))

    def forward(self, x):
        output = x
        for block in self.block_list:
            output = block(output)
        return output


class ResConvKXKX(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 force_resproj=False,
                 act='silu'):
        super(ResConvKXKX, self).__init__()
        self.stride = stride
        if self.stride == 2:
            self.downsampler = ConvKXBNRELU(in_c, out_c, 3, 2, act=act)
        else:
            self.conv1 = ConvKXBN(in_c, btn_c, kernel_size, 1)
            self.conv2 = RepVggBlock(
                btn_c, out_c, kernel_size, stride, act='identity')

            if act is None:
                self.activation_function = torch.relu
            else:
                self.activation_function = get_activation(act)

            if stride == 2:
                self.residual_downsample = nn.AvgPool2d(
                    kernel_size=2, stride=2)
            else:
                self.residual_downsample = nn.Identity()

            if in_c != out_c or force_resproj:
                self.residual_proj = ConvKXBN(in_c, out_c, 1, 1)
            else:
                self.residual_proj = nn.Identity()

    def forward(self, x):
        if self.stride == 2:
            return self.downsampler(x)
        reslink = self.residual_downsample(x)
        reslink = self.residual_proj(reslink)

        output = x
        output = self.conv1(output)
        output = self.activation_function(output)
        output = self.conv2(output)

        output = output + reslink
        output = self.activation_function(output)

        return output


class SuperResConvKXKX(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 btn_c,
                 kernel_size,
                 stride,
                 num_blocks,
                 with_spp=False,
                 act='silu'):
        super(SuperResConvKXKX, self).__init__()
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
                force_resproj = False  # as a part of CSPLayer, DO NOT need this flag
                this_kernel_size = kernel_size
            else:
                in_channels = out_c
                out_channels = out_c
                this_stride = 1
                force_resproj = False
                this_kernel_size = kernel_size
            the_block = ResConvKXKX(
                in_channels,
                out_channels,
                btn_c,
                this_kernel_size,
                this_stride,
                force_resproj,
                act=act)
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
                 out_indices=[0, 1, 2, 4, 5],
                 out_channels=[None, None, 128, 256, 512],
                 with_spp=False,
                 use_focus=False,
                 need_conv1=True,
                 act='silu',
                 reparam=False):
        super(TinyNAS, self).__init__()
        assert len(out_indices) == len(out_channels)
        self.out_indices = out_indices
        self.need_conv1 = need_conv1

        self.block_list = nn.ModuleList()
        if need_conv1:
            self.conv1_list = nn.ModuleList()
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
                the_block = SuperResConvK1KX(
                    block_info['in'],
                    block_info['out'],
                    block_info['btn'],
                    block_info['k'],
                    block_info['s'],
                    block_info['L'],
                    spp,
                    act=act,
                    reparam=reparam)
                self.block_list.append(the_block)
            elif the_block_class == 'SuperResConvKXKX':
                spp = with_spp if idx == len(structure_info) - 1 else False
                the_block = SuperResConvKXKX(
                    block_info['in'],
                    block_info['out'],
                    block_info['btn'],
                    block_info['k'],
                    block_info['s'],
                    block_info['L'],
                    spp,
                    act=act)
                self.block_list.append(the_block)
            if need_conv1:
                if idx in self.out_indices and out_channels[
                        self.out_indices.index(idx)] is not None:
                    self.conv1_list.append(
                        nn.Conv2d(block_info['out'],
                                  out_channels[self.out_indices.index(idx)],
                                  1))
                else:
                    self.conv1_list.append(None)

    def init_weights(self, pretrain=None):
        pass

    def forward(self, x):
        output = x
        stage_feature_list = []
        for idx, block in enumerate(self.block_list):
            output = block(output)
            if idx in self.out_indices:
                if self.need_conv1 and self.conv1_list[idx] is not None:
                    true_out = self.conv1_list[idx](output)
                    stage_feature_list.append(true_out)
                else:
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
        out_channels=backbone_cfg.out_channels,
        with_spp=backbone_cfg.with_spp,
        use_focus=backbone_cfg.use_focus,
        act=backbone_cfg.act,
        need_conv1=backbone_cfg.need_conv1,
        reparam=backbone_cfg.reparam)

    return model
