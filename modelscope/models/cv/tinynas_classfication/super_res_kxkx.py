# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

import uuid

from torch import nn

from . import basic_blocks, global_utils
from .global_utils import get_right_parentheses_index
from .super_blocks import PlainNetSuperBlockClass


class SuperResKXKX(PlainNetSuperBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 kernel_size=None,
                 no_create=False,
                 no_reslink=False,
                 no_BN=False,
                 use_se=False,
                 **kwargs):
        super(SuperResKXKX, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for i in range(self.sub_layers):
            inner_str = ''

            inner_str += 'ConvKX({},{},{},{})'.format(last_channels,
                                                      self.bottleneck_channels,
                                                      self.kernel_size,
                                                      current_stride)
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.bottleneck_channels)
            inner_str += 'RELU({})'.format(self.bottleneck_channels)
            if self.use_se:
                inner_str += 'SE({})'.format(bottleneck_channels)

            inner_str += 'ConvKX({},{},{},{})'.format(self.bottleneck_channels,
                                                      self.out_channels,
                                                      self.kernel_size, 1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.out_channels)

            if not self.no_reslink:
                if i == 0:
                    res_str = 'ResBlockProj({})RELU({})'.format(
                        inner_str, out_channels)
                else:
                    res_str = 'ResBlock({})RELU({})'.format(
                        inner_str, out_channels)
            else:
                res_str = '{}RELU({})'.format(inner_str, out_channels)

            full_str += res_str

            last_channels = out_channels
            current_stride = 1
        pass

        netblocks_dict = basic_blocks.register_netblocks_dict({})
        self.block_list = global_utils.create_netblock_list_from_str(
            full_str,
            netblocks_dict=netblocks_dict,
            no_create=no_create,
            no_reslink=no_reslink,
            no_BN=no_BN,
            **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + '({},{},{},{},{})'.format(
            self.in_channels, self.out_channels, self.stride,
            self.bottleneck_channels, self.sub_layers)

    def __repr__(self):
        return type(
            self
        ).__name__ + '({}|in={},out={},stride={},btl_channels={},sub_layers={},kernel_size={})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride,
            self.bottleneck_channels, self.sub_layers, self.kernel_size)

    @classmethod
    def create_from_str(cls, s, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        bottleneck_channels = int(param_str_split[3])
        sub_layers = int(param_str_split[4])
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            block_name=tmp_block_name,
            **kwargs), s[idx + 1:]


class SuperResK3K3(SuperResKXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResK3K3, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=3,
            no_create=no_create,
            **kwargs)


class SuperResK5K5(SuperResKXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResK5K5, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=5,
            no_create=no_create,
            **kwargs)


class SuperResK7K7(SuperResKXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResK7K7, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=7,
            no_create=no_create,
            **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'SuperResK3K3': SuperResK3K3,
        'SuperResK5K5': SuperResK5K5,
        'SuperResK7K7': SuperResK7K7,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
