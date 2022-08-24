# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

import uuid

from torch import nn

from . import basic_blocks, global_utils
from .global_utils import get_right_parentheses_index
from .super_blocks import PlainNetSuperBlockClass


class SuperResIDWEXKX(PlainNetSuperBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 kernel_size=None,
                 expension=None,
                 no_create=False,
                 no_reslink=False,
                 no_BN=False,
                 use_se=False,
                 **kwargs):
        super(SuperResIDWEXKX, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.expension = expension
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN

        self.use_se = use_se

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for i in range(self.sub_layers):
            inner_str = ''
            dw_channels = global_utils.smart_round(
                self.bottleneck_channels * self.expension, base=8)
            inner_str += 'ConvKX({},{},{},{})'.format(last_channels,
                                                      dw_channels, 1, 1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(dw_channels)
            inner_str += 'RELU({})'.format(dw_channels)

            inner_str += 'ConvDW({},{},{})'.format(dw_channels,
                                                   self.kernel_size,
                                                   current_stride)
            if not self.no_BN:
                inner_str += 'BN({})'.format(dw_channels)
            inner_str += 'RELU({})'.format(dw_channels)
            if self.use_se:
                inner_str += 'SE({})'.format(dw_channels)

            inner_str += 'ConvKX({},{},{},{})'.format(dw_channels,
                                                      bottleneck_channels, 1,
                                                      1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(bottleneck_channels)

            if not self.no_reslink:
                if i == 0:
                    res_str = 'ResBlockProj({})RELU({})'.format(
                        inner_str, self.out_channels)
                else:
                    res_str = 'ResBlock({})RELU({})'.format(
                        inner_str, self.out_channels)

            else:
                res_str = '{}RELU({})'.format(inner_str, self.out_channels)

            full_str += res_str

            inner_str = ''
            dw_channels = global_utils.smart_round(
                self.out_channels * self.expension, base=8)
            inner_str += 'ConvKX({},{},{},{})'.format(bottleneck_channels,
                                                      dw_channels, 1, 1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(dw_channels)
            inner_str += 'RELU({})'.format(dw_channels)

            inner_str += 'ConvDW({},{},{})'.format(dw_channels,
                                                   self.kernel_size, 1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(dw_channels)
            inner_str += 'RELU({})'.format(dw_channels)
            if self.use_se:
                inner_str += 'SE({})'.format(dw_channels)

            inner_str += 'ConvKX({},{},{},{})'.format(dw_channels,
                                                      self.out_channels, 1, 1)
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.out_channels)

            if not self.no_reslink:
                res_str = 'ResBlock({})RELU({})'.format(
                    inner_str, self.out_channels)
            else:
                res_str = '{}RELU({})'.format(inner_str, self.out_channels)

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


class SuperResIDWE1K3(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE1K3, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=3,
            expension=1.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE2K3(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE2K3, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=3,
            expension=2.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE4K3(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE4K3, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=3,
            expension=4.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE6K3(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE6K3, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=3,
            expension=6.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE1K5(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE1K5, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=5,
            expension=1.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE2K5(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE2K5, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=5,
            expension=2.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE4K5(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE4K5, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=5,
            expension=4.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE6K5(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE6K5, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=5,
            expension=6.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE1K7(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE1K7, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=7,
            expension=1.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE2K7(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE2K7, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=7,
            expension=2.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE4K7(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE4K7, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=7,
            expension=4.0,
            no_create=no_create,
            **kwargs)


class SuperResIDWE6K7(SuperResIDWEXKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=None,
                 bottleneck_channels=None,
                 sub_layers=None,
                 no_create=False,
                 **kwargs):
        super(SuperResIDWE6K7, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_channels=bottleneck_channels,
            sub_layers=sub_layers,
            kernel_size=7,
            expension=6.0,
            no_create=no_create,
            **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'SuperResIDWE1K3': SuperResIDWE1K3,
        'SuperResIDWE2K3': SuperResIDWE2K3,
        'SuperResIDWE4K3': SuperResIDWE4K3,
        'SuperResIDWE6K3': SuperResIDWE6K3,
        'SuperResIDWE1K5': SuperResIDWE1K5,
        'SuperResIDWE2K5': SuperResIDWE2K5,
        'SuperResIDWE4K5': SuperResIDWE4K5,
        'SuperResIDWE6K5': SuperResIDWE6K5,
        'SuperResIDWE1K7': SuperResIDWE1K7,
        'SuperResIDWE2K7': SuperResIDWE2K7,
        'SuperResIDWE4K7': SuperResIDWE4K7,
        'SuperResIDWE6K7': SuperResIDWE6K7,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict
