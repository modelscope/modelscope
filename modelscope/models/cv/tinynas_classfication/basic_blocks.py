# Copyright (c) Alibaba, Inc. and its affiliates.
# The ZenNAS implementation is also open-sourced by the authors, and available at https://github.com/idstcv/ZenNAS.

import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .global_utils import (create_netblock_list_from_str_inner,
                           get_right_parentheses_index)


class PlainNetBasicBlockClass(nn.Module):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 stride=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(PlainNetBasicBlockClass, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name
        if self.block_name is None:
            self.block_name = 'uuid{}'.format(uuid.uuid4().hex)

    def forward(self, x):
        raise RuntimeError('Not implemented')

    def __str__(self):
        return type(self).__name__ + '({},{},{})'.format(
            self.in_channels, self.out_channels, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{},{})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride)

    def get_output_resolution(self, input_resolution):
        raise RuntimeError('Not implemented')

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert PlainNetBasicBlockClass.is_instance_from_str(s)
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
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @classmethod
    def is_instance_from_str(cls, s):
        if s.startswith(cls.__name__ + '(') and s[-1] == ')':
            return True
        else:
            return False


class AdaptiveAvgPool(PlainNetBasicBlockClass):

    def __init__(self, out_channels, output_size, no_create=False, **kwargs):
        super(AdaptiveAvgPool, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.AdaptiveAvgPool2d(
                output_size=(self.output_size, self.output_size))

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return type(self).__name__ + '({},{})'.format(
            self.out_channels // self.output_size**2, self.output_size)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{})'.format(
            self.block_name, self.out_channels // self.output_size**2,
            self.output_size)

    def get_output_resolution(self, input_resolution):
        return self.output_size

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert AdaptiveAvgPool.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('AdaptiveAvgPool('):idx]

        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        output_size = int(param_str_split[1])
        return AdaptiveAvgPool(
            out_channels=out_channels,
            output_size=output_size,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]


class BN(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(BN, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.BatchNorm2d)
            self.in_channels = copy_from.weight.shape[0]
            self.out_channels = copy_from.weight.shape[0]
            assert out_channels is None or out_channels == self.out_channels
            self.netblock = copy_from

        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            if no_create:
                return
            else:
                self.netblock = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'BN({})'.format(self.out_channels)

    def __repr__(self):
        return 'BN({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert BN.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('BN('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        out_channels = int(param_str)
        return BN(
            out_channels=out_channels,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]


class ConvKX(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 groups=1,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKX, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            self.groups = copy_from.groups
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride
            self.netblock = copy_from
        else:
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.groups = groups
            self.kernel_size = kernel_size
            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=False,
                    groups=self.groups)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return type(self).__name__ + '({},{},{},{})'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{},{},{})'.format(
            self.block_name, self.in_channels, self.out_channels,
            self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
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

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        kernel_size = int(split_str[2])
        stride = int(split_str[3])
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class ConvDW(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvDW, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            assert self.in_channels == self.out_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride

            self.netblock = copy_from
        else:

            self.in_channels = out_channels
            self.out_channels = out_channels
            self.stride = stride
            self.kernel_size = kernel_size

            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=False,
                    groups=self.in_channels)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'ConvDW({},{},{})'.format(self.out_channels, self.kernel_size,
                                         self.stride)

    def __repr__(self):
        return 'ConvDW({}|{},{},{})'.format(self.block_name, self.out_channels,
                                            self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ConvDW.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('ConvDW('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        out_channels = int(split_str[0])
        kernel_size = int(split_str[1])
        stride = int(split_str[2])
        return ConvDW(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class ConvKXG2(ConvKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKXG2, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            copy_from=copy_from,
            no_create=no_create,
            groups=2,
            **kwargs)


class ConvKXG4(ConvKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKXG4, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            copy_from=copy_from,
            no_create=no_create,
            groups=4,
            **kwargs)


class ConvKXG8(ConvKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKXG8, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            copy_from=copy_from,
            no_create=no_create,
            groups=8,
            **kwargs)


class ConvKXG16(ConvKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKXG16, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            copy_from=copy_from,
            no_create=no_create,
            groups=16,
            **kwargs)


class ConvKXG32(ConvKX):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(ConvKXG32, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            copy_from=copy_from,
            no_create=no_create,
            groups=32,
            **kwargs)


class Flatten(PlainNetBasicBlockClass):

    def __init__(self, out_channels, no_create=False, **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, x):
        return torch.flatten(x, 1)

    def __str__(self):
        return 'Flatten({})'.format(self.out_channels)

    def __repr__(self):
        return 'Flatten({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return 1

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Flatten.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('Flatten('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Flatten(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class Linear(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 bias=True,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            assert isinstance(copy_from, nn.Linear)
            self.in_channels = copy_from.weight.shape[1]
            self.out_channels = copy_from.weight.shape[0]
            self.use_bias = copy_from.bias is not None
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels

            self.netblock = copy_from
        else:

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.use_bias = bias
            if not no_create:
                self.netblock = nn.Linear(
                    self.in_channels, self.out_channels, bias=self.use_bias)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'Linear({},{},{})'.format(self.in_channels, self.out_channels,
                                         int(self.use_bias))

    def __repr__(self):
        return 'Linear({}|{},{},{})'.format(self.block_name, self.in_channels,
                                            self.out_channels,
                                            int(self.use_bias))

    def get_output_resolution(self, input_resolution):
        assert input_resolution == 1
        return 1

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Linear.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('Linear('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        use_bias = int(split_str[2])

        return Linear(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=use_bias == 1,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]


class MaxPool(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride,
                 no_create=False,
                 **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.MaxPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding)

    def forward(self, x):
        return self.netblock(x)

    def __str__(self):
        return 'MaxPool({},{},{})'.format(self.out_channels, self.kernel_size,
                                          self.stride)

    def __repr__(self):
        return 'MaxPool({}|{},{},{})'.format(self.block_name,
                                             self.out_channels,
                                             self.kernel_size, self.stride)

    def get_output_resolution(self, input_resolution):
        return input_resolution // self.stride

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MaxPool.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('MaxPool('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        kernel_size = int(param_str_split[1])
        stride = int(param_str_split[2])
        return MaxPool(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class Sequential(PlainNetBasicBlockClass):

    def __init__(self, block_list, no_create=False, **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = block_list[0].in_channels
        self.out_channels = block_list[-1].out_channels
        self.no_create = no_create
        res = 1024
        for block in self.block_list:
            res = block.get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output = x
        for inner_block in self.block_list:
            output = inner_block(output)
        return output

    def __str__(self):
        s = 'Sequential('
        for inner_block in self.block_list:
            s += str(inner_block)
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)
        return the_res

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Sequential.is_instance_from_str(s)
        the_right_paraen_idx = get_right_parentheses_index(s)
        param_str = s[len('Sequential(') + 1:the_right_paraen_idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_block_list, remaining_s = create_netblock_list_from_str_inner(
            param_str, netblocks_dict=bottom_basic_dict, no_create=no_create)
        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, ''
        return Sequential(
            block_list=the_block_list,
            no_create=no_create,
            block_name=tmp_block_name), ''


class MultiSumBlock(PlainNetBasicBlockClass):

    def __init__(self, block_list, no_create=False, **kwargs):
        super(MultiSumBlock, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.max([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output = self.block_list[0](x)
        for inner_block in self.block_list[1:]:
            output2 = inner_block(x)
            output = output + output2
        return output

    def __str__(self):
        s = 'MultiSumBlock({}|'.format(self.block_name)
        for inner_block in self.block_list:
            s += str(inner_block) + ';'
        s = s[:-1]
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MultiSumBlock.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('MultiSumBlock('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = create_netblock_list_from_str_inner(
                the_s, netblocks_dict=bottom_basic_dict, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(
                    Sequential(block_list=tmp_block_list, no_create=no_create))
        pass

        if len(the_block_list) == 0:
            return None, s[idx + 1:]

        return MultiSumBlock(
            block_list=the_block_list,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]


class MultiCatBlock(PlainNetBasicBlockClass):

    def __init__(self, block_list, no_create=False, **kwargs):
        super(MultiCatBlock, self).__init__(**kwargs)
        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)
        self.in_channels = np.max([x.in_channels for x in block_list])
        self.out_channels = np.sum([x.out_channels for x in block_list])
        self.no_create = no_create

        res = 1024
        res = self.block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output_list = []
        for inner_block in self.block_list:
            output = inner_block(x)
            output_list.append(output)

        return torch.cat(output_list, dim=1)

    def __str__(self):
        s = 'MultiCatBlock({}|'.format(self.block_name)
        for inner_block in self.block_list:
            s += str(inner_block) + ';'

        s = s[:-1]
        s += ')'
        return s

    def __repr__(self):
        return str(self)

    def get_output_resolution(self, input_resolution):
        the_res = self.block_list[0].get_output_resolution(input_resolution)
        for the_block in self.block_list:
            assert the_res == the_block.get_output_resolution(input_resolution)

        return the_res

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert MultiCatBlock.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('MultiCatBlock('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = create_netblock_list_from_str_inner(
                the_s, netblocks_dict=bottom_basic_dict, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_block_list.append(tmp_block_list[0])
            else:
                the_block_list.append(
                    Sequential(block_list=tmp_block_list, no_create=no_create))

        if len(the_block_list) == 0:
            return None, s[idx + 1:]

        return MultiCatBlock(
            block_list=the_block_list,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]


class RELU(PlainNetBasicBlockClass):

    def __init__(self, out_channels, no_create=False, **kwargs):
        super(RELU, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.no_create = no_create

    def forward(self, x):
        return F.relu(x)

    def __str__(self):
        return 'RELU({})'.format(self.out_channels)

    def __repr__(self):
        return 'RELU({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert RELU.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('RELU('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return RELU(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class ResBlock(PlainNetBasicBlockClass):
    """
    ResBlock(in_channles, inner_blocks_str). If in_channels is missing, use block_list[0].in_channels as in_channels
    """

    def __init__(self,
                 block_list,
                 in_channels=None,
                 stride=None,
                 no_create=False,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

        self.proj = None
        if self.stride > 1 or self.in_channels != self.out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                nn.BatchNorm2d(self.out_channels),
            )

    def forward(self, x):
        if len(self.block_list) == 0:
            return x

        output = x
        for inner_block in self.block_list:
            output = inner_block(output)

        if self.proj is not None:
            output = output + self.proj(x)
        else:
            output = output + x

        return output

    def __str__(self):
        s = 'ResBlock({},{},'.format(self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def __repr__(self):
        s = 'ResBlock({}|{},{},'.format(self.block_name, self.in_channels,
                                        self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ResBlock.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        the_stride = None
        param_str = s[len('ResBlock('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit(
        ):
            in_channels = None
            the_block_list, remaining_s = create_netblock_list_from_str_inner(
                param_str,
                netblocks_dict=bottom_basic_dict,
                no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[
                    0:second_comma_index].isdigit():
                the_block_list, remaining_s = create_netblock_list_from_str_inner(
                    param_str,
                    netblocks_dict=bottom_basic_dict,
                    no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = create_netblock_list_from_str_inner(
                    param_str,
                    netblocks_dict=bottom_basic_dict,
                    no_create=no_create)
            pass
        pass

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, s[idx + 1:]
        return ResBlock(
            block_list=the_block_list,
            in_channels=in_channels,
            stride=the_stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class ResBlockProj(PlainNetBasicBlockClass):
    """
    ResBlockProj(in_channles, inner_blocks_str). If in_channels is missing, use block_list[0].in_channels as in_channels
    """

    def __init__(self,
                 block_list,
                 in_channels=None,
                 stride=None,
                 no_create=False,
                 **kwargs):
        super(ResBlockProj, self).__init__(**kwargs)
        self.block_list = block_list
        self.stride = stride
        self.no_create = no_create
        if not no_create:
            self.module_list = nn.ModuleList(block_list)

        if in_channels is None:
            self.in_channels = block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = block_list[-1].out_channels

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

        self.proj = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        if len(self.block_list) == 0:
            return x

        output = x
        for inner_block in self.block_list:
            output = inner_block(output)
        output = output + self.proj(x)
        return output

    def __str__(self):
        s = 'ResBlockProj({},{},'.format(self.in_channels, self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def __repr__(self):
        s = 'ResBlockProj({}|{},{},'.format(self.block_name, self.in_channels,
                                            self.stride)
        for inner_block in self.block_list:
            s += str(inner_block)

        s += ')'
        return s

    def get_output_resolution(self, input_resolution):
        the_res = input_resolution
        for the_block in self.block_list:
            the_res = the_block.get_output_resolution(the_res)

        return the_res

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert ResBlockProj.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        the_stride = None
        param_str = s[len('ResBlockProj('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit(
        ):
            in_channels = None
            the_block_list, remaining_s = create_netblock_list_from_str_inner(
                param_str,
                netblocks_dict=bottom_basic_dict,
                no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[
                    0:second_comma_index].isdigit():
                the_block_list, remaining_s = create_netblock_list_from_str_inner(
                    param_str,
                    netblocks_dict=bottom_basic_dict,
                    no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_block_list, remaining_s = create_netblock_list_from_str_inner(
                    param_str,
                    netblocks_dict=bottom_basic_dict,
                    no_create=no_create)
            pass
        pass

        assert len(remaining_s) == 0
        if the_block_list is None or len(the_block_list) == 0:
            return None, s[idx + 1:]
        return ResBlockProj(
            block_list=the_block_list,
            in_channels=in_channels,
            stride=the_stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class SE(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(SE, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            self.se_ratio = 0.25
            self.se_channels = max(
                1, int(round(self.out_channels * self.se_ratio)))
            if no_create or self.out_channels == 0:
                return
            else:
                self.netblock = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(
                        in_channels=self.out_channels,
                        out_channels=self.se_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False), nn.BatchNorm2d(self.se_channels),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=self.se_channels,
                        out_channels=self.out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False), nn.BatchNorm2d(self.out_channels),
                    nn.Sigmoid())

    def forward(self, x):
        se_x = self.netblock(x)
        return se_x * x

    def __str__(self):
        return 'SE({})'.format(self.out_channels)

    def __repr__(self):
        return 'SE({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert SE.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('SE('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return SE(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 copy_from=None,
                 no_create=False,
                 **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.no_create = no_create

        if copy_from is not None:
            raise RuntimeError('Not implemented')
        else:
            self.in_channels = out_channels
            self.out_channels = out_channels

    def forward(self, x):
        return SwishImplementation.apply(x)

    def __str__(self):
        return 'Swish({})'.format(self.out_channels)

    def __repr__(self):
        return 'Swish({}|{})'.format(self.block_name, self.out_channels)

    def get_output_resolution(self, input_resolution):
        return input_resolution

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert Swish.is_instance_from_str(s)
        idx = get_right_parentheses_index(s)
        assert idx is not None
        param_str = s[len('Swish('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Swish(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]


bottom_basic_dict = {
    'AdaptiveAvgPool': AdaptiveAvgPool,
    'BN': BN,
    'ConvDW': ConvDW,
    'ConvKX': ConvKX,
    'ConvKXG2': ConvKXG2,
    'ConvKXG4': ConvKXG4,
    'ConvKXG8': ConvKXG8,
    'ConvKXG16': ConvKXG16,
    'ConvKXG32': ConvKXG32,
    'Flatten': Flatten,
    'Linear': Linear,
    'MaxPool': MaxPool,
    'PlainNetBasicBlockClass': PlainNetBasicBlockClass,
    'RELU': RELU,
    'SE': SE,
    'Swish': Swish,
}


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'MultiSumBlock': MultiSumBlock,
        'MultiCatBlock': MultiCatBlock,
        'ResBlock': ResBlock,
        'ResBlockProj': ResBlockProj,
        'Sequential': Sequential,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    netblocks_dict.update(bottom_basic_dict)
    return netblocks_dict
