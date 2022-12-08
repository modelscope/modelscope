# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()

    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))


class ConvBNLayer(nn.Module):

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class RepVGGBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        self.in_channels = ch_in
        self.groups = 1
        if self.deploy is False:
            self.rbr_dense = ConvBNLayer(
                ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.rbr_1x1 = ConvBNLayer(
                ch_in, ch_out, 1, stride=1, padding=0, act=None)
            # self.rbr_identity = nn.BatchNorm2d(num_features=ch_in) if ch_out == ch_in else None
            self.rbr_identity = None
        else:
            self.rbr_reparam = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        self.act = get_activation(act) if act is None or isinstance(
            act, (str, dict)) else act

    def forward(self, x):
        if self.deploy:
            print('----------deploy----------')
            y = self.rbr_reparam(x)
        else:
            if self.rbr_identity is None:
                y = self.rbr_dense(x) + self.rbr_1x1(x)
            else:
                y = self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

        y = self.act(y)
        return y

    def switch_to_deploy(self):
        print('switch')
        if not hasattr(self, 'rbr_reparam'):
            # return
            self.rbr_reparam = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        print('switch')
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        # self.__delattr__(self.rbr_dense)
        # self.__delattr__(self.rbr_1x1)
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        # if isinstance(branch, nn.Sequential):
        if isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        # self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        # self.conv1 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=act)
        self.conv2 = RepVGGBlock(ch_in, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        # y = self.conv1(x)
        y = self.conv2(x)
        if self.shortcut:
            return x + y
        else:
            return y


class BasicBlock_3x3(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock_3x3, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=act)
        # self.conv1 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=act)
        self.conv2 = RepVGGBlock(ch_in, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class BasicBlock_3x3_Reverse(nn.Module):

    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=act)
        # self.conv1 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=act)
        self.conv2 = RepVGGBlock(ch_in, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPP(nn.Module):

    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(
                kernel_size=size, stride=1, padding=size // 2, ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y


class CSPStage(nn.Module):

    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        # self.conv2 = ConvBNLayer(ch_in, ch_mid, 3, stride=1, padding=1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_module(
                    str(i),
                    BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            elif block_fn == 'BasicBlock_3x3':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3(next_ch_in, ch_mid, act=act, shortcut=True))
            elif block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(
                        next_ch_in, ch_mid, act=act, shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        # self.convs = nn.Sequential(*convs)
        self.conv3 = ConvBNLayer(ch_mid * (n + 1), ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)

        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
