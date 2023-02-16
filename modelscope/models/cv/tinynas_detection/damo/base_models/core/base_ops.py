# Copyright (c) Alibaba, Inc. and its affiliates.
# The DAMO-YOLO implementation is also open-sourced by the authors at https://github.com/tinyvision/damo-yolo.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .repvgg_block import RepVggBlock


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    elif name == 'gn':
        module = nn.GroupNorm(num_channels=out_channels, num_groups=32)
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 groups=1,
                 bias=False,
                 act='silu',
                 norm='bn'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None

    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            # x = self.norm(x)
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DepthWiseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride=1,
                 groups=None,
                 bias=False,
                 act='silu',
                 norm='bn'):
        super().__init__()
        padding = (ksize - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias)
        if norm is not None:
            self.dwnorm = get_norm(norm, in_channels, inplace=True)
            self.pwnorm = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)

        self.with_norm = norm is not None
        self.with_act = act is not None
        self.order = ['depthwise', 'dwnorm', 'pointwise', 'act']

    def forward(self, x):

        for layer_name in self.order:
            layer = self.__getattr__(layer_name)
            if layer is not None:
                x = layer(x)
        return x


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act='silu',
        reparam=False,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        k_conv1 = 3 if reparam else 1
        self.conv1 = BaseConv(
            in_channels, hidden_channels, k_conv1, stride=1, act=act)
        if reparam:
            self.conv2 = RepVggBlock(
                hidden_channels, out_channels, 3, stride=1, act=act)
        else:
            self.conv2 = Conv(
                hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    'Residual layer with `in_channels` inputs.'

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act='lrelu')
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act='lrelu')

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act='silu',
        reparam=False,
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(
            2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                1.0,
                depthwise,
                act=act,
                reparam=reparam) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = BaseConv(
            in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class fast_Focus(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super(Focus, self).__init__()
        self.conv1 = self.focus_conv(w1=1.0)
        self.conv2 = self.focus_conv(w3=1.0)
        self.conv3 = self.focus_conv(w2=1.0)
        self.conv4 = self.focus_conv(w4=1.0)

        self.conv = BaseConv(
            in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        return self.conv(
            torch.cat(
                [self.conv1(x),
                 self.conv2(x),
                 self.conv3(x),
                 self.conv4(x)], 1))

    def focus_conv(self, w1=0.0, w2=0.0, w3=0.0, w4=0.0):
        conv = nn.Conv2d(3, 3, 2, 2, groups=3, bias=False)
        conv.weight = self.init_weights_constant(w1, w2, w3, w4)
        conv.weight.requires_grad = False
        return conv

    def init_weights_constant(self, w1=0.0, w2=0.0, w3=0.0, w4=0.0):
        return nn.Parameter(
            torch.tensor([[[[w1, w2], [w3, w4]]], [[[w1, w2], [w3, w4]]],
                          [[[w1, w2], [w3, w4]]]]))


# shufflenet block
def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


def conv_1x1_bn(in_c, out_c, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(True))


def conv_bn(in_c, out_c, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(True))


class ShuffleBlock(nn.Module):

    def __init__(self, in_c, out_c, downsample=False):
        super(ShuffleBlock, self).__init__()
        self.downsample = downsample
        half_c = out_c // 2
        if downsample:
            self.branch1 = nn.Sequential(
                # 3*3 dw conv, stride = 2
                # nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(in_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 2
                # nn.Conv2d(half_c, half_c, 3, 2, 1, groups=half_c, bias=False),
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))
        else:
            # in_c = out_c
            assert in_c == out_c

            self.branch2 = nn.Sequential(
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True),
                # 3*3 dw conv, stride = 1
                nn.Conv2d(half_c, half_c, 3, 1, 1, groups=half_c, bias=False),
                nn.BatchNorm2d(half_c),
                # 1*1 pw conv
                nn.Conv2d(half_c, half_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_c),
                nn.ReLU(True))

    def forward(self, x):
        out = None
        if self.downsample:
            # if it is downsampling, we don't need to do channel split
            out = torch.cat((self.branch1(x), self.branch2(x)), 1)
        else:
            # channel split
            channels = x.shape[1]
            c = channels // 2
            x1 = x[:, :c, :, :]
            x2 = x[:, c:, :, :]
            out = torch.cat((x1, self.branch2(x2)), 1)
        return channel_shuffle(out, 2)


class ShuffleCSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act='silu',
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                1.0,
                depthwise,
                act=act) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        # add channel shuffle
        return channel_shuffle(x, 2)
