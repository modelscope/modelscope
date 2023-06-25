from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size,
                      int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def count_conv_flop(layer, x):
    out_h = int(x.size(2) / layer.stride[0])
    out_w = int(x.size(3) / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[
        0] * layer.kernel_size[1] * out_h * out_w / layer.groups
    return delta_ops


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None
    name2layer = {
        ZeroLayer.__name__: ZeroLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        IdentityLayer.__name__: IdentityLayer,
    }
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (self.mobile_inverted_conv.module_str,
                             self.shortcut.module_str
                             if self.shortcut is not None else None)

    @property
    def config(self):
        return {
            'name':
            MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv':
            self.mobile_inverted_conv.config,
            'shortcut':
            self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(
            config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, _ = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0
        return flops1 + flops2, self.forward(x)


class MBInvertedConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=(1, 1),
                 expand_ratio=6,
                 mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        feature_dim = round(
            self.in_channels
            * self.expand_ratio) if mid_channels is None else mid_channels
        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict([
                    ('conv',
                     nn.Conv2d(
                         self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(feature_dim)),
                    ('act', nn.PReLU()),
                ]))
        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(
                     feature_dim,
                     feature_dim,
                     kernel_size,
                     stride,
                     pad,
                     groups=feature_dim,
                     bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.PReLU()),
            ]))
        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    @property
    def module_str(self):
        return '%dx%d_MBConv%d' % (self.kernel_size, self.kernel_size,
                                   self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        '''count conv flops, skip BN and other small flops
        '''
        total_flops = 0
        if self.inverted_bottleneck:
            total_flops += count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        total_flops += count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)
        total_flops += count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)
        return total_flops, x


class IdentityLayer(nn.Module):

    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

    @property
    def module_str(self):
        return 'Identity'

    @property
    def config(self):
        return {
            'name': IdentityLayer.__name__,
        }

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        return 0, self.forward(x)


class ZeroLayer(nn.Module):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.shape
        h //= self.stride[0]
        w //= self.stride[1]
        device = x.device
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding

    @property
    def module_str(self):
        return 'Zero'

    @property
    def config(self):
        return {'name': ZeroLayer.__name__, 'stride': self.stride}

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    @staticmethod
    def is_zero_layer():
        return True

    def get_flops(self, x):
        return 0, self.forward(x)


def split_layer(total_channels, num_groups):
    split = [
        int(np.ceil(total_channels / num_groups)) for _ in range(num_groups)
    ]
    split[num_groups - 1] += total_channels - sum(split)
    return split


class MBInvertedMixConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mix_conv_size=[1, 3, 5],
                 stride=(1, 1),
                 expand_ratio=6,
                 mid_channels=None):
        super(MBInvertedMixConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_conv_size = mix_conv_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        feature_dim = round(
            self.in_channels
            * self.expand_ratio) if mid_channels is None else mid_channels
        self.inverted_bottleneck = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0,
                           bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.PReLU()),
            ]))

        self.mix_conv_size = mix_conv_size
        self.n_chunks = len(mix_conv_size)
        self.split_in_channels = split_layer(feature_dim, self.n_chunks)

        self.mix_conv = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = self.mix_conv_size[idx]
            pad = get_same_padding(kernel_size)
            split_in_channels_ = self.split_in_channels[idx]
            self.mix_conv.append(
                nn.Sequential(
                    OrderedDict([
                        ('conv',
                         nn.Conv2d(
                             split_in_channels_,
                             split_in_channels_,
                             kernel_size,
                             stride,
                             pad,
                             groups=split_in_channels_,
                             bias=False)),
                        ('bn', nn.BatchNorm2d(split_in_channels_)),
                        ('act', nn.PReLU()),
                    ])))

        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        x = self.inverted_bottleneck(x)
        split = torch.split(x, self.split_in_channels, dim=1)
        x = torch.cat([layer(s) for layer, s in zip(self.mix_conv, split)],
                      dim=1)

        x = self.point_conv(x)
        return x

    @property
    def module_str(self):
        return '%s_MixConv%d' % (str(self.mix_conv_size), self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedMixConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'mix_conv_size': self.mix_conv_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedMixConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        '''count conv flops, skip BN and other small flops
        '''
        total_flops = 0
        total_flops += count_conv_flop(self.inverted_bottleneck.conv, x)
        x = self.inverted_bottleneck(x)
        split = torch.split(x, self.split_in_channels, dim=1)
        out = []
        for layer, s in zip(self.mix_conv, split):
            out.append(layer(s))
            total_flops += count_conv_flop(layer.conv, s)
        x = torch.cat(out, dim=1)
        total_flops += count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)
        return total_flops, x


class LinearMixConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mix_conv_size=[1, 3, 5],
                 stride=(1, 1),
                 mid_channels=None):
        super(LinearMixConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_conv_size = mix_conv_size
        self.stride = stride
        self.mid_channels = mid_channels

        self.mix_conv_size = mix_conv_size
        self.n_chunks = len(mix_conv_size)
        self.mix_conv = nn.ModuleList()

        for idx in range(self.n_chunks):
            kernel_size = self.mix_conv_size[idx]
            pad = get_same_padding(kernel_size)
            self.mix_conv.append(
                nn.Sequential(
                    OrderedDict([
                        ('conv',
                         nn.Conv2d(
                             in_channels,
                             in_channels,
                             kernel_size,
                             stride,
                             pad,
                             groups=in_channels,
                             bias=False)),
                        ('bn', nn.BatchNorm2d(in_channels)),
                        ('act', nn.ReLU6(inplace=True)),
                    ])))

        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(
                     in_channels * self.n_chunks,
                     out_channels,
                     1,
                     1,
                     0,
                     bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        x = torch.cat([layer(x) for layer in self.mix_conv], dim=1)

        x = self.point_conv(x)
        return x

    @property
    def module_str(self):
        return '%s_LinearMixConv' % (str(self.mix_conv_size))

    @property
    def config(self):
        return {
            'name': LinearMixConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'mix_conv_size': self.mix_conv_size,
            'stride': self.stride,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return LinearMixConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        '''count conv flops, skip BN and other small flops
        '''
        total_flops = 0
        out = []
        for layer in self.mix_conv:
            out.append(layer(x))
            total_flops += count_conv_flop(layer.conv, x)
        x = torch.cat(out, dim=1)
        total_flops += count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)
        return total_flops, x


class SELayer(nn.Module):
    '''
    '''

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super(SELayer, self).__init__()
        self.input_channels = input_channels
        self.squeeze_factor = squeeze_factor
        self.squeeze_channels = input_channels // squeeze_factor
        self.fc1 = nn.Conv2d(self.input_channels, self.squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.squeeze_channels, self.input_channels, 1)

    def _scale(self, input):
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return torch.sigmoid(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input

    @property
    def module_str(self):
        return 'SE_%d' % (self.squeeze_factor)

    @property
    def config(self):
        return {
            'name': SELayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'squeeze_factor': self.squeeze_factor,
        }

    @staticmethod
    def build_from_config(config):
        return SELayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        '''
        count se flops, only compute the fc layers' calculation
        '''
        total_flops = 0
        total_flops += self.input_channels * self.squeeze_channels * 2
        b, c, h, w = x.shape
        total_flops += c * h * w
        return total_flops, x


class MHSA(nn.Module):

    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.width = width
        self.height = height

        self.rel_h = nn.Parameter(
            torch.randn([1, heads, n_dims // heads, 1, height]),
            requires_grad=True)
        self.rel_w = nn.Parameter(
            torch.randn([1, heads, n_dims // heads, width, 1]),
            requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(
            1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

    def get_flops(self, x):
        '''
        count se flops, only compute the fc layers' calculation
        '''
        n_batch, C, width, height = x.size()

        total_flops = 0
        total_flops += count_conv_flop(self.query, x) * 3

        # content_content
        total_flops += (width * height) * C * (width * height)

        # content_position
        total_flops += (width * height) * C
        total_flops += (width * height) * C * (width * height)

        # attention
        total_flops += (width * height) * C * (width * height)

        return total_flops, x


class MBInvertedMHSALayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=6,
                 width=1,
                 height=175,
                 mid_channels=None):
        super(MBInvertedMHSALayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        feature_dim = round(
            self.in_channels
            * self.expand_ratio) if mid_channels is None else mid_channels
        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict([
                    ('conv',
                     nn.Conv2d(
                         self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                    ('bn', nn.BatchNorm2d(feature_dim)),
                    #                 ('act', nn.PReLU()),
                    ('act', nn.ReLU6(inplace=True)),
                ]))
        self.mhsa = MHSA(feature_dim, width, height)
        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.mhsa(x)
        x = self.point_conv(x)
        return x

    @property
    def module_str(self):
        return 'MSHA%d' % (self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedMHSALayer.__name__,
            'in_channels': self.in_channels,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedMHSALayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def get_flops(self, x):
        '''count conv flops, skip BN and other small flops
        '''
        total_flops = 0
        if self.inverted_bottleneck:
            total_flops += count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        total_flops += self.mhsa.get_flops(x)[0]
        x = self.mhsa(x)
        total_flops += count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)
        return total_flops, x


class MBInvertedRepConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 rep_conv_size=[1, 3, 5],
                 stride=(1, 1),
                 expand_ratio=6,
                 mid_channels=None):
        super(MBInvertedRepConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_conv_size = rep_conv_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        feature_dim = round(
            self.in_channels
            * self.expand_ratio) if mid_channels is None else mid_channels

        self.inverted_bottleneck = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0,
                           bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.PReLU()),
            ]))

        self.rep_conv_size = rep_conv_size
        self.n_chunks = len(rep_conv_size)

        self.rep_conv = nn.ModuleList()
        for idx in range(self.n_chunks):
            kernel_size = self.rep_conv_size[idx]
            pad = get_same_padding(kernel_size)
            self.rep_conv.append(
                nn.Sequential(
                    OrderedDict([
                        ('conv',
                         nn.Conv2d(
                             feature_dim,
                             feature_dim,
                             kernel_size,
                             stride,
                             pad,
                             groups=feature_dim,
                             bias=False)),
                        ('bn', nn.BatchNorm2d(feature_dim)),
                    ])))

        self.rep_conv_deploy = None
        self.act = nn.PReLU()
        self.point_conv = nn.Sequential(
            OrderedDict([
                ('conv',
                 nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels)),
            ]))

        self.deploy = False

    def forward(self, x):
        x = self.inverted_bottleneck(x)
        if not self.deploy:
            out = []
            for layer in self.rep_conv:
                out.append(layer(x))
            x = out[0]
            for out_ in out[1:]:
                x += out_
        else:
            x = self.rep_conv_deploy(x)
        x = self.act(x)
        x = self.point_conv(x)
        return x

    @property
    def module_str(self):
        return '%s_RepConv%d' % (str(self.rep_conv_size), self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedMixConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'rep_conv_size': self.rep_conv_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels,
        }

    @staticmethod
    def build_from_config(config):
        return MBInvertedMixConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False

    def switch_to_deploy(self):
        self.deploy = True
        feature_dim = self.rep_conv[0].conv.in_channels
        stride = self.rep_conv[0].conv.stride
        kernel_size = max(self.rep_conv_size)
        pad = get_same_padding(kernel_size)
        self.rep_conv_deploy = nn.Conv2d(
            feature_dim,
            feature_dim,
            kernel_size,
            stride,
            pad,
            groups=feature_dim,
            bias=True)

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rep_conv_deploy.weight.data = kernel
        self.rep_conv_deploy.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rep_conv')

    def get_equivalent_kernel_bias(self):
        max_kernel_size = max(self.rep_conv_size)

        if max_kernel_size == 5:
            if 1 in self.rep_conv_size:
                kernel1x1, bias1x1 = self._fuse_bn_tensor(
                    self.rep_conv[self.rep_conv_size.index(1)])
            else:
                kernel1x1 = None
                bias1x1 = 0

            if 3 in self.rep_conv_size:
                kernel3x3, bias3x3 = self._fuse_bn_tensor(
                    self.rep_conv[self.rep_conv_size.index(3)])
            else:
                kernel3x3 = None
                bias3x3 = 0

            if 5 in self.rep_conv_size:
                kernel5x5, bias5x5 = self._fuse_bn_tensor(
                    self.rep_conv[self.rep_conv_size.index(5)])

            else:
                kernel5x5 = 0
                bias5x5 = 0

            return kernel5x5 + self._pad_1x1_to_5x5_tensor(
                kernel1x1) + self._pad_3x3_to_5x5_tensor(
                    kernel3x3), bias5x5 + bias3x3 + bias1x1
        else:
            if 1 in self.rep_conv_size:
                kernel1x1, bias1x1 = self._fuse_bn_tensor(
                    self.rep_conv[self.rep_conv_size.index(1)])
            else:
                kernel1x1 = None
                bias1x1 = 0

            if 3 in self.rep_conv_size:
                kernel3x3, bias3x3 = self._fuse_bn_tensor(
                    self.rep_conv[self.rep_conv_size.index(3)])
            else:
                kernel3x3 = None
                bias3x3 = 0

            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _pad_1x1_to_5x5_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [2, 2, 2, 2])

    def _pad_3x3_to_5x5_tensor(self, kernel3x3):
        if kernel3x3 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel3x3, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
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

    def get_flops(self, x):
        '''count conv flops, skip BN and other small flops
        '''
        total_flops = 0
        total_flops += count_conv_flop(self.inverted_bottleneck.conv, x)
        x = self.inverted_bottleneck(x)

        total_flops += count_conv_flop(self.rep_conv[-1].conv, x)
        out = []
        for layer in self.rep_conv:
            out.append(layer(x))
        x = out[0]
        for out_ in out[1:]:
            x += out_
        total_flops += count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)
        return total_flops, x
