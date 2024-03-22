from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_function import EqualConv2d, EqualLinear


def TVLoss(x):
    tv_h = x[:, :, 1:, :] - x[:, :, :-1, :]
    tv_w = x[:, :, :, 1:] - x[:, :, :, :-1]

    return torch.mean(torch.abs(tv_h)) + torch.mean(torch.abs(tv_w))


class MaskStyle(nn.Module):

    def __init__(self, channels, log_size, style_in, channels_multiplier=2):
        super().__init__()
        self.log_size = log_size
        padding_type = 'zero'
        actvn = 'lrelu'
        normalize_mlp = False
        modulated_conv = True

        self.netM = nn.ModuleDict()

        for i in range(4, self.log_size + 1):
            out_channel = channels[2**i]

            style_mask = StyledConvBlock(
                channels_multiplier * out_channel,
                channels_multiplier * out_channel,
                latent_dim=style_in,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv)

            scale = str(2**i)
            self.netM[scale] = style_mask


class StyleFlow(nn.Module):

    def __init__(self, channels, log_size, style_in):
        super().__init__()
        self.log_size = log_size
        padding_type = 'zero'
        actvn = 'lrelu'
        normalize_mlp = False
        modulated_conv = True

        self.netRefine = nn.ModuleDict()
        self.netStyle = nn.ModuleDict()
        self.netF = nn.ModuleDict()

        for i in range(4, self.log_size + 1):
            out_channel = channels[2**i]

            netRefine_layer = torch.nn.Sequential(
                torch.nn.Conv2d(
                    2 * out_channel,
                    out_channels=128,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(
                    in_channels=128,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1),
                torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                torch.nn.Conv2d(
                    in_channels=32,
                    out_channels=2,
                    kernel_size=3,
                    stride=1,
                    padding=1))

            style_block = StyledConvBlock(
                out_channel,
                49,
                latent_dim=style_in,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv)

            style_F_block = Styled_F_ConvBlock(
                49,
                2,
                latent_dim=style_in,
                padding=padding_type,
                actvn=actvn,
                normalize_affine_output=normalize_mlp,
                modulated_conv=modulated_conv)

            scale = str(2**i)
            self.netRefine[scale] = (netRefine_layer)
            self.netStyle[scale] = (style_block)
            self.netF[scale] = (style_F_block)


class StyledConvBlock(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 latent_dim=256,
                 padding='zero',
                 actvn='lrelu',
                 normalize_affine_output=False,
                 modulated_conv=False):
        super(StyledConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                fout,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, fout, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                fout,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(fout, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

        self.actvn1 = activation

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        out = self.actvn1(out) * self.actvn_gain

        return out


class Styled_F_ConvBlock(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 latent_dim=256,
                 padding='zero',
                 actvn='lrelu',
                 normalize_affine_output=False,
                 modulated_conv=False):
        super(Styled_F_ConvBlock, self).__init__()
        if not modulated_conv:
            if padding == 'reflect':
                padding_layer = nn.ReflectionPad2d
            else:
                padding_layer = nn.ZeroPad2d

        if modulated_conv:
            conv2d = ModulatedConv2d
        else:
            conv2d = EqualConv2d

        if modulated_conv:
            self.actvn_gain = sqrt(2)
        else:
            self.actvn_gain = 1.0

        self.modulated_conv = modulated_conv

        if actvn == 'relu':
            activation = nn.ReLU(True)
        else:
            activation = nn.LeakyReLU(0.2, True)

        if self.modulated_conv:
            self.conv0 = conv2d(
                fin,
                128,
                kernel_size=3,
                padding_type=padding,
                upsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv0 = conv2d(fin, 128, kernel_size=3)

            seq0 = [padding_layer(1), conv0]
            self.conv0 = nn.Sequential(*seq0)

        self.actvn0 = activation

        if self.modulated_conv:
            self.conv1 = conv2d(
                128,
                fout,
                kernel_size=3,
                padding_type=padding,
                downsample=False,
                latent_dim=latent_dim,
                normalize_mlp=normalize_affine_output)
        else:
            conv1 = conv2d(128, fout, kernel_size=3)
            seq1 = [padding_layer(1), conv1]
            self.conv1 = nn.Sequential(*seq1)

    def forward(self, input, latent=None):
        if self.modulated_conv:
            out = self.conv0(input, latent)
        else:
            out = self.conv0(input)

        out = self.actvn0(out) * self.actvn_gain

        if self.modulated_conv:
            out = self.conv1(out, latent)
        else:
            out = self.conv1(out)

        return out


class ModulatedConv2d(nn.Module):

    def __init__(self,
                 fin,
                 fout,
                 kernel_size,
                 padding_type='zero',
                 upsample=False,
                 downsample=False,
                 latent_dim=512,
                 normalize_mlp=False):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = fin
        self.out_channels = fout
        self.kernel_size = kernel_size
        padding_size = kernel_size // 2

        if kernel_size == 1:
            self.demudulate = False
        else:
            self.demudulate = True

        self.weight = nn.Parameter(
            torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(1, fout, 1, 1))

        if normalize_mlp:
            self.mlp_class_std = nn.Sequential(
                EqualLinear(latent_dim, fin), PixelNorm())
        else:
            self.mlp_class_std = EqualLinear(latent_dim, fin)

        if padding_type == 'reflect':
            self.padding = nn.ReflectionPad2d(padding_size)
        else:
            self.padding = nn.ZeroPad2d(padding_size)

        self.weight.data.normal_()
        self.bias.data.zero_()

    def forward(self, input, latent):
        fan_in = self.weight.data.size(1) * self.weight.data[0][0].numel()
        weight = self.weight * sqrt(2 / fan_in)
        weight = weight.view(1, self.out_channels, self.in_channels,
                             self.kernel_size, self.kernel_size)

        s = self.mlp_class_std(latent).view(-1, 1, self.in_channels, 1, 1)
        weight = s * weight
        if self.demudulate:
            d = torch.rsqrt((weight**2).sum(4).sum(3).sum(2) + 1e-5).view(
                -1, self.out_channels, 1, 1, 1)
            weight = (d * weight).view(-1, self.in_channels, self.kernel_size,
                                       self.kernel_size)
        else:
            weight = weight.view(-1, self.in_channels, self.kernel_size,
                                 self.kernel_size)

        batch, _, height, width = input.shape

        input = input.reshape(1, -1, height, width)
        input = self.padding(input)
        out = F.conv2d(
            input, weight, groups=batch).view(batch, self.out_channels, height,
                                              width) + self.bias

        return out
