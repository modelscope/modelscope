# The implementation is modified from ErenBalatkan/Bts-PyTorch
# made publicly available under the MIT license
# https://github.com/ErenBalatkan/Bts-PyTorch/blob/master/BTS.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

activation_fn = nn.ELU()
MAX_DEPTH = 81


class UpscaleLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpscaleLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.005)

    def forward(self, input):
        input = nn.functional.interpolate(
            input, scale_factor=2, mode='nearest')
        input = activation_fn(self.conv(input))
        input = self.bn(input)
        return input


class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpscaleBlock, self).__init__()
        self.uplayer = UpscaleLayer(in_channels, out_channels)
        self.conv = nn.Conv2d(
            out_channels + skip_channels,
            out_channels,
            3,
            padding=1,
            bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels, 0.005)

    def forward(self, input_j):
        input, skip = input_j
        input = self.uplayer(input)
        cat = torch.cat((input, skip), 1)
        input = activation_fn(self.conv(cat))
        input = self.bn2(input)
        return input, cat


class UpscaleNetwork(nn.Module):

    def __init__(self, filters=[512, 256]):
        super(
            UpscaleNetwork,
            self,
        ).__init__()
        self.upscale_block1 = UpscaleBlock(2208, 384, filters[0])  # H16
        self.upscale_block2 = UpscaleBlock(filters[0], 192, filters[1])  # H8

    def forward(self, raw_input):
        input, h2, h4, h8, h16 = raw_input
        input, _ = self.upscale_block1((input, h16))
        input, cat = self.upscale_block2((input, h8))
        return input, cat


class AtrousBlock(nn.Module):

    def __init__(self,
                 input_filters,
                 filters,
                 dilation,
                 apply_initial_bn=True):
        super(AtrousBlock, self).__init__()

        self.initial_bn = nn.BatchNorm2d(input_filters, 0.005)
        self.apply_initial_bn = apply_initial_bn

        self.conv1 = nn.Conv2d(input_filters, filters * 2, 1, 1, 0, bias=False)
        self.norm1 = nn.BatchNorm2d(filters * 2, 0.005)

        self.atrous_conv = nn.Conv2d(
            filters * 2, filters, 3, 1, dilation, dilation, bias=False)
        self.norm2 = nn.BatchNorm2d(filters, 0.005)

    def forward(self, input):
        if self.apply_initial_bn:
            input = self.initial_bn(input)

        input = self.conv1(input.relu())
        input = self.norm1(input)
        input = self.atrous_conv(input.relu())
        input = self.norm2(input)
        return input


class ASSPBlock(nn.Module):

    def __init__(self, input_filters=256, cat_filters=448, atrous_filters=128):
        super(ASSPBlock, self).__init__()

        self.atrous_conv_r3 = AtrousBlock(
            input_filters, atrous_filters, 3, apply_initial_bn=False)
        self.atrous_conv_r6 = AtrousBlock(cat_filters + atrous_filters,
                                          atrous_filters, 6)
        self.atrous_conv_r12 = AtrousBlock(cat_filters + atrous_filters * 2,
                                           atrous_filters, 12)
        self.atrous_conv_r18 = AtrousBlock(cat_filters + atrous_filters * 3,
                                           atrous_filters, 18)
        self.atrous_conv_r24 = AtrousBlock(cat_filters + atrous_filters * 4,
                                           atrous_filters, 24)

        self.conv = nn.Conv2d(
            5 * atrous_filters + cat_filters,
            atrous_filters,
            3,
            1,
            1,
            bias=True)

    def forward(self, input):
        input, cat = input
        layer1_out = self.atrous_conv_r3(input)
        concat1 = torch.cat((cat, layer1_out), 1)

        layer2_out = self.atrous_conv_r6(concat1)
        concat2 = torch.cat((concat1, layer2_out), 1)

        layer3_out = self.atrous_conv_r12(concat2)
        concat3 = torch.cat((concat2, layer3_out), 1)

        layer4_out = self.atrous_conv_r18(concat3)
        concat4 = torch.cat((concat3, layer4_out), 1)

        layer5_out = self.atrous_conv_r24(concat4)
        concat5 = torch.cat((concat4, layer5_out), 1)

        features = activation_fn(self.conv(concat5))
        return features


class Reduction(nn.Module):

    def __init__(self, scale, input_filters, is_final=False):
        super(Reduction, self).__init__()
        reduction_count = int(math.log(input_filters, 2)) - 2
        self.reductions = torch.nn.Sequential()
        for i in range(reduction_count):
            if i != reduction_count - 1:
                self.reductions.add_module(
                    '1x1_reduc_%d_%d' % (scale, i),
                    nn.Sequential(
                        nn.Conv2d(
                            int(input_filters / math.pow(2, i)),
                            int(input_filters / math.pow(2, i + 1)),
                            1,
                            1,
                            0,
                            bias=True), activation_fn))
            else:
                if not is_final:
                    self.reductions.add_module(
                        '1x1_reduc_%d_%d' % (scale, i),
                        nn.Sequential(
                            nn.Conv2d(
                                int(input_filters / math.pow(2, i)),
                                int(input_filters / math.pow(2, i + 1)),
                                1,
                                1,
                                0,
                                bias=True)))
                else:
                    self.reductions.add_module(
                        '1x1_reduc_%d_%d' % (scale, i),
                        nn.Sequential(
                            nn.Conv2d(
                                int(input_filters / math.pow(2, i)),
                                1,
                                1,
                                1,
                                0,
                                bias=True), nn.Sigmoid()))

    def forward(self, ip):
        return self.reductions(ip)


class LPGBlock(nn.Module):

    def __init__(self, scale, input_filters=128):
        super(LPGBlock, self).__init__()
        self.scale = scale

        self.reduction = Reduction(scale, input_filters)
        self.conv = nn.Conv2d(4, 3, 1, 1, 0)

        self.u = torch.arange(self.scale).reshape([1, 1, self.scale]).float()
        self.v = torch.arange(int(self.scale)).reshape([1, self.scale,
                                                        1]).float()

    def forward(self, input):
        input = self.reduction(input)

        plane_parameters = torch.zeros_like(input)
        input = self.conv(input)

        theta = input[:, 0, :, :].sigmoid() * 3.1415926535 / 6
        phi = input[:, 1, :, :].sigmoid() * 3.1415926535 * 2
        dist = input[:, 2, :, :].sigmoid() * MAX_DEPTH

        plane_parameters[:, 0, :, :] = torch.sin(theta) * torch.cos(phi)
        plane_parameters[:, 1, :, :] = torch.sin(theta) * torch.sin(phi)
        plane_parameters[:, 2, :, :] = torch.cos(theta)
        plane_parameters[:, 3, :, :] = dist

        plane_parameters[:, 0:3, :, :] = F.normalize(
            plane_parameters.clone()[:, 0:3, :, :], 2, 1)

        plane_eq = plane_parameters.float()

        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.scale),
                                                    2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded,
                                                    int(self.scale), 3)

        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(
            plane_eq.size(0),
            plane_eq.size(2) * int(self.scale), plane_eq.size(3)).cuda()
        u = (u - (self.scale - 1) * 0.5) / self.scale

        v = self.v.repeat(
            plane_eq.size(0), plane_eq.size(2),
            plane_eq.size(3) * int(self.scale)).cuda()
        v = (v - (self.scale - 1) * 0.5) / self.scale

        depth = n4 / (n1 * u + n2 * v + n3)
        depth = depth.unsqueeze(1)
        return depth
