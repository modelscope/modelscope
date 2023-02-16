# The implementation is modified from cleinc / bts
# made publicly available under the GPL-3.0-or-later
# https://github.com/cleinc/bts/blob/master/pytorch/bts.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func


class AtrousConv(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 apply_bn_first=True):
        super(AtrousConv, self).__init__()
        self.atrous_conv = torch.nn.Sequential()
        if apply_bn_first:
            self.atrous_conv.add_module(
                'first_bn',
                nn.BatchNorm2d(
                    in_channels,
                    momentum=0.01,
                    affine=True,
                    track_running_stats=True,
                    eps=1.1e-5))

        self.atrous_conv.add_module(
            'aconv_sequence',
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels * 2,
                    bias=False,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.BatchNorm2d(
                    out_channels * 2,
                    momentum=0.01,
                    affine=True,
                    track_running_stats=True), nn.ReLU(),
                nn.Conv2d(
                    in_channels=out_channels * 2,
                    out_channels=out_channels,
                    bias=False,
                    kernel_size=3,
                    stride=1,
                    padding=(dilation, dilation),
                    dilation=dilation)))

    def forward(self, x):
        return self.atrous_conv.forward(x)


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, ratio=2):
        super(UpConv, self).__init__()
        self.elu = nn.ELU()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=3,
            stride=1,
            padding=1)
        self.ratio = ratio

    def forward(self, x):
        up_x = torch_nn_func.interpolate(
            x, scale_factor=self.ratio, mode='nearest')
        out = self.conv(up_x)
        out = self.elu(out)
        return out


class Reduction1x1(nn.Sequential):

    def __init__(self,
                 num_in_filters,
                 num_out_filters,
                 max_depth,
                 is_final=False):
        super(Reduction1x1, self).__init__()
        self.max_depth = max_depth
        self.is_final = is_final
        self.sigmoid = nn.Sigmoid()
        self.reduc = torch.nn.Sequential()

        while num_out_filters >= 4:
            if num_out_filters < 8:
                if self.is_final:
                    self.reduc.add_module(
                        'final',
                        torch.nn.Sequential(
                            nn.Conv2d(
                                num_in_filters,
                                out_channels=1,
                                bias=False,
                                kernel_size=1,
                                stride=1,
                                padding=0), nn.Sigmoid()))
                else:
                    self.reduc.add_module(
                        'plane_params',
                        torch.nn.Conv2d(
                            num_in_filters,
                            out_channels=3,
                            bias=False,
                            kernel_size=1,
                            stride=1,
                            padding=0))
                break
            else:
                self.reduc.add_module(
                    'inter_{}_{}'.format(num_in_filters, num_out_filters),
                    torch.nn.Sequential(
                        nn.Conv2d(
                            in_channels=num_in_filters,
                            out_channels=num_out_filters,
                            bias=False,
                            kernel_size=1,
                            stride=1,
                            padding=0), nn.ELU()))

            num_in_filters = num_out_filters
            num_out_filters = num_out_filters // 2

    def forward(self, net):
        net = self.reduc.forward(net)
        if not self.is_final:
            theta = self.sigmoid(net[:, 0, :, :]) * math.pi / 3
            phi = self.sigmoid(net[:, 1, :, :]) * math.pi * 2
            dist = self.sigmoid(net[:, 2, :, :]) * self.max_depth
            n1 = torch.mul(torch.sin(theta), torch.cos(phi)).unsqueeze(1)
            n2 = torch.mul(torch.sin(theta), torch.sin(phi)).unsqueeze(1)
            n3 = torch.cos(theta).unsqueeze(1)
            n4 = dist.unsqueeze(1)
            net = torch.cat([n1, n2, n3, n4], dim=1)

        return net


class LocalPlanarGuidance(nn.Module):

    def __init__(self, upratio):
        super(LocalPlanarGuidance, self).__init__()
        self.upratio = upratio
        self.u = torch.arange(self.upratio).reshape([1, 1,
                                                     self.upratio]).float()
        self.v = torch.arange(int(self.upratio)).reshape([1, self.upratio,
                                                          1]).float()
        self.upratio = float(upratio)

    def forward(self, plane_eq, focal):
        plane_eq_expanded = torch.repeat_interleave(plane_eq,
                                                    int(self.upratio), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded,
                                                    int(self.upratio), 3)
        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(
            plane_eq.size(0),
            plane_eq.size(2) * int(self.upratio), plane_eq.size(3)).cuda()
        u = (u - (self.upratio - 1) * 0.5) / self.upratio

        v = self.v.repeat(
            plane_eq.size(0), plane_eq.size(2),
            plane_eq.size(3) * int(self.upratio)).cuda()
        v = (v - (self.upratio - 1) * 0.5) / self.upratio

        d = n4 / (n1 * u + n2 * v + n3)

        return d
