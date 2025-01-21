# --------------------------------------------------------
# The implementation is also open-sourced by the authors as Yang Liu, and is available publicly on
# https://github.com/damo-cv/MogFace
# --------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_bn(nn.Module):
    """docstring for conv"""

    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(conv_bn, self).__init__()
        self.conv1 = nn.Conv2d(
            in_plane,
            out_plane,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn1 = nn.BatchNorm2d(out_plane)

    def forward(self, x):
        x = self.conv1(x)
        return self.bn1(x)


class SSHContext(nn.Module):

    def __init__(self, channels, Xchannels=256):
        super(SSHContext, self).__init__()

        self.conv1 = nn.Conv2d(
            channels, Xchannels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            channels,
            Xchannels // 2,
            kernel_size=3,
            dilation=2,
            stride=1,
            padding=2)
        self.conv2_1 = nn.Conv2d(
            Xchannels // 2, Xchannels // 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(
            Xchannels // 2,
            Xchannels // 2,
            kernel_size=3,
            dilation=2,
            stride=1,
            padding=2)
        self.conv2_2_1 = nn.Conv2d(
            Xchannels // 2, Xchannels // 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x), inplace=True)
        x2_1 = F.relu(self.conv2_1(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2(x2), inplace=True)
        x2_2 = F.relu(self.conv2_2_1(x2_2), inplace=True)

        return torch.cat([x1, x2_1, x2_2], 1)


class DeepHead(nn.Module):

    def __init__(self,
                 in_channel=256,
                 out_channel=256,
                 use_gn=False,
                 num_conv=4):
        super(DeepHead, self).__init__()
        self.use_gn = use_gn
        self.num_conv = num_conv
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        if self.use_gn:
            self.gn1 = nn.GroupNorm(16, out_channel)
            self.gn2 = nn.GroupNorm(16, out_channel)
            self.gn3 = nn.GroupNorm(16, out_channel)
            self.gn4 = nn.GroupNorm(16, out_channel)

    def forward(self, x):
        if self.use_gn:
            x1 = F.relu(self.gn1(self.conv1(x)), inplace=True)
            x2 = F.relu(self.gn2(self.conv1(x1)), inplace=True)
            x3 = F.relu(self.gn3(self.conv1(x2)), inplace=True)
            x4 = F.relu(self.gn4(self.conv1(x3)), inplace=True)
        else:
            x1 = F.relu(self.conv1(x), inplace=True)
            x2 = F.relu(self.conv1(x1), inplace=True)
            if self.num_conv == 2:
                return x2
            x3 = F.relu(self.conv1(x2), inplace=True)
            x4 = F.relu(self.conv1(x3), inplace=True)

        return x4


class MogPredNet(nn.Module):

    def __init__(self,
                 num_anchor_per_pixel=1,
                 num_classes=1,
                 input_ch_list=[256, 256, 256, 256, 256, 256],
                 use_deep_head=True,
                 deep_head_with_gn=True,
                 use_ssh=True,
                 deep_head_ch=512):
        super(MogPredNet, self).__init__()
        self.num_classes = num_classes
        self.use_deep_head = use_deep_head
        self.deep_head_with_gn = deep_head_with_gn

        self.use_ssh = use_ssh

        self.deep_head_ch = deep_head_ch

        if self.use_ssh:
            self.conv_SSH = SSHContext(input_ch_list[0],
                                       self.deep_head_ch // 2)

        if self.use_deep_head:
            if self.deep_head_with_gn:
                self.deep_loc_head = DeepHead(
                    self.deep_head_ch, self.deep_head_ch, use_gn=True)
                self.deep_cls_head = DeepHead(
                    self.deep_head_ch, self.deep_head_ch, use_gn=True)

            self.pred_cls = nn.Conv2d(self.deep_head_ch,
                                      1 * num_anchor_per_pixel, 3, 1, 1)
            self.pred_loc = nn.Conv2d(self.deep_head_ch,
                                      4 * num_anchor_per_pixel, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pyramid_feature_list, dsfd_ft_list=None):
        loc = []
        conf = []

        if self.use_deep_head:
            for x in pyramid_feature_list:
                if self.use_ssh:
                    x = self.conv_SSH(x)
                x_cls = self.deep_cls_head(x)
                x_loc = self.deep_loc_head(x)

                conf.append(
                    self.pred_cls(x_cls).permute(0, 2, 3, 1).contiguous())
                loc.append(
                    self.pred_loc(x_loc).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat(
            [o.view(o.size(0), -1, self.num_classes) for o in conf], 1)
        output = (
            self.sigmoid(conf.view(conf.size(0), -1, self.num_classes)),
            loc.view(loc.size(0), -1, 4),
        )

        return output
