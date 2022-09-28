# The implementation is adopted from https://github.com/TengdaHan/CoCLR,
# made pubicly available under the Apache License, Version 2.0 at https://github.com/TengdaHan/CoCLR
# Copyright 2021-2022 The Alibaba FVI Team Authors. All rights reserved.
import torch
import torch.nn as nn


class InceptionBaseConv3D(nn.Module):
    """
    Constructs basic inception 3D conv.
    Modified from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self,
                 cfg,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride,
                 padding=0):
        super(InceptionBaseConv3D, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv.weight.data.normal_(
            mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionBlock3D(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg, in_planes, out_planes):
        super(InceptionBlock3D, self).__init__()

        _gating = cfg.VIDEO.BACKBONE.BRANCH.GATING

        assert len(out_planes) == 6
        assert isinstance(out_planes, list)

        [
            num_out_0_0a, num_out_1_0a, num_out_1_0b, num_out_2_0a,
            num_out_2_0b, num_out_3_0b
        ] = out_planes

        self.branch0 = nn.Sequential(
            InceptionBaseConv3D(
                cfg, in_planes, num_out_0_0a, kernel_size=1, stride=1), )
        self.branch1 = nn.Sequential(
            InceptionBaseConv3D(
                cfg, in_planes, num_out_1_0a, kernel_size=1, stride=1),
            STConv3d(
                cfg,
                num_out_1_0a,
                num_out_1_0b,
                kernel_size=3,
                stride=1,
                padding=1),
        )
        self.branch2 = nn.Sequential(
            InceptionBaseConv3D(
                cfg, in_planes, num_out_2_0a, kernel_size=1, stride=1),
            STConv3d(
                cfg,
                num_out_2_0a,
                num_out_2_0b,
                kernel_size=3,
                stride=1,
                padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            InceptionBaseConv3D(
                cfg, in_planes, num_out_3_0b, kernel_size=1, stride=1),
        )

        self.out_channels = sum(
            [num_out_0_0a, num_out_1_0b, num_out_2_0b, num_out_3_0b])

        self.gating = _gating
        if _gating:
            self.gating_b0 = SelfGating(num_out_0_0a)
            self.gating_b1 = SelfGating(num_out_1_0b)
            self.gating_b2 = SelfGating(num_out_2_0b)
            self.gating_b3 = SelfGating(num_out_3_0b)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        if self.gating:
            x0 = self.gating_b0(x0)
            x1 = self.gating_b1(x1)
            x2 = self.gating_b2(x2)
            x3 = self.gating_b3(x3)

        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
        """Feature gating as used in S3D-G"""
        spatiotemporal_average = torch.mean(input_tensor, dim=[2, 3, 4])
        weights = self.fc(spatiotemporal_average)
        weights = torch.sigmoid(weights)
        return weights[:, :, None, None, None] * input_tensor


class STConv3d(nn.Module):
    """
    Element constructing the S3D/S3DG.
    See models/base/backbone.py L99-186.

    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self,
                 cfg,
                 in_planes,
                 out_planes,
                 kernel_size,
                 stride,
                 padding=0):
        super(STConv3d, self).__init__()
        if isinstance(stride, tuple):
            t_stride = stride[0]
            stride = stride[-1]
        else:  # int
            t_stride = stride

        self.bn_mmt = cfg.BN.MOMENTUM
        self.bn_eps = float(cfg.BN.EPS)
        self._construct_branch(cfg, in_planes, out_planes, kernel_size, stride,
                               t_stride, padding)

    def _construct_branch(self,
                          cfg,
                          in_planes,
                          out_planes,
                          kernel_size,
                          stride,
                          t_stride,
                          padding=0):
        self.conv1 = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=(1, kernel_size, kernel_size),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)
        self.conv2 = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=(kernel_size, 1, 1),
            stride=(t_stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False)

        self.bn1 = nn.BatchNorm3d(
            out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.bn2 = nn.BatchNorm3d(
            out_planes, eps=self.bn_eps, momentum=self.bn_mmt)
        self.relu = nn.ReLU(inplace=True)

        # init
        self.conv1.weight.data.normal_(
            mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.conv2.weight.data.normal_(
            mean=0, std=0.01)  # original s3d is truncated normal within 2 std
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Inception3D(nn.Module):
    """
    Backbone architecture for I3D/S3DG.
    Modifed from https://github.com/TengdaHan/CoCLR/blob/main/backbone/s3dg.py.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (Config): global config object.
        """
        super(Inception3D, self).__init__()
        _input_channel = cfg.DATA.NUM_INPUT_CHANNELS
        self._construct_backbone(cfg, _input_channel)

    def _construct_backbone(self, cfg, input_channel):
        # ------------------- Block 1 -------------------
        self.Conv_1a = STConv3d(
            cfg, input_channel, 64, kernel_size=7, stride=2, padding=3)

        self.block1 = nn.Sequential(self.Conv_1a)  # (64, 32, 112, 112)

        # ------------------- Block 2 -------------------
        self.MaxPool_2a = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Conv_2b = InceptionBaseConv3D(
            cfg, 64, 64, kernel_size=1, stride=1)
        self.Conv_2c = STConv3d(
            cfg, 64, 192, kernel_size=3, stride=1, padding=1)

        self.block2 = nn.Sequential(
            self.MaxPool_2a,  # (64, 32, 56, 56)
            self.Conv_2b,  # (64, 32, 56, 56)
            self.Conv_2c)  # (192, 32, 56, 56)

        # ------------------- Block 3 -------------------
        self.MaxPool_3a = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.Mixed_3b = InceptionBlock3D(
            cfg, in_planes=192, out_planes=[64, 96, 128, 16, 32, 32])
        self.Mixed_3c = InceptionBlock3D(
            cfg, in_planes=256, out_planes=[128, 128, 192, 32, 96, 64])

        self.block3 = nn.Sequential(
            self.MaxPool_3a,  # (192, 32, 28, 28)
            self.Mixed_3b,  # (256, 32, 28, 28)
            self.Mixed_3c)  # (480, 32, 28, 28)

        # ------------------- Block 4 -------------------
        self.MaxPool_4a = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.Mixed_4b = InceptionBlock3D(
            cfg, in_planes=480, out_planes=[192, 96, 208, 16, 48, 64])
        self.Mixed_4c = InceptionBlock3D(
            cfg, in_planes=512, out_planes=[160, 112, 224, 24, 64, 64])
        self.Mixed_4d = InceptionBlock3D(
            cfg, in_planes=512, out_planes=[128, 128, 256, 24, 64, 64])
        self.Mixed_4e = InceptionBlock3D(
            cfg, in_planes=512, out_planes=[112, 144, 288, 32, 64, 64])
        self.Mixed_4f = InceptionBlock3D(
            cfg, in_planes=528, out_planes=[256, 160, 320, 32, 128, 128])

        self.block4 = nn.Sequential(
            self.MaxPool_4a,  # (480, 16, 14, 14)
            self.Mixed_4b,  # (512, 16, 14, 14)
            self.Mixed_4c,  # (512, 16, 14, 14)
            self.Mixed_4d,  # (512, 16, 14, 14)
            self.Mixed_4e,  # (528, 16, 14, 14)
            self.Mixed_4f)  # (832, 16, 14, 14)

        # ------------------- Block 5 -------------------
        self.MaxPool_5a = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.Mixed_5b = InceptionBlock3D(
            cfg, in_planes=832, out_planes=[256, 160, 320, 32, 128, 128])
        self.Mixed_5c = InceptionBlock3D(
            cfg, in_planes=832, out_planes=[384, 192, 384, 48, 128, 128])

        self.block5 = nn.Sequential(
            self.MaxPool_5a,  # (832, 8, 7, 7)
            self.Mixed_5b,  # (832, 8, 7, 7)
            self.Mixed_5c)  # (1024, 8, 7, 7)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['video']
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
