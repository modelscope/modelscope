#!/usr/bin/env python3
#
# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed and modified from MP-SENet,
# public available at https://github.com/yxlu-0102/MP-SENet

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubPixelConvTranspose2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(1, 3),
                 stride=(1, 2),
                 padding=(0, 1)):
        super(SubPixelConvTranspose2d, self).__init__()
        self.upscale_width_factor = stride[1]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels * self.upscale_width_factor,
            kernel_size=kernel_size,
            padding=padding)  # only change the width

    def forward(self, x):

        b, c, t, f = x.size()
        # Use conv1 for upsampling, followed by expansion only in the width dimension.
        x = self.conv1(x)
        # print(x.size())
        # Note: Here we do not directly use PixelShuffle because we only intend to expand in the width dimension,
        # whereas PixelShuffle operates simultaneously on both height and width, hence we manually adjust accordingly.
        # b, 2c, t, f
        # print(x.size())
        x = x.view(b, c, self.upscale_width_factor, t,
                   f).permute(0, 1, 3, 4, 2).contiguous()
        # b, c, 2, t, f -> b, c, t, f, 2
        x = x.view(b, c, t, f * self.upscale_width_factor)
        # b, c, t, 2f = 202
        # x = nn.functional.pad(x, (0, 1))
        # b, c, t, 2f = 202

        return x


class DenseBlockV2(nn.Module):
    """
    A denseblock for ZipEnhancer
    """

    def __init__(self, h, kernel_size=(2, 3), depth=4):
        super(DenseBlockV2, self).__init__()
        self.h = h
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2**i
            pad_length = kernel_size[0] + (dil - 1) * (kernel_size[0] - 1) - 1
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.),
                nn.Conv2d(
                    h.dense_channel * (i + 1),
                    h.dense_channel,
                    kernel_size,
                    dilation=(dil, 1)),
                # nn.Conv2d(h.dense_channel * (i + 1), h.dense_channel, kernel_size, dilation=(dil, 1),
                #           padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(h.dense_channel, affine=True),
                nn.PReLU(h.dense_channel))
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        # b, c, t, f
        for i in range(self.depth):
            _x = skip
            x = self.dense_block[i](_x)
            # print(x.size())
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):

    def __init__(self, h, in_channel):
        """
        Initialize the DenseEncoder module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        in_channel (int): Number of input channels. Example: mag + phase: 2 channels
        """
        super(DenseEncoder, self).__init__()
        self.h = h
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, h.dense_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

        self.dense_block = DenseBlockV2(h, depth=4)

        encoder_pad_kersize = (0, 1)
        # Here pad was originally (0, 0)，now change to (0, 1)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(
                h.dense_channel,
                h.dense_channel, (1, 3), (1, 2),
                padding=encoder_pad_kersize),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))

    def forward(self, x):
        """
        Forward pass of the DenseEncoder module.

        Args:
        x (Tensor): Input tensor of shape [B, C=in_channel, T, F].

        Returns:
        Tensor: Output tensor after passing through the dense encoder. Maybe: [B, C=dense_channel, T, F // 2].
        """
        # print("x: {}".format(x.size()))
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        if self.dense_block is not None:
            x = self.dense_block(x)  # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x


class BaseDecoder(nn.Module):

    def __init__(self, h):
        """
        Initialize the BaseDecoder module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        including upsample_type, dense_block_type.
        """
        super(BaseDecoder, self).__init__()

        self.upsample_module_class = SubPixelConvTranspose2d

        # for both mag and phase decoder
        self.dense_block = DenseBlockV2(h, depth=4)


class MappingDecoder(BaseDecoder):

    def __init__(self, h, out_channel=1):
        """
        Initialize the MappingDecoderV3 module.

        Args:
        h (object): Configuration object containing various hyperparameters and settings.
        out_channel (int): Number of output channels. Default is 1. The number of output spearkers.
        """
        super(MappingDecoder, self).__init__(h)
        decoder_final_kersize = (1, 2)

        self.mask_conv = nn.Sequential(
            self.upsample_module_class(h.dense_channel, h.dense_channel,
                                       (1, 3), (1, 2)),
            # nn.Conv2d(h.dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel),
            nn.Conv2d(h.dense_channel, out_channel, decoder_final_kersize))
        # Upsample at F dimension

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the MappingDecoderV3 module.

        Args:
        x (Tensor): Input tensor. [B, C, T, F]

        Returns:
        Tensor: Output tensor after passing through the decoder. [B, Num_Spks, T, F]
        """
        if self.dense_block is not None:
            x = self.dense_block(x)
        x = self.mask_conv(x)
        x = self.relu(x)
        # b, c=1, t, f
        return x


class PhaseDecoder(BaseDecoder):

    def __init__(self, h, out_channel=1):
        super(PhaseDecoder, self).__init__(h)

        # now change to (1, 2), previous (1, 1)
        decoder_final_kersize = (1, 2)

        self.phase_conv = nn.Sequential(
            self.upsample_module_class(h.dense_channel, h.dense_channel,
                                       (1, 3), (1, 2)),
            nn.InstanceNorm2d(h.dense_channel, affine=True),
            nn.PReLU(h.dense_channel))
        self.phase_conv_r = nn.Conv2d(h.dense_channel, out_channel,
                                      decoder_final_kersize)
        self.phase_conv_i = nn.Conv2d(h.dense_channel, out_channel,
                                      decoder_final_kersize)

    def forward(self, x):
        if self.dense_block is not None:
            x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
