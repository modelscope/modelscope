# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .layers import Conv3x3, ConvBlock, upsample


class DepthDecoder(nn.Module):

    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=1,
                 use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i
                                                                           + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[('dispconv', s)] = Conv3x3(self.num_ch_dec[s],
                                                  self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            self.outputs[('feat', i)] = x
            if i in self.scales:
                self.outputs[('disp',
                              i)] = self.sigmoid(self.convs[('dispconv',
                                                             i)](x))

        return self.outputs


class DepthDecoderShare(nn.Module):

    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=1,
                 stride=8,
                 use_skips=True,
                 num_ch_dec=[16, 32, 64, 128, 256]):
        super(DepthDecoderShare, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256]) #(s1:16, s2:32, s4:64, s8:128, s16:256)
        self.num_ch_dec = num_ch_dec

        self.stride = stride
        # (4:s16, 3:s8, 2:s4, 1:s2, 0:s1)
        if self.stride == 8:
            self.scale_idx = 3
        elif self.stride == 4:
            self.scale_idx = 2
        else:
            raise NotImplementedError

        # decoder
        self.convs = OrderedDict()
        for i in range(4, self.scale_idx - 1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i
                                                                           + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        self.outputs[('feat', -1)] = x
        for i in range(4, self.scale_idx - 1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            self.outputs[('feat', i)] = x

        return self.outputs


class DepthDecoderShareFeat(nn.Module):

    def __init__(self,
                 num_ch_enc,
                 scales=range(4),
                 num_output_channels=1,
                 stride=8,
                 use_skips=True,
                 num_ch_dec=[16, 32, 64, 128, 256]):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        # self.num_ch_dec = np.array([16, 32, 64, 128, 256]) #(s1:16, s2:32, s4:64, s8:128, s16:256)
        self.num_ch_dec = num_ch_dec

        self.stride = stride
        # (4:s16, 3:s8, 2:s4, 1:s2, 0:s1)
        if self.stride == 8:
            self.scale_idx = 3
        elif self.stride == 4:
            self.scale_idx = 2
        else:
            raise NotImplementedError

        # decoder
        self.convs = OrderedDict()
        for i in range(4, self.scale_idx - 1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i
                                                                           + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = nn.Conv2d(num_ch_in, num_ch_out, 3,
                                                     1, 1)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = nn.Conv2d(num_ch_in, num_ch_out, 3,
                                                     1, 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        self.outputs[('feat', -1)] = x
        for i in range(4, self.scale_idx - 1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            self.outputs[('feat', i)] = x

        return self.outputs


class UnetDecoder(nn.Module):

    def __init__(self,
                 num_ch_enc,
                 num_output_channels=1,
                 stride=8,
                 out_chs=128,
                 use_skips=True):
        super(UnetDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.stride = stride
        # (4:s16, 3:s8, 2:s4, 1:s2, 0:s1)
        if self.stride == 8:
            self.scale_idx = 3
        elif self.stride == 4:
            self.scale_idx = 2
        else:
            raise NotImplementedError
        # decoder
        self.convs = OrderedDict()
        for i in range(4, self.scale_idx - 1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i
                                                                           + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i] if i != self.scale_idx else out_chs

            self.convs[('upconv', i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, self.scale_idx - 1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            self.outputs[('feat', i)] = x

        return self.outputs
