# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from .layers import Conv3x3, ConvBlock, upsample
from .mobilenet import mobilenet_v2
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class Equi(nn.Module):
    """ Model: Resnet based Encoder + Decoder
    """

    def __init__(self,
                 num_layers,
                 equi_h,
                 equi_w,
                 pretrained=False,
                 max_depth=10.0,
                 **kwargs):
        super(Equi, self).__init__()

        self.num_layers = num_layers
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.cube_h = equi_h // 2

        # encoder
        encoder = {
            2: mobilenet_v2,
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152
        }

        if num_layers not in encoder:
            raise ValueError(
                '{} is not a valid number of resnet layers'.format(num_layers))
        self.equi_encoder = encoder[num_layers](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        if num_layers < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])

        # decoder
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.equi_dec_convs = OrderedDict()

        self.equi_dec_convs['upconv_5'] = ConvBlock(self.num_ch_enc[4],
                                                    self.num_ch_dec[4])

        self.equi_dec_convs['deconv_4'] = ConvBlock(
            self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs['upconv_4'] = ConvBlock(self.num_ch_dec[4],
                                                    self.num_ch_dec[3])

        self.equi_dec_convs['deconv_3'] = ConvBlock(
            self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs['upconv_3'] = ConvBlock(self.num_ch_dec[3],
                                                    self.num_ch_dec[2])

        self.equi_dec_convs['deconv_2'] = ConvBlock(
            self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs['upconv_2'] = ConvBlock(self.num_ch_dec[2],
                                                    self.num_ch_dec[1])

        self.equi_dec_convs['deconv_1'] = ConvBlock(
            self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs['upconv_1'] = ConvBlock(self.num_ch_dec[1],
                                                    self.num_ch_dec[0])

        self.equi_dec_convs['deconv_0'] = ConvBlock(self.num_ch_dec[0],
                                                    self.num_ch_dec[0])
        self.equi_dec_convs['depthconv_0'] = Conv3x3(self.num_ch_dec[0], 1)

        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))

        self.sigmoid = nn.Sigmoid()
        self.max_depth = nn.Parameter(
            torch.tensor(max_depth), requires_grad=False)

    def forward(self, input_equi_image, input_cube_image):

        # euqi image encoding
        if self.num_layers < 18:
            equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4 \
                = self.equi_encoder(input_equi_image)
        else:
            x = self.equi_encoder.conv1(input_equi_image)
            x = self.equi_encoder.relu(self.equi_encoder.bn1(x))
            equi_enc_feat0 = x

            x = self.equi_encoder.maxpool(x)
            equi_enc_feat1 = self.equi_encoder.layer1(x)
            equi_enc_feat2 = self.equi_encoder.layer2(equi_enc_feat1)
            equi_enc_feat3 = self.equi_encoder.layer3(equi_enc_feat2)
            equi_enc_feat4 = self.equi_encoder.layer4(equi_enc_feat3)

        # euqi image decoding
        outputs = {}

        equi_x = equi_enc_feat4
        equi_x = upsample(self.equi_dec_convs['upconv_5'](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)
        equi_x = self.equi_dec_convs['deconv_4'](equi_x)
        equi_x = upsample(self.equi_dec_convs['upconv_4'](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = self.equi_dec_convs['deconv_3'](equi_x)
        equi_x = upsample(self.equi_dec_convs['upconv_3'](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs['deconv_2'](equi_x)
        equi_x = upsample(self.equi_dec_convs['upconv_2'](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)
        equi_x = self.equi_dec_convs['deconv_1'](equi_x)
        equi_x = upsample(self.equi_dec_convs['upconv_1'](equi_x))

        equi_x = self.equi_dec_convs['deconv_0'](equi_x)
        equi_depth = self.equi_dec_convs['depthconv_0'](equi_x)
        outputs['pred_depth'] = self.max_depth * self.sigmoid(equi_depth)

        return outputs
