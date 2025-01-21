# Copyright (c) Alibaba, Inc. and its affiliates.
#
# The implementation here is modified based on
# Jongho Choi(sweetcocoa@snu.ac.kr / Seoul National Univ., ESTsoft )
# and publicly available at
# https://github.com/sweetcocoa/DeepComplexUNetPyTorch

import torch
import torch.nn as nn

from . import complex_nn
from .se_module_complex import SELayer


class Encoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=None,
                 complex=False,
                 padding_mode='zeros'):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding

        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=(0, 0),
                 complex=False):
        super().__init__()
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d

        self.transconv = tconv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):

    def __init__(self,
                 input_channels=1,
                 complex=False,
                 model_complexity=45,
                 model_depth=20,
                 padding_mode='zeros'):
        super().__init__()

        if complex:
            model_complexity = int(model_complexity // 1.414)

        self.set_size(
            model_complexity=model_complexity,
            input_channels=input_channels,
            model_depth=model_depth)
        self.encoders = []
        self.model_length = model_depth // 2
        self.fsmn = complex_nn.ComplexUniDeepFsmn(128, 128, 128)
        self.se_layers_enc = []
        self.fsmn_enc = []
        for i in range(self.model_length):
            fsmn_enc = complex_nn.ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module('fsmn_enc{}'.format(i), fsmn_enc)
            self.fsmn_enc.append(fsmn_enc)
            module = Encoder(
                self.enc_channels[i],
                self.enc_channels[i + 1],
                kernel_size=self.enc_kernel_sizes[i],
                stride=self.enc_strides[i],
                padding=self.enc_paddings[i],
                complex=complex,
                padding_mode=padding_mode)
            self.add_module('encoder{}'.format(i), module)
            self.encoders.append(module)
            se_layer_enc = SELayer(self.enc_channels[i + 1], 8)
            self.add_module('se_layer_enc{}'.format(i), se_layer_enc)
            self.se_layers_enc.append(se_layer_enc)
        self.decoders = []
        self.fsmn_dec = []
        self.se_layers_dec = []
        for i in range(self.model_length):
            fsmn_dec = complex_nn.ComplexUniDeepFsmn_L1(128, 128, 128)
            self.add_module('fsmn_dec{}'.format(i), fsmn_dec)
            self.fsmn_dec.append(fsmn_dec)
            module = Decoder(
                self.dec_channels[i] * 2,
                self.dec_channels[i + 1],
                kernel_size=self.dec_kernel_sizes[i],
                stride=self.dec_strides[i],
                padding=self.dec_paddings[i],
                complex=complex)
            self.add_module('decoder{}'.format(i), module)
            self.decoders.append(module)
            if i < self.model_length - 1:
                se_layer_dec = SELayer(self.dec_channels[i + 1], 8)
                self.add_module('se_layer_dec{}'.format(i), se_layer_dec)
                self.se_layers_dec.append(se_layer_dec)
        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d

        linear = conv(self.dec_channels[-1], 1, 1)

        self.add_module('linear', linear)
        self.complex = complex
        self.padding_mode = padding_mode

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.se_layers_enc = nn.ModuleList(self.se_layers_enc)
        self.se_layers_dec = nn.ModuleList(self.se_layers_dec)
        self.fsmn_enc = nn.ModuleList(self.fsmn_enc)
        self.fsmn_dec = nn.ModuleList(self.fsmn_dec)

    def forward(self, inputs):
        x = inputs
        # go down
        xs = []
        xs_se = []
        xs_se.append(x)
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            if i > 0:
                x = self.fsmn_enc[i](x)
            x = encoder(x)
            xs_se.append(self.se_layers_enc[i](x))
        # xs : x0=input x1 ... x9
        x = self.fsmn(x)

        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i < self.model_length - 1:
                p = self.fsmn_dec[i](p)
            if i == self.model_length - 1:
                break
            if i < self.model_length - 2:
                p = self.se_layers_dec[i](p)
            p = torch.cat([p, xs_se[self.model_length - 1 - i]], dim=1)

        # cmp_spec: [12, 1, 513, 64, 2]
        cmp_spec = self.linear(p)
        return cmp_spec

    def set_size(self, model_complexity, model_depth=20, input_channels=1):

        if model_depth == 14:
            self.enc_channels = [
                input_channels, 128, 128, 128, 128, 128, 128, 128
            ]
            self.enc_kernel_sizes = [(5, 2), (5, 2), (5, 2), (5, 2), (5, 2),
                                     (5, 2), (2, 2)]
            self.enc_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
                                (2, 1)]
            self.enc_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                                 (0, 1), (0, 1)]
            self.dec_channels = [64, 128, 128, 128, 128, 128, 128, 1]
            self.dec_kernel_sizes = [(2, 2), (5, 2), (5, 2), (5, 2), (6, 2),
                                     (5, 2), (5, 2)]
            self.dec_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
                                (2, 1)]
            self.dec_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1),
                                 (0, 1), (0, 1)]

        elif model_depth == 10:
            self.enc_channels = [
                input_channels,
                16,
                32,
                64,
                128,
                256,
            ]
            self.enc_kernel_sizes = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
            self.enc_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
            self.enc_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
            self.dec_channels = [128, 128, 64, 32, 16, 1]
            self.dec_kernel_sizes = [(3, 3), (3, 3), (3, 3), (4, 3), (3, 3)]
            self.dec_strides = [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1)]
            self.dec_paddings = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]

        elif model_depth == 20:
            self.enc_channels = [
                input_channels, model_complexity, model_complexity,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, 128
            ]

            self.enc_kernel_sizes = [(7, 1), (1, 7), (6, 4), (7, 5), (5, 3),
                                     (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)]

            self.enc_strides = [(1, 1), (1, 1), (2, 2), (2, 1), (2, 2), (2, 1),
                                (2, 2), (2, 1), (2, 2), (2, 1)]

            self.enc_paddings = [
                (3, 0),
                (0, 3),
                None,  # (0, 2),
                None,
                None,  # (3,1),
                None,  # (3,1),
                None,  # (1,2),
                None,
                None,
                None
            ]

            self.dec_channels = [
                0, model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2, model_complexity * 2,
                model_complexity * 2
            ]

            self.dec_kernel_sizes = [(4, 3), (4, 2), (4, 3), (4, 2), (4, 3),
                                     (4, 2), (6, 3), (7, 4), (1, 7), (7, 1)]

            self.dec_strides = [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1), (2, 2),
                                (2, 1), (2, 2), (1, 1), (1, 1)]

            self.dec_paddings = [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1),
                                 (1, 0), (2, 1), (2, 1), (0, 3), (3, 0)]
        else:
            raise ValueError('Unknown model depth : {}'.format(model_depth))
