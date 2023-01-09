# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward
from torch.nn.utils import spectral_norm, weight_norm

from modelscope.models.audio.tts.kantts.utils.audio_torch import stft
from .layers import (CausalConv1d, CausalConvTranspose1d, Conv1d,
                     ConvTranspose1d, ResidualBlock, SourceModule)

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion('1.7')


class Generator(torch.nn.Module):

    def __init__(
        self,
        in_channels=80,
        out_channels=1,
        channels=512,
        kernel_size=7,
        upsample_scales=(8, 8, 2, 2),
        upsample_kernal_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=[(1, 3, 5), (1, 3, 5), (1, 3, 5)],
        repeat_upsample=True,
        bias=True,
        causal=True,
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
        use_weight_norm=True,
        nsf_params=None,
    ):
        super(Generator, self).__init__()

        # check hyperparameters are valid
        assert kernel_size % 2 == 1, 'Kernal size must be odd number.'
        assert len(upsample_scales) == len(upsample_kernal_sizes)
        assert len(resblock_dilations) == len(resblock_kernel_sizes)

        self.upsample_scales = upsample_scales
        self.repeat_upsample = repeat_upsample
        self.num_upsamples = len(upsample_kernal_sizes)
        self.num_kernels = len(resblock_kernel_sizes)
        self.out_channels = out_channels
        self.nsf_enable = nsf_params is not None

        self.transpose_upsamples = torch.nn.ModuleList()
        self.repeat_upsamples = torch.nn.ModuleList()  # for repeat upsampling
        self.conv_blocks = torch.nn.ModuleList()

        conv_cls = CausalConv1d if causal else Conv1d
        conv_transposed_cls = CausalConvTranspose1d if causal else ConvTranspose1d

        self.conv_pre = conv_cls(
            in_channels,
            channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2)

        for i in range(len(upsample_kernal_sizes)):
            self.transpose_upsamples.append(
                torch.nn.Sequential(
                    getattr(
                        torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
                    conv_transposed_cls(
                        channels // (2**i),
                        channels // (2**(i + 1)),
                        upsample_kernal_sizes[i],
                        upsample_scales[i],
                        padding=(upsample_kernal_sizes[i] - upsample_scales[i])
                        // 2,
                    ),
                ))

            if repeat_upsample:
                self.repeat_upsamples.append(
                    nn.Sequential(
                        nn.Upsample(
                            mode='nearest', scale_factor=upsample_scales[i]),
                        getattr(torch.nn, nonlinear_activation)(
                            **nonlinear_activation_params),
                        conv_cls(
                            channels // (2**i),
                            channels // (2**(i + 1)),
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size - 1) // 2,
                        ),
                    ))

            for j in range(len(resblock_kernel_sizes)):
                self.conv_blocks.append(
                    ResidualBlock(
                        channels=channels // (2**(i + 1)),
                        kernel_size=resblock_kernel_sizes[j],
                        dilation=resblock_dilations[j],
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        causal=causal,
                    ))

        self.conv_post = conv_cls(
            channels // (2**(i + 1)),
            out_channels,
            kernel_size,
            1,
            padding=(kernel_size - 1) // 2,
        )

        if self.nsf_enable:
            self.source_module = SourceModule(
                nb_harmonics=nsf_params['nb_harmonics'],
                upsample_ratio=np.cumprod(self.upsample_scales)[-1],
                sampling_rate=nsf_params['sampling_rate'],
            )
            self.source_downs = nn.ModuleList()
            self.downsample_rates = [1] + self.upsample_scales[::-1][:-1]
            self.downsample_cum_rates = np.cumprod(self.downsample_rates)

            for i, u in enumerate(self.downsample_cum_rates[::-1]):
                if u == 1:
                    self.source_downs.append(
                        Conv1d(1, channels // (2**(i + 1)), 1, 1))
                else:
                    self.source_downs.append(
                        conv_cls(
                            1,
                            channels // (2**(i + 1)),
                            u * 2,
                            u,
                            padding=u // 2,
                        ))

    def forward(self, x):
        if self.nsf_enable:
            mel = x[:, :-2, :]
            pitch = x[:, -2:-1, :]
            uv = x[:, -1:, :]
            excitation = self.source_module(pitch, uv)
        else:
            mel = x

        x = self.conv_pre(mel)
        for i in range(self.num_upsamples):
            #  FIXME: sin function here seems to be causing issues
            x = torch.sin(x) + x
            rep = self.repeat_upsamples[i](x)

            if self.nsf_enable:
                # Downsampling the excitation signal
                e = self.source_downs[i](excitation)
                # augment inputs with the excitation
                x = rep + e
            else:
                # transconv
                up = self.transpose_upsamples[i](x)
                x = rep + up[:, :, :rep.shape[-1]]

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.conv_blocks[i * self.num_kernels + j](x)
                else:
                    xs += self.conv_blocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for layer in self.transpose_upsamples:
            layer[-1].remove_weight_norm()
        for layer in self.repeat_upsamples:
            layer[-1].remove_weight_norm()
        for layer in self.conv_blocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()
        if self.nsf_enable:
            self.source_module.remove_weight_norm()
            for layer in self.source_downs:
                layer.remove_weight_norm()


class PeriodDiscriminator(torch.nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
        use_spectral_norm=False,
    ):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList()
        in_chs, out_chs = in_channels, channels

        for downsample_scale in downsample_scales:
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv2d(
                            in_chs,
                            out_chs,
                            (kernel_sizes[0], 1),
                            (downsample_scale, 1),
                            padding=((kernel_sizes[0] - 1) // 2, 0),
                        )),
                    getattr(
                        torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
                ))
            in_chs = out_chs
            out_chs = min(out_chs * 4, max_downsample_channels)

        self.conv_post = nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            'in_channels': 1,
            'out_channels': 1,
            'kernel_sizes': [5, 3],
            'channels': 32,
            'downsample_scales': [3, 3, 3, 3, 1],
            'max_downsample_channels': 1024,
            'bias': True,
            'nonlinear_activation': 'LeakyReLU',
            'nonlinear_activation_params': {
                'negative_slope': 0.1
            },
            'use_spectral_norm': False,
        },
    ):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params['period'] = period
            self.discriminators += [PeriodDiscriminator(**params)]

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class ScaleDiscriminator(torch.nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
        use_spectral_norm=False,
    ):
        super(ScaleDiscriminator, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm

        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        self.convs = nn.ModuleList()

        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv1d(
                        in_channels,
                        channels,
                        kernel_sizes[0],
                        bias=bias,
                        padding=(kernel_sizes[0] - 1) // 2,
                    )),
                getattr(torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
            ))
        in_chs = channels
        out_chs = channels
        groups = 4

        for downsample_scale in downsample_scales:
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv1d(
                            in_chs,
                            out_chs,
                            kernel_size=kernel_sizes[1],
                            stride=downsample_scale,
                            padding=(kernel_sizes[1] - 1) // 2,
                            groups=groups,
                            bias=bias,
                        )),
                    getattr(
                        torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
                ))
            in_chs = out_chs
            out_chs = min(in_chs * 2, max_downsample_channels)
            groups = min(groups * 4, max_groups)

        out_chs = min(in_chs * 2, max_downsample_channels)
        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[2],
                        stride=1,
                        padding=(kernel_sizes[2] - 1) // 2,
                        bias=bias,
                    )),
                getattr(torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
            ))

        self.conv_post = norm_f(
            nn.Conv1d(
                out_chs,
                out_channels,
                kernel_size=kernel_sizes[3],
                stride=1,
                padding=(kernel_sizes[3] - 1) // 2,
                bias=bias,
            ))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(
        self,
        scales=3,
        downsample_pooling='DWT',
        # follow the official implementation setting
        downsample_pooling_params={
            'kernel_size': 4,
            'stride': 2,
            'padding': 2,
        },
        discriminator_params={
            'in_channels': 1,
            'out_channels': 1,
            'kernel_sizes': [15, 41, 5, 3],
            'channels': 128,
            'max_downsample_channels': 1024,
            'max_groups': 16,
            'bias': True,
            'downsample_scales': [2, 2, 4, 4, 1],
            'nonlinear_activation': 'LeakyReLU',
            'nonlinear_activation_params': {
                'negative_slope': 0.1
            },
        },
        follow_official_norm=False,
    ):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                params['use_spectral_norm'] = True if i == 0 else False
            self.discriminators += [ScaleDiscriminator(**params)]

        if downsample_pooling == 'DWT':
            self.meanpools = nn.ModuleList(
                [DWT1DForward(wave='db3', J=1),
                 DWT1DForward(wave='db3', J=1)])
            self.aux_convs = nn.ModuleList([
                weight_norm(nn.Conv1d(2, 1, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(2, 1, 15, 1, padding=7)),
            ])
        else:
            self.meanpools = nn.ModuleList(
                [nn.AvgPool1d(4, 2, padding=2),
                 nn.AvgPool1d(4, 2, padding=2)])
            self.aux_convs = None

    def forward(self, y):
        y_d_rs = []
        fmap_rs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                if self.aux_convs is None:
                    y = self.meanpools[i - 1](y)
                else:
                    yl, yh = self.meanpools[i - 1](y)
                    y = torch.cat([yl, yh[0]], dim=1)
                    y = self.aux_convs[i - 1](y)
                    y = F.leaky_relu(y, 0.1)

            y_d_r, fmap_r = d(y)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)

        return y_d_rs, fmap_rs


class SpecDiscriminator(torch.nn.Module):

    def __init__(
        self,
        channels=32,
        init_kernel=15,
        kernel_size=11,
        stride=2,
        use_spectral_norm=False,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window='hann_window',
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
    ):
        super(SpecDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # fft_size // 2 + 1
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        final_kernel = 5
        post_conv_kernel = 3
        blocks = 3
        self.convs = nn.ModuleList()
        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv2d(
                        fft_size // 2 + 1,
                        channels,
                        (init_kernel, 1),
                        (1, 1),
                        padding=(init_kernel - 1) // 2,
                    )),
                getattr(torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
            ))

        for i in range(blocks):
            self.convs.append(
                torch.nn.Sequential(
                    norm_f(
                        nn.Conv2d(
                            channels,
                            channels,
                            (kernel_size, 1),
                            (stride, 1),
                            padding=(kernel_size - 1) // 2,
                        )),
                    getattr(
                        torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
                ))

        self.convs.append(
            torch.nn.Sequential(
                norm_f(
                    nn.Conv2d(
                        channels,
                        channels,
                        (final_kernel, 1),
                        (1, 1),
                        padding=(final_kernel - 1) // 2,
                    )),
                getattr(torch.nn,
                        nonlinear_activation)(**nonlinear_activation_params),
            ))

        self.conv_post = norm_f(
            nn.Conv2d(
                channels,
                1,
                (post_conv_kernel, 1),
                (1, 1),
                padding=((post_conv_kernel - 1) // 2, 0),
            ))
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, wav):
        with torch.no_grad():
            wav = torch.squeeze(wav, 1)
            x_mag = stft(wav, self.fft_size, self.shift_size, self.win_length,
                         self.window)
            x = torch.transpose(x_mag, 2, 1).unsqueeze(-1)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.squeeze(-1)

        return x, fmap


class MultiSpecDiscriminator(torch.nn.Module):

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        discriminator_params={
            'channels': 15,
            'init_kernel': 1,
            'kernel_sizes': 11,
            'stride': 2,
            'use_spectral_norm': False,
            'window': 'hann_window',
            'nonlinear_activation': 'LeakyReLU',
            'nonlinear_activation_params': {
                'negative_slope': 0.1
            },
        },
    ):
        super(MultiSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes,
                                                  win_lengths):
            params = copy.deepcopy(discriminator_params)
            params['fft_size'] = fft_size
            params['shift_size'] = hop_size
            params['win_length'] = win_length
            self.discriminators += [SpecDiscriminator(**params)]

    def forward(self, y):
        y_d = []
        fmap = []
        for i, d in enumerate(self.discriminators):
            x, x_map = d(y)
            y_d.append(x)
            fmap.append(x_map)

        return y_d, fmap
