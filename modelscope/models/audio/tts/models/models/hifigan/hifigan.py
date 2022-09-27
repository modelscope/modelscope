# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from https://github.com/jik876/hifi-gan

from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from modelscope.models.audio.tts.models.utils import get_padding, init_weights
from modelscope.utils.logger import get_logger

logger = get_logger()
is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion('1.7')


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


LRELU_SLOPE = 0.1


def get_padding_casual(kernel_size, dilation=1):
    return int(kernel_size * dilation - dilation)


class Conv1dCasual(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(Conv1dCasual, self).__init__()
        self.pad = padding
        self.conv1d = weight_norm(
            Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode))
        self.conv1d.apply(init_weights)

    def forward(self, x):  # bdt
        # described starting from the last dimension and moving forward.
        x = F.pad(x, (self.pad, 0, 0, 0, 0, 0), 'constant')
        x = self.conv1d(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


class ConvTranspose1dCausal(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0):
        """Initialize CausalConvTranspose1d module."""
        super(ConvTranspose1dCausal, self).__init__()
        self.deconv = weight_norm(
            ConvTranspose1d(in_channels, out_channels, kernel_size, stride))
        self.stride = stride
        self.deconv.apply(init_weights)
        self.pad = kernel_size - stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        # x = F.pad(x, (self.pad, 0, 0, 0, 0, 0), "constant")
        return self.deconv(x)[:, :, :-self.pad]

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


class ResBlock1(torch.nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            Conv1dCasual(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[i],
                padding=get_padding_casual(kernel_size, dilation[i]))
            for i in range(len(dilation))
        ])

        self.convs2 = nn.ModuleList([
            Conv1dCasual(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding_casual(kernel_size, 1))
            for i in range(len(dilation))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class Generator(torch.nn.Module):

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        logger.info('num_kernels={}, num_upsamples={}'.format(
            self.num_kernels, self.num_upsamples))
        self.conv_pre = Conv1dCasual(
            80, h.upsample_initial_channel, 7, 1, padding=7 - 1)
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        self.repeat_ups = nn.ModuleList()
        for i, (u, k) in enumerate(
                zip(h.upsample_rates, h.upsample_kernel_sizes)):
            upsample = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=u),
                nn.LeakyReLU(LRELU_SLOPE),
                Conv1dCasual(
                    h.upsample_initial_channel // (2**i),
                    h.upsample_initial_channel // (2**(i + 1)),
                    kernel_size=7,
                    stride=1,
                    padding=7 - 1))
            self.repeat_ups.append(upsample)
            self.ups.append(
                ConvTranspose1dCausal(
                    h.upsample_initial_channel // (2**i),
                    h.upsample_initial_channel // (2**(i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(
                    zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = Conv1dCasual(ch, 1, 7, 1, padding=7 - 1)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch.sin(x) + x
            # transconv
            x1 = F.leaky_relu(x, LRELU_SLOPE)
            x1 = self.ups[i](x1)
            # repeat
            x2 = self.repeat_ups[i](x)
            x = x1 + x2
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        logger.info('Removing weight norm...')
        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.repeat_ups:
            layer[-1].remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()
