# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.nn.utils import remove_weight_norm, weight_norm

from modelscope.models.audio.tts.kantts.models.utils import init_weights


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Conv1d(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
    ):
        super(Conv1d, self).__init__()
        self.conv1d = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ))
        self.conv1d.apply(init_weights)

    def forward(self, x):
        x = self.conv1d(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


class CausalConv1d(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
    ):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1d = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            ))
        self.conv1d.apply(init_weights)

    def forward(self, x):  # bdt
        x = F.pad(
            x, (self.pad, 0, 0, 0, 0, 0), 'constant'
        )  # described starting from the last dimension and moving forward.
        #  x = F.pad(x, (self.pad, self.pad, 0, 0, 0, 0), "constant")
        x = self.conv1d(x)[:, :, :x.size(2)]
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1d)


class ConvTranspose1d(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
    ):
        super(ConvTranspose1d, self).__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                output_padding=0,
            ))
        self.deconv.apply(init_weights)

    def forward(self, x):
        return self.deconv(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


#  FIXME: HACK to get shape right
class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        output_padding=0,
    ):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = weight_norm(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=0,
                output_padding=0,
            ))
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
        #  x = F.pad(x, (self.pad, 0, 0, 0, 0, 0), "constant")
        return self.deconv(x)[:, :, :-self.pad]
        #  return self.deconv(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.deconv)


class ResidualBlock(torch.nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        nonlinear_activation='LeakyReLU',
        nonlinear_activation_params={'negative_slope': 0.1},
        causal=False,
    ):
        super(ResidualBlock, self).__init__()
        assert kernel_size % 2 == 1, 'Kernal size must be odd number.'
        conv_cls = CausalConv1d if causal else Conv1d
        self.convs1 = nn.ModuleList([
            conv_cls(
                channels,
                channels,
                kernel_size,
                1,
                dilation=dilation[i],
                padding=get_padding(kernel_size, dilation[i]),
            ) for i in range(len(dilation))
        ])

        self.convs2 = nn.ModuleList([
            conv_cls(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            ) for i in range(len(dilation))
        ])

        self.activation = getattr(
            torch.nn, nonlinear_activation)(**nonlinear_activation_params)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = c1(xt)
            xt = self.activation(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class SourceModule(torch.nn.Module):

    def __init__(self,
                 nb_harmonics,
                 upsample_ratio,
                 sampling_rate,
                 alpha=0.1,
                 sigma=0.003):
        super(SourceModule, self).__init__()

        self.nb_harmonics = nb_harmonics
        self.upsample_ratio = upsample_ratio
        self.sampling_rate = sampling_rate
        self.alpha = alpha
        self.sigma = sigma

        self.ffn = nn.Sequential(
            weight_norm(
                nn.Conv1d(self.nb_harmonics + 1, 1, kernel_size=1, stride=1)),
            nn.Tanh(),
        )

    def forward(self, pitch, uv):
        """
        :param pitch: [B, 1, frame_len], Hz
        :param uv: [B, 1, frame_len] vuv flag
        :return: [B, 1, sample_len]
        """
        with torch.no_grad():
            pitch_samples = F.interpolate(
                pitch, scale_factor=(self.upsample_ratio), mode='nearest')
            uv_samples = F.interpolate(
                uv, scale_factor=(self.upsample_ratio), mode='nearest')

            F_mat = torch.zeros(
                (pitch_samples.size(0), self.nb_harmonics + 1,
                 pitch_samples.size(-1))).to(pitch_samples.device)
            for i in range(self.nb_harmonics + 1):
                F_mat[:, i:i
                      + 1, :] = pitch_samples * (i + 1) / self.sampling_rate

            theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
            u_dist = Uniform(low=-np.pi, high=np.pi)
            phase_vec = u_dist.sample(
                sample_shape=(pitch.size(0), self.nb_harmonics + 1,
                              1)).to(F_mat.device)
            phase_vec[:, 0, :] = 0

            n_dist = Normal(loc=0.0, scale=self.sigma)
            noise = n_dist.sample(
                sample_shape=(
                    pitch_samples.size(0),
                    self.nb_harmonics + 1,
                    pitch_samples.size(-1),
                )).to(F_mat.device)

            e_voice = self.alpha * torch.sin(theta_mat + phase_vec) + noise
            e_unvoice = self.alpha / 3 / self.sigma * noise

            e = e_voice * uv_samples + e_unvoice * (1 - uv_samples)

        return self.ffn(e)

    def remove_weight_norm(self):
        remove_weight_norm(self.ffn[0])
