from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from .utils import get_padding, init_weights

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
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

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
        print('num_kernels={}, num_upsamples={}'.format(
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
        print('Removing weight norm...')
        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.repeat_ups:
            layer[-1].remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):

    def __init__(self,
                 period,
                 kernel_size=5,
                 stride=3,
                 use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    1,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    128, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    128,
                    512, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(
                Conv2d(
                    512,
                    1024, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

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
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        from pytorch_wavelets import DWT1DForward
        self.meanpools = nn.ModuleList(
            [DWT1DForward(wave='db3', J=1),
             DWT1DForward(wave='db3', J=1)])
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(2, 1, 15, 1, padding=7)),
            weight_norm(Conv1d(2, 1, 15, 1, padding=7))
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                yl, yh = self.meanpools[i - 1](y)
                y = torch.cat([yl, yh[0]], dim=1)
                y = self.convs[i - 1](y)
                y = F.leaky_relu(y, LRELU_SLOPE)

                yl_hat, yh_hat = self.meanpools[i - 1](y_hat)
                y_hat = torch.cat([yl_hat, yh_hat[0]], dim=1)
                y_hat = self.convs[i - 1](y_hat)
                y_hat = F.leaky_relu(y_hat, LRELU_SLOPE)

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorSTFT(torch.nn.Module):

    def __init__(self,
                 kernel_size=11,
                 stride=2,
                 use_spectral_norm=False,
                 fft_size=1024,
                 shift_size=120,
                 win_length=600,
                 window='hann_window'):
        super(DiscriminatorSTFT, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(
                Conv2d(
                    fft_size // 2 + 1,
                    32, (15, 1), (1, 1),
                    padding=(get_padding(15, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(9, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(9, 1), 0))),
            norm_f(
                Conv2d(
                    32,
                    32, (kernel_size, 1), (stride, 1),
                    padding=(get_padding(9, 1), 0))),
            norm_f(Conv2d(32, 32, (5, 1), (1, 1), padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(32, 1, (3, 1), (1, 1), padding=(1, 0)))
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, wav):
        wav = torch.squeeze(wav, 1)
        x_mag = stft(wav, self.fft_size, self.shift_size, self.win_length,
                     self.window)
        x = torch.transpose(x_mag, 2, 1).unsqueeze(-1)
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.squeeze(-1)

        return x, fmap


class MultiSTFTDiscriminator(torch.nn.Module):

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window='hann_window',
    ):
        super(MultiSTFTDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.discriminators += [
                DiscriminatorSTFT(fft_size=fs, shift_size=ss, win_length=wl)
            ]

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        temp_loss = torch.mean((1 - dg)**2)
        gen_losses.append(temp_loss)
        loss += temp_loss

    return loss, gen_losses
