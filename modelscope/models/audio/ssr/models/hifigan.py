# from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from torch.nn import Conv1d, ConvTranspose1d

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d

LRELU_SLOPE = 0.1


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """Sinusoid position encoding table"""

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    # gcd=Greatest Common Divisor
    subframe_length = math.gcd(frame_length, frame_step)
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    device_of_result = result.device
    result.index_add_(-2, frame.to(device_of_result), subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class LastLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear_activation, nonlinear_activation_params, pad, kernel_size, pad_params, bias):
        super(LastLayer, self).__init__()
        self.activation = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        x = self.activation(x)
        x = self.pad(x)
        x = self.conv(x)
        return x


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias)


class LastLinear(nn.Module):
    def __init__(self, hidden_channel, out_channel, bias=True):
        super(LastLinear, self).__init__()
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.bn_1 = nn.BatchNorm1d(hidden_channel)
        self.linear_1 = Conv1d1x1(hidden_channel, hidden_channel, bias=bias)
        self.bn_2 = nn.BatchNorm1d(hidden_channel)
        self.linear_2 = Conv1d1x1(hidden_channel, out_channel, bias=bias)

    def forward(self, x):
        x = self.activation(x)
        x = self.bn_1(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.bn_2(x)
        x = self.linear_2(x)
        return x


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module.
        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.
        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, C, F, T).
        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),
        """
        return F.interpolate(x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class UpsampleLayer(nn.Module):
    def __init__(self, in_channel, out_channel, upsample_rate, kernel_size, stride, padding, dilation=1, bias=True):
        super(UpsampleLayer, self).__init__()
        self.upsample = Stretch2d(upsample_rate, 1, mode="nearest")
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.upsample(x.unsqueeze(1))
        x = self.conv(x.squeeze(1))
        return x


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), bias=True):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]), bias=bias),
                Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]), bias=bias),
                Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]), bias=bias),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1), bias=bias),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1), bias=bias),
                Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1), bias=bias),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3), bias=True):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]), bias=bias),
                Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]), bias=bias),
            ]
        )

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class BasisSignalLayer(nn.Module):
    """Basis Signal"""

    def __init__(self, basis_signal_weight, L=64):
        super(BasisSignalLayer, self).__init__()
        self.layer = nn.Linear(basis_signal_weight.size(0), basis_signal_weight.size(1), bias=False)
        self.layer.weight = nn.Parameter(basis_signal_weight)
        self.L = L

    def forward(self, weight):
        source = self.layer(weight)
        source = overlap_and_add(source, self.L // 2)
        return source


"""Residual stack module in MelGAN."""


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, pad="ConstantPad1d", pad_params={"value": 0.0}):
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=True):
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_in).
        Returns:
            Tensor: Output tensor (B, out_channels, T_out).
        """
        return self.deconv(x)[:, :, : -self.stride]


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(
        self,
        kernel_size=3,
        channels=32,
        dilation=1,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_causal_conv=False,
    ):
        """Initialize ResidualStack module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(ResidualStack, self).__init__()

        # defile residual stack part
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
                torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(channels, channels, kernel_size, dilation=dilation, bias=bias, pad=pad, pad_params=pad_params),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        return self.stack(c) + self.skip_layer(c)


class HiFiGANGenerator(torch.nn.Module):
    def __init__(
        self,
        input_channels=80,
        resblock_kernel_sizes=[3, 7, 11],
        upsample_rates=[5, 4, 4, 2],
        upsample_initial_channel=256,
        resblock_type="1",
        upsample_kernel_sizes=[10, 8, 8, 4],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        transposedconv=True,
        weight_norm=True,
        bias=True,
    ):
        super(HiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(input_channels, upsample_initial_channel, 7, 1, padding=3, bias=bias)
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                UpsampleLayer(upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), upsample_rate=u, kernel_size=k, stride=1, padding=k // 2, bias=bias)
                if transposedconv == False
                else ConvTranspose1d(
                    upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), k, u, padding=(u // 2 + u % 2), output_padding=u % 2, bias=bias
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, bias=bias))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=bias)
        # apply weight norm
        if weight_norm:
            self.apply_weight_norm()
        # reset parameters
        self.reset_parameters()

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        # x = torch.tanh(x)

        return x

    def inference(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
        x = x.transpose(1, 0).unsqueeze(0)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        # x = torch.tanh(x)

        return x


if __name__ == "__main__":
    import thop

    layer = HiFiGANGenerator(input_channels=256, upsample_initial_channel=256, upsample_rates=[4, 4, 4, 5], upsample_kernel_sizes=[8, 8, 8, 10])
    a = torch.randn([1, 256, 50])
    b = layer(a)

    fp, p = thop.profile(layer, [a])
    print(b.shape)
    print(fp / 1024 / 1024 / 1024)
    print(p / 1024)
    count = 0
    for p in layer.parameters():
        count += p.numel()
    print(count)
