import collections
import math
import sys

import torch
from pytorch_wavelets import DWTForward, DWTInverse
from torch import kl_div, nn
from torch.nn import functional as F

from modelscope.ops.human_image_generation.fused_act import (FusedLeakyReLU,
                                                             fused_leaky_relu)
from modelscope.ops.human_image_generation.upfirdn2d import upfirdn2d
from .conv2d_gradfix import conv2d, conv_transpose2d
from .wavelet_module import *


# add flow
class ExtractionOperation_flow(nn.Module):

    def __init__(self, in_channel, num_label, match_kernel):
        super(ExtractionOperation_flow, self).__init__()
        self.value_conv = EqualConv2d(
            in_channel,
            in_channel,
            match_kernel,
            1,
            match_kernel // 2,
            bias=True)
        self.semantic_extraction_filter = EqualConv2d(
            in_channel,
            num_label,
            match_kernel,
            1,
            match_kernel // 2,
            bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.num_label = num_label

    def forward(self, value, recoder):
        key = value
        b, c, h, w = value.shape
        key = self.semantic_extraction_filter(self.feature_norm(key))
        extraction_softmax = self.softmax(key.view(b, -1, h * w))
        values_flatten = self.value_conv(value).view(b, -1, h * w)
        neural_textures = torch.einsum('bkm,bvm->bvk', extraction_softmax,
                                       values_flatten)
        recoder['extraction_softmax'].insert(0, extraction_softmax)
        recoder['neural_textures'].insert(0, neural_textures)
        return neural_textures, extraction_softmax

    def feature_norm(self, input_tensor):
        input_tensor = input_tensor - input_tensor.mean(dim=1, keepdim=True)
        norm = torch.norm(
            input_tensor, 2, 1, keepdim=True) + sys.float_info.epsilon
        out = torch.div(input_tensor, norm)
        return out


class DistributionOperation_flow(nn.Module):

    def __init__(self, num_label, input_dim, match_kernel=3):
        super(DistributionOperation_flow, self).__init__()
        self.semantic_distribution_filter = EqualConv2d(
            input_dim,
            num_label,
            kernel_size=match_kernel,
            stride=1,
            padding=match_kernel // 2)
        self.num_label = num_label

    def forward(self, query, extracted_feature, recoder):
        b, c, h, w = query.shape

        query = self.semantic_distribution_filter(query)
        query_flatten = query.view(b, self.num_label, -1)
        query_softmax = F.softmax(query_flatten, 1)
        values_q = torch.einsum('bkm,bkv->bvm', query_softmax,
                                extracted_feature.permute(0, 2, 1))
        attn_out = values_q.view(b, -1, h, w)
        recoder['semantic_distribution'].append(query)
        return attn_out


class EncoderLayer_flow(nn.Sequential):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 downsample=False,
                 blur_kernel=[1, 3, 3, 1],
                 bias=True,
                 activate=True,
                 use_extraction=False,
                 num_label=None,
                 match_kernel=None,
                 num_extractions=2):
        super().__init__()

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

            stride = 2
            padding = 0

        else:
            self.blur = None
            stride = 1
            padding = kernel_size // 2

        self.conv = EqualConv2d(
            in_channel,
            out_channel,
            kernel_size,
            padding=padding,
            stride=stride,
            bias=bias and not activate,
        )

        self.activate = FusedLeakyReLU(
            out_channel, bias=bias) if activate else None
        self.use_extraction = use_extraction
        if self.use_extraction:
            self.extraction_operations = nn.ModuleList()
            for _ in range(num_extractions):
                self.extraction_operations.append(
                    ExtractionOperation_flow(out_channel, num_label,
                                             match_kernel))

    def forward(self, input, recoder=None):
        out = self.blur(input) if self.blur is not None else input
        out = self.conv(out)
        out = self.activate(out) if self.activate is not None else out
        if self.use_extraction:
            for extraction_operation in self.extraction_operations:
                extraction_operation(out, recoder)
        return out


class DecoderLayer_flow_wavelet_fuse24(nn.Module):

    # add fft refinement and tps

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        use_distribution=True,
        num_label=16,
        match_kernel=3,
        wavelet_down_level=False,
        window_size=8,
    ):
        super().__init__()
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(
                blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.conv = EqualTransposeConv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=2,
                padding=0,
                bias=bias and not activate,
            )
        else:
            self.conv = EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=bias and not activate,
            )
            self.blur = None

        self.distribution_operation = DistributionOperation_flow(
            num_label, out_channel,
            match_kernel=match_kernel) if use_distribution else None
        self.activate = FusedLeakyReLU(
            out_channel, bias=bias) if activate else None
        self.use_distribution = use_distribution

        # mask prediction network
        if use_distribution:
            self.conv_mask_lf = nn.Sequential(*[
                EqualConv2d(
                    out_channel, 1, 3, stride=1, padding=3 // 2, bias=False),
                nn.Sigmoid()
            ])
            self.conv_mask_dict = nn.ModuleDict()
            for level in range(wavelet_down_level):
                conv_mask = nn.Sequential(*[
                    EqualConv2d(
                        out_channel,
                        1,
                        3,
                        stride=1,
                        padding=3 // 2,
                        bias=False),
                    nn.Sigmoid()
                ])
                self.conv_mask_dict[str(level)] = conv_mask

        self.wavelet_down_level = wavelet_down_level
        if wavelet_down_level:
            self.dwt = DWTForward(
                J=self.wavelet_down_level, mode='zero', wave='haar')
            self.idwt = DWTInverse(mode='zero', wave='haar')

            # for mask input channel squeeze and expand
            self.conv_l_squeeze = EqualConv2d(
                2 * out_channel, out_channel, 1, 1, 0, bias=False)
            self.conv_h_squeeze = EqualConv2d(
                6 * out_channel, out_channel, 1, 1, 0, bias=False)

            self.conv_l = EqualConv2d(
                out_channel, out_channel, 3, 1, 3 // 2, bias=False)

            self.hf_modules = nn.ModuleDict()
            for level in range(wavelet_down_level):
                hf_module = nn.Module()
                prev_channel = out_channel if level == self.wavelet_down_level - 1 else 3 * out_channel
                hf_module.conv_prev = EqualConv2d(
                    prev_channel, 3 * out_channel, 3, 1, 3 // 2, bias=False)
                hf_module.conv_hf = GatedConv2dWithActivation(
                    3 * out_channel, 3 * out_channel, 3, 1, 3 // 2, bias=False)
                hf_module.conv_out = GatedConv2dWithActivation(
                    3 * out_channel, 3 * out_channel, 3, 1, 3 // 2, bias=False)
                self.hf_modules[str(level)] = hf_module

        self.amp_fuse = nn.Sequential(
            EqualConv2d(2 * out_channel, out_channel, 1, 1, 0),
            FusedLeakyReLU(out_channel, bias=False),
            EqualConv2d(out_channel, out_channel, 1, 1, 0))
        self.pha_fuse = nn.Sequential(
            EqualConv2d(2 * out_channel, out_channel, 1, 1, 0),
            FusedLeakyReLU(out_channel, bias=False),
            EqualConv2d(out_channel, out_channel, 1, 1, 0))
        self.post = EqualConv2d(out_channel, out_channel, 1, 1, 0)
        self.eps = 1e-8

    def forward(self,
                input,
                neural_texture=None,
                recoder=None,
                warped_texture=None,
                style_net=None,
                gstyle=None):
        out = self.conv(input)
        out = self.blur(out) if self.blur is not None else out

        mask_l, mask_h = None, None
        out_attn = None
        if self.use_distribution and neural_texture is not None:
            out_ori = out
            out_attn = self.distribution_operation(out, neural_texture,
                                                   recoder)
            # wavelet fusion
            if self.wavelet_down_level:
                assert out.shape[2] % 2 == 0, \
                    f'out shape {out.shape} is not appropriate for processing'
                b, c, h, w = out.shape

                # wavelet decomposition
                LF_attn, HF_attn = self.dwt(out_attn)
                LF_warp, HF_warp = self.dwt(warped_texture)
                LF_out, HF_out = self.dwt(out)

                # generate mask
                hf_dict = {}
                l_mask_input = torch.cat([LF_attn, LF_warp], dim=1)
                l_mask_input = self.conv_l_squeeze(l_mask_input)
                l_mask_input = style_net(l_mask_input, gstyle)
                ml = self.conv_mask_lf(l_mask_input)
                mask_l = ml

                for level in range(self.wavelet_down_level):
                    # level up, feature size down
                    scale = 2**(level + 1)
                    hfa = HF_attn[level].view(b, c * 3, h // scale, w // scale)
                    hfw = HF_warp[level].view(b, c * 3, h // scale, w // scale)
                    hfg = HF_out[level].view(b, c * 3, h // scale, w // scale)

                    h_mask_input = torch.cat([hfa, hfw], dim=1)
                    h_mask_input = self.conv_h_squeeze(h_mask_input)
                    h_mask_input = style_net(h_mask_input, gstyle)
                    mh = self.conv_mask_dict[str(level)](h_mask_input)
                    if level == 0:
                        mask_h = mh

                    # fuse high frequency
                    xh = (mh * hfa + (1 - mh) * hfw + hfg) / math.sqrt(2)
                    hf_dict[str(level)] = xh

                temp_result = (1 - ml) * LF_warp + LF_out
                out_l = (ml * LF_attn + temp_result) / math.sqrt(2)
                out_h_list = []
                for level in range(self.wavelet_down_level - 1, -1, -1):
                    xh = hf_dict[str(level)]
                    b, c, h, w = xh.shape
                    out_h_list.append(xh.view(b, c // 3, 3, h, w))
                out_h_list = (
                    out_h_list)[::-1]  # the h list from large to small size
                #
                out = self.idwt((out_l, out_h_list))
            else:
                out = (out + out_attn) / math.sqrt(2)

            # fourier refinement
            _, _, H, W = out.shape
            fuseF = torch.fft.rfft2(out + self.eps, norm='backward')
            outF = torch.fft.rfft2(out_ori + self.eps, norm='backward')
            amp = self.amp_fuse(
                torch.cat([torch.abs(fuseF), torch.abs(outF)], 1))
            pha = self.pha_fuse(
                torch.cat(
                    [torch.angle(fuseF), torch.angle(outF)], 1))
            out_fft = torch.fft.irfft2(
                amp * torch.exp(1j * pha) + self.eps,
                s=(H, W),
                dim=(-2, -1),
                norm='backward')

            out = out + self.post(out_fft)

        out = self.activate(
            out.contiguous()) if self.activate is not None else out
        return out, mask_h, mask_l


# base functions


class EqualConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True,
                 dilation=1):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualTransposeConv2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        weight = self.weight.transpose(0, 1)
        out = conv_transpose2d(
            input,
            weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ToRGB(nn.Module):

    def __init__(self, in_channel, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = EqualConv2d(in_channel, 3, 3, stride=1, padding=1)

    def forward(self, input, skip=None):
        out = self.conv(input)
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out


class EqualLinear(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 bias_init=0,
                 lr_mul=1,
                 activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class Upsample(nn.Module):

    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor**2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(
            input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class ResBlock(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 blur_kernel=[1, 3, 3, 1],
                 downsample=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(
            in_channel, out_channel, 3, downsample=downsample)

        self.skip = ConvLayer(
            in_channel,
            out_channel,
            1,
            downsample=downsample,
            activate=False,
            bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConvLayer(nn.Sequential):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            ))

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class Blur(nn.Module):

    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class GatedConv2dWithActivation(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 activation=None):
        super(GatedConv2dWithActivation, self).__init__()
        self.activation = FusedLeakyReLU(out_channels, bias=False)
        self.conv2d = EqualConv2d(in_channels, out_channels, kernel_size,
                                  stride, padding, bias, dilation)
        self.mask_conv2d = EqualConv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, bias, dilation)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)

        return x


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class SPDNorm(nn.Module):

    def __init__(self,
                 norm_channel,
                 label_nc,
                 norm_type='position',
                 use_equal=False):
        super().__init__()
        param_free_norm_type = norm_type
        ks = 3
        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(
                norm_channel, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == 'position':
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError(
                '%s is not a recognized param-free norm type in SPADE'
                % param_free_norm_type)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        pw = ks // 2
        nhidden = 128
        if not use_equal:
            self.mlp_activate = nn.Sequential(
                nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                nn.ReLU())
            self.mlp_gamma = nn.Conv2d(
                nhidden, norm_channel, kernel_size=ks, padding=pw)
            self.mlp_beta = nn.Conv2d(
                nhidden, norm_channel, kernel_size=ks, padding=pw)
        else:
            self.mlp_activate = nn.Sequential(*[
                EqualConv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
                FusedLeakyReLU(nhidden, bias=False)
            ])
            self.mlp_gamma = EqualConv2d(
                nhidden, norm_channel, kernel_size=ks, padding=pw)
            self.mlp_beta = EqualConv2d(
                nhidden, norm_channel, kernel_size=ks, padding=pw)

    def forward(self, x, prior_f, weight=1.0):
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on condition feature
        actv = self.mlp_activate(prior_f)
        gamma = self.mlp_gamma(actv) * weight
        beta = self.mlp_beta(actv) * weight
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output
