# Copyright (c) Alibaba, Inc. and its affiliates.
#
# The implementation of class ComplexConv2d, ComplexConvTranspose2d and
# ComplexBatchNorm2d here is modified based on Jongho Choi(sweetcocoa@snu.ac.kr
# / Seoul National Univ., ESTsoft ) and publicly available at
# https://github.com/sweetcocoa/DeepComplexUNetPyTorch

import torch
import torch.nn as nn

from modelscope.models.audio.ans.layers.uni_deep_fsmn import UniDeepFsmn


class ComplexUniDeepFsmn(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn, self).__init__()

        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_re_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)
        self.fsmn_im_L2 = UniDeepFsmn(nHidden, nOut, 20, nHidden)

    def forward(self, x):
        r"""

        Args:
            x: torch with shape [batch, channel, feature, sequence, 2], eg: [6, 256, 1, 106, 2]

        Returns:
            [batch, feature, sequence, 2], eg: [6, 99, 1024, 2]
        """
        #
        b, c, h, T, d = x.size()
        x = torch.reshape(x, (b, c * h, T, d))
        # x: [b,h,T,2], [6, 256, 106, 2]
        x = torch.transpose(x, 1, 2)
        # x: [b,T,h,2], [6, 106, 256, 2]

        real_L1 = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary_L1 = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # GRU output: [99, 6, 128]
        real = self.fsmn_re_L2(real_L1) - self.fsmn_im_L2(imaginary_L1)
        imaginary = self.fsmn_re_L2(imaginary_L1) + self.fsmn_im_L2(real_L1)
        # output: [b,T,h,2], [99, 6, 1024, 2]
        output = torch.stack((real, imaginary), dim=-1)

        # output: [b,h,T,2], [6, 99, 1024, 2]
        output = torch.transpose(output, 1, 2)
        output = torch.reshape(output, (b, c, h, T, d))

        return output


class ComplexUniDeepFsmn_L1(nn.Module):

    def __init__(self, nIn, nHidden=128, nOut=128):
        super(ComplexUniDeepFsmn_L1, self).__init__()
        self.fsmn_re_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)
        self.fsmn_im_L1 = UniDeepFsmn(nIn, nHidden, 20, nHidden)

    def forward(self, x):
        r"""

        Args:
            x: torch with shape [batch, channel, feature, sequence, 2], eg: [6, 256, 1, 106, 2]
        """
        b, c, h, T, d = x.size()
        # x : [b,T,h,c,2]
        x = torch.transpose(x, 1, 3)
        x = torch.reshape(x, (b * T, h, c, d))

        real = self.fsmn_re_L1(x[..., 0]) - self.fsmn_im_L1(x[..., 1])
        imaginary = self.fsmn_re_L1(x[..., 1]) + self.fsmn_im_L1(x[..., 0])
        # output: [b*T,h,c,2], [6*106, h, 256, 2]
        output = torch.stack((real, imaginary), dim=-1)

        output = torch.reshape(output, (b, T, h, c, d))
        output = torch.transpose(output, 1, 3)
        return output


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 **kwargs):
        super().__init__()

        # Model components
        self.conv_re = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs)
        self.conv_im = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs)

    def forward(self, x):
        r"""

        Args:
            x: torch with shape: [batch,channel,axis1,axis2,2]
        """
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexConvTranspose2d(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 **kwargs):
        super().__init__()

        # Model components
        self.tconv_re = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            **kwargs)
        self.tconv_im = nn.ConvTranspose2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            **kwargs)

    def forward(self, x):  # shpae of x : [batch,channel,axis1,axis2,2]
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True,
                 **kwargs):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs)
        self.bn_im = nn.BatchNorm2d(
            num_features=num_features,
            momentum=momentum,
            affine=affine,
            eps=eps,
            track_running_stats=track_running_stats,
            **kwargs)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output
