# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_deinterlace.models.archs import (DoubleConv,
                                                                 DownConv,
                                                                 TripleConv,
                                                                 UpCatConv)
from modelscope.models.cv.video_deinterlace.models.deep_fourier_upsampling import \
    freup_Periodicpadding


class DeinterlaceFre(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=3, ngf=64):
        """Defines a video deinterlace module.
           input a [b,c,h,w] tensor with range [0,1] as frame,
           it will output a [b,c,h,w] tensor with range [0,1] whitout interlace.

        Args:
            num_in_ch (int): Channel number of inputs. Default: 3.
            num_out_ch (int): Channel number of outputs. Default: 3.
            ngf(int): Channel number of features. Default: 64.
        """
        super(DeinterlaceFre, self).__init__()

        self.inconv = DoubleConv(num_in_ch, 48)
        self.down_0 = DownConv(48, 80)
        self.down_1 = DownConv(80, 144)

        self.opfre_0 = freup_Periodicpadding(80)
        self.opfre_1 = freup_Periodicpadding(144)

        self.conv_up1 = nn.Conv2d(80, ngf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(144, 80, 3, 1, 1)

        self.conv_hr = nn.Conv2d(ngf, ngf, 3, 1, 1)
        self.conv_last = nn.Conv2d(ngf, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.enh_inconv = DoubleConv(num_in_ch + num_out_ch, 48)
        # downsample
        self.enh_down_0 = DownConv(48, 80)
        self.enh_down_1 = DownConv(80, 144)
        self.enh_down_2 = DownConv(144, 256)
        self.enh_down_3 = DownConv(256, 448, num_conv=3)
        # upsample
        self.enh_up_3 = UpCatConv(704, 256)
        self.enh_up_2 = UpCatConv(400, 144)
        self.enh_up_1 = UpCatConv(224, 80)
        self.enh_up_0 = UpCatConv(128, 48)
        # extra convolutions
        self.enh_outconv = nn.Conv2d(48, num_out_ch, 3, 1, 1, bias=False)

    def interpolate(self, feat, x2, fn):
        x1f = fn(feat)
        x1 = F.interpolate(feat, scale_factor=2, mode='nearest')
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1f = F.pad(
            x1f,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x1 + x1f

    def forward(self, x):
        x1_0 = self.inconv(x)
        # downsample
        x1_1 = self.down_0(x1_0)  # 1/2
        x1_2 = self.down_1(x1_1)  # 1/4

        feat = self.lrelu(
            self.conv_up2(self.interpolate(x1_2, x1_1, self.opfre_1)))
        feat = self.lrelu(
            self.conv_up1(self.interpolate(feat, x1_0, self.opfre_0)))
        x_new = self.conv_last(self.lrelu(self.conv_hr(feat)))

        x2_0 = self.enh_inconv(torch.cat([x_new, x], 1))
        # downsample
        x2_1 = self.enh_down_0(x2_0)  # 1/2
        x2_2 = self.enh_down_1(x2_1)  # 1/4
        x2_3 = self.enh_down_2(x2_2)  # 1/8
        x2_4 = self.enh_down_3(x2_3)  # 1/16

        x2_5 = self.enh_up_3(x2_4, x2_3)  # 1/8
        x2_5 = self.enh_up_2(x2_5, x2_2)  # 1/4
        x2_5 = self.enh_up_1(x2_5, x2_1)  # 1/2
        x2_5 = self.enh_up_0(x2_5, x2_0)  # 1
        out = self.enh_outconv(x2_5)
        return out
