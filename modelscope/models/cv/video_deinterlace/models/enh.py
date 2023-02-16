# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_deinterlace.models.archs import (DoubleConv,
                                                                 DownConv,
                                                                 TripleConv,
                                                                 UpCatConv)
from modelscope.models.cv.video_deinterlace.models.utils import warp


class DeinterlaceEnh(nn.Module):
    """Defines a U-Net video enhancement module

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
    """

    def __init__(self, num_in_ch=3, num_feat=64):
        super(DeinterlaceEnh, self).__init__()
        self.channel = num_in_ch

        # extra convolutions
        self.inconv2_1 = DoubleConv(num_in_ch * 3, 48)
        # downsample
        self.down2_0 = DownConv(48, 80)
        self.down2_1 = DownConv(80, 144)
        self.down2_2 = DownConv(144, 256)
        self.down2_3 = DownConv(256, 448, num_conv=3)
        # upsample
        self.up2_3 = UpCatConv(704, 256)
        self.up2_2 = UpCatConv(400, 144)
        self.up2_1 = UpCatConv(224, 80)
        self.up2_0 = UpCatConv(128, 48)
        # extra convolutions
        self.outconv2_1 = nn.Conv2d(48, num_in_ch, 3, 1, 1, bias=False)

        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(num_in_ch * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_in_ch * 2, kernel_size=3, padding=1))
        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(num_in_ch * 2, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(num_feat, num_in_ch * 2, kernel_size=3, padding=1))

    def forward(self, frames):
        frame1, frame2, frame3 = frames
        flow1 = self.offset_conv1(torch.cat([frame1, frame2], 1))
        warp1 = warp(frame1, flow1)
        flow3 = self.offset_conv2(torch.cat([frame3, frame2], 1))
        warp3 = warp(frame3, flow3)
        x2_0 = self.inconv2_1(torch.cat((warp1, frame2, warp3), 1))
        # downsample
        x2_1 = self.down2_0(x2_0)  # 1/2
        x2_2 = self.down2_1(x2_1)  # 1/4
        x2_3 = self.down2_2(x2_2)  # 1/8
        x2_4 = self.down2_3(x2_3)  # 1/16

        x2_5 = self.up2_3(x2_4, x2_3)  # 1/8
        x2_5 = self.up2_2(x2_5, x2_2)  # 1/4
        x2_5 = self.up2_1(x2_5, x2_1)  # 1/2
        x2_5 = self.up2_0(x2_5, x2_0)  # 1
        out_final = self.outconv2_1(x2_5)
        return out_final
