# Implementation in this file is modified based on deeplabv3
# Originally MIT license,publicly avaialbe at https://github.com/fregu856/deeplabv3/blob/master/model/aspp.py
# Implementation in this file is modified based on attention-module
# Originally MIT license,publicly avaialbe at https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias), nn.BatchNorm2d(planes), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ASPP(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ASPP, self).__init__()
        mid_dim = 128
        self.conv1 = ConvBNReLU(in_dim, mid_dim, kernel_size=1)
        self.conv2 = ConvBNReLU(
            in_dim, mid_dim, kernel_size=3, padding=2, dilation=2)
        self.conv3 = ConvBNReLU(
            in_dim, mid_dim, kernel_size=3, padding=5, dilation=5)
        self.conv4 = ConvBNReLU(
            in_dim, mid_dim, kernel_size=3, padding=7, dilation=7)
        self.conv5 = ConvBNReLU(in_dim, mid_dim, kernel_size=1, padding=0)
        self.fuse = ConvBNReLU(5 * mid_dim, out_dim, 3, 1, 1)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        xg = self.conv5(self.global_pooling(x))
        conv5 = nn.Upsample((x.shape[2], x.shape[3]), mode='nearest')(xg)
        return self.fuse(torch.cat((conv1, conv2, conv3, conv4, conv5), 1))


class ChannelAttention(nn.Module):

    def __init__(self, inchs, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inchs, inchs // 16, 1, bias=False), nn.ReLU(),
            nn.Conv2d(inchs // 16, inchs, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):

    def __init__(self, inchs, kernel_size=7):
        super().__init__()
        self.calayer = ChannelAttention(inchs=inchs)
        self.saLayer = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        xca = self.calayer(x) * x
        xsa = self.saLayer(xca) * xca
        return xsa
