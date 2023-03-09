# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlurPool(nn.Module):

    def __init__(self,
                 channels,
                 pad_type='reflect',
                 filt_size=4,
                 stride=2,
                 pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2)),
            int(1. * (filt_size - 1) / 2),
            int(np.ceil(1. * (filt_size - 1) / 2))
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([
                1.,
            ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer(
            'filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if (self.filt_size == 1):
            if (self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(
                self.pad(inp),
                self.filt,
                stride=self.stride,
                groups=inp.shape[1])


def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class ConvBlockv1(nn.Module):

    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlockv1, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            int(out_planes / 2),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.Conv2d(
            int(out_planes / 2),
            int(out_planes / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv3 = nn.Conv2d(
            int(out_planes / 4),
            int(out_planes / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        if norm == 'batch':
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(out_planes)
        elif norm == 'group':
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, out_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=1,
                    bias=False), )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)
        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)
        out3 += residual
        out4 = self.bn4(out3)
        out4 = F.relu(out4, True)
        return out4


class Conv2(nn.Module):

    def __init__(self, in_planes, out_planes, norm='batch'):
        super(Conv2, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            int(out_planes / 4),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.conv2 = nn.Conv2d(
            in_planes,
            int(out_planes / 4),
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False)
        self.conv3 = nn.Conv2d(
            in_planes,
            int(out_planes / 2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.conv4 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn2 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn4 = nn.BatchNorm2d(out_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn2 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn4 = nn.GroupNorm(32, out_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=1,
                    bias=False), )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1, True)

        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        out2 = F.relu(out2, True)

        out3 = self.conv3(x)
        out3 = self.bn3(out3)
        out3 = F.relu(out3, True)
        out3 = torch.cat((out1, out2, out3), 1)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = out3 + residual
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out, True)
        return out


class Res_hournet(nn.Module):

    def __init__(self, norm: str = 'group', use_front=False, use_back=False):
        """
        Defines a backbone of human reconstruction
        use_front & use_back is the normal map of input image
        """
        super(Res_hournet, self).__init__()
        self.name = 'Res Backbone'
        self.norm = norm
        inc = 3
        self.use_front = use_front
        self.use_back = use_back
        if self.use_front:
            inc += 3
        if self.use_back:
            inc += 3
        self.conv1 = nn.Conv2d(inc, 64, kernel_size=7, stride=1, padding=3)
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)
        self.down_conv1 = BlurPool(
            64, pad_type='reflect', filt_size=7, stride=2)
        self.conv2 = ConvBlockv1(64, 128, self.norm)
        self.down_conv2 = BlurPool(
            128, pad_type='reflect', filt_size=7, stride=2)
        self.conv3 = ConvBlockv1(128, 128, self.norm)
        self.conv5 = ConvBlockv1(128, 256, self.norm)
        self.conv6 = ConvBlockv1(256, 256, self.norm)
        self.down_conv3 = BlurPool(
            256, pad_type='reflect', filt_size=5, stride=2)
        self.conv7 = ConvBlockv1(256, 256, self.norm)
        self.conv8 = ConvBlockv1(256, 256, self.norm)
        self.conv9 = ConvBlockv1(256, 256, self.norm)
        self.conv10 = ConvBlockv1(256, 256, self.norm)
        self.conv10_1 = ConvBlockv1(256, 512, self.norm)
        self.conv10_2 = Conv2(512, 512, self.norm)
        self.down_conv4 = BlurPool(
            512, pad_type='reflect', filt_size=5, stride=2)
        self.conv11 = Conv2(512, 512, self.norm)
        self.conv12 = ConvBlockv1(512, 512, self.norm)
        self.conv13 = Conv2(512, 512, self.norm)
        self.conv14 = ConvBlockv1(512, 512, self.norm)
        self.conv15 = Conv2(512, 512, self.norm)
        self.conv16 = ConvBlockv1(512, 512, self.norm)
        self.conv17 = Conv2(512, 512, self.norm)
        self.conv18 = ConvBlockv1(512, 512, self.norm)
        self.conv19 = Conv2(512, 512, self.norm)
        self.conv20 = ConvBlockv1(512, 512, self.norm)
        self.conv21 = Conv2(512, 512, self.norm)
        self.conv22 = ConvBlockv1(512, 512, self.norm)

        self.up_down1 = nn.Conv2d(1024, 512, 3, 1, 1, bias=False)
        self.upconv1 = ConvBlockv1(512, 512, self.norm)
        self.upconv1_1 = ConvBlockv1(512, 512, self.norm)
        self.up_down2 = nn.Conv2d(768, 512, 3, 1, 1, bias=False)
        self.upconv2 = ConvBlockv1(512, 256, self.norm)
        self.upconv2_1 = ConvBlockv1(256, 256, self.norm)
        self.up_down3 = nn.Conv2d(384, 256, 3, 1, 1, bias=False)
        self.upconv3 = ConvBlockv1(256, 256, self.norm)
        self.upconv3_4 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.up_down4 = nn.Conv2d(192, 64, 3, 1, 1, bias=False)
        self.upconv4 = ConvBlockv1(64, 64, 'batch')

    def forward(self, x):
        out0 = self.bn1(self.conv1(x))
        out1 = self.down_conv1(out0)
        out1 = self.conv2(out1)
        out2 = self.down_conv2(out1)
        out2 = self.conv3(out2)
        out2 = self.conv5(out2)
        out2 = self.conv6(out2)
        out3 = self.down_conv3(out2)
        out3 = self.conv7(out3)
        out3 = self.conv9(self.conv8(out3))
        out3 = self.conv10(out3)
        out3 = self.conv10_2(self.conv10_1(out3))
        out4 = self.down_conv4(out3)
        out4 = self.conv12(self.conv11(out4))
        out4 = self.conv14(self.conv13(out4))
        out4 = self.conv16(self.conv15(out4))
        out4 = self.conv18(self.conv17(out4))
        out4 = self.conv20(self.conv19(out4))
        out4 = self.conv22(self.conv21(out4))

        up1 = F.interpolate(
            out4, scale_factor=2, mode='bicubic', align_corners=True)
        up1 = torch.cat((up1, out3), 1)
        up1 = self.up_down1(up1)
        up1 = self.upconv1(up1)
        up1 = self.upconv1_1(up1)

        up2 = F.interpolate(
            up1, scale_factor=2, mode='bicubic', align_corners=True)
        up2 = torch.cat((up2, out2), 1)
        up2 = self.up_down2(up2)
        up2 = self.upconv2(up2)
        up2 = self.upconv2_1(up2)

        up3 = F.interpolate(
            up2, scale_factor=2, mode='bicubic', align_corners=True)
        up3 = torch.cat((up3, out1), 1)
        up3 = self.up_down3(up3)
        up3 = self.upconv3(up3)

        up34 = self.upconv3_4(up3)
        up4 = F.interpolate(
            up34, scale_factor=2, mode='bicubic', align_corners=True)
        up4 = torch.cat((up4, out0), 1)
        up4 = self.up_down4(up4)
        up4 = self.upconv4(up4)
        return up3, up4
