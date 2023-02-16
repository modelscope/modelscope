# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
from torch.nn import functional as F

INPUT_SIZE = 224


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup), nn.PReLU(oup))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 stride,
                 padding,
                 use_res_connect,
                 expand_ratio=6):

        super(InvertedResidual, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect
        hid_channels = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.PReLU(hid_channels),
            nn.Conv2d(
                hid_channels,
                hid_channels,
                3,
                stride,
                padding,
                groups=hid_channels,
                bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.PReLU(hid_channels),
            nn.Conv2d(hid_channels, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SoftArgmax(nn.Module):

    def __init__(self, beta: int = 1, infer=False):
        if not 0.0 <= beta:
            raise ValueError(f'Invalid beta: {beta}')
        super().__init__()
        self.beta = beta
        self.infer = infer

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        heatmap = heatmap.mul(self.beta)
        batch_size, num_channel, height, width = heatmap.size()
        device: str = heatmap.device

        if not self.infer:
            softmax: torch.Tensor = F.softmax(
                heatmap.view(batch_size, num_channel, height * width),
                dim=2).view(batch_size, num_channel, height, width)

            xx, yy = torch.meshgrid(list(map(torch.arange, [width, height])))

            approx_x = (
                softmax.mul(xx.float().to(device)).view(
                    batch_size, num_channel,
                    height * width).sum(2).unsqueeze(2))
            approx_y = (
                softmax.mul(yy.float().to(device)).view(
                    batch_size, num_channel,
                    height * width).sum(2).unsqueeze(2))

            output = [approx_x / width, approx_y / height]
            output = torch.cat(output, 2)
            output = output.view(-1, output.size(1) * output.size(2))
            return output
        else:
            softmax: torch.Tensor = F.softmax(
                heatmap.view(batch_size, num_channel, height * width), dim=2)

            return softmax


class LargeBaseLmksNet(nn.Module):

    def __init__(self, er=1.0, infer=False):

        super(LargeBaseLmksNet, self).__init__()

        self.infer = infer

        self.block1 = conv_bn(3, int(64 * er), 3, 2, 1)
        self.block2 = InvertedResidual(
            int(64 * er), int(64 * er), 1, 1, False, 2)

        self.block3 = InvertedResidual(
            int(64 * er), int(64 * er), 2, 1, False, 2)
        self.block4 = InvertedResidual(
            int(64 * er), int(64 * er), 1, 1, True, 2)
        self.block5 = InvertedResidual(
            int(64 * er), int(64 * er), 1, 1, True, 2)
        self.block6 = InvertedResidual(
            int(64 * er), int(64 * er), 1, 1, True, 2)
        self.block7 = InvertedResidual(
            int(64 * er), int(64 * er), 1, 1, True, 2)

        self.block8 = InvertedResidual(
            int(64 * er), int(128 * er), 2, 1, False, 2)

        self.block9 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, False, 4)
        self.block10 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, True, 4)
        self.block11 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, True, 4)
        self.block12 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, True, 4)
        self.block13 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, True, 4)
        self.block14 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, True, 4)

        self.block15 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, False, 2)  # [128, 14, 14]
        self.block16 = InvertedResidual(
            int(128 * er), int(128 * er), 2, 1, False, 2)
        self.block17 = InvertedResidual(
            int(128 * er), int(128 * er), 1, 1, False, 2)

        self.block18 = conv_bn(int(128 * er), int(256 * er), 3, 1, 1)
        self.block19 = nn.Conv2d(int(256 * er), 106, 3, 1, 1, bias=False)
        self.softargmax = SoftArgmax(infer=infer)

    def forward(self, x):  # x: 3, 224, 224

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.softargmax(x)

        return x
