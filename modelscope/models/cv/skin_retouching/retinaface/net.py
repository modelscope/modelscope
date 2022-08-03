# Implementation in this file is modifed from source code avaiable via https://github.com/ternaus/retinaface
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(inp: int,
            oup: int,
            stride: int = 1,
            leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp: int, oup: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp: int,
               oup: int,
               stride: int,
               leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp: int,
            oup: int,
            stride: int,
            leaky: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        if out_channel % 4 != 0:
            raise ValueError(
                f'Expect out channel % 4 == 0, but we got {out_channel % 4}')

        leaky: float = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(
            in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(
            out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(
            out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)

        return F.relu(out)


class FPN(nn.Module):

    def __init__(self, in_channels_list: List[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.0
        if out_channels <= 64:
            leaky = 0.1

        self.output1 = conv_bn1X1(
            in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(
            in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(
            in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        y = list(x.values())

        output1 = self.output1(y[0])
        output2 = self.output2(y[1])
        output3 = self.output3(y[2])

        up3 = F.interpolate(
            output3, size=[output2.size(2), output2.size(3)], mode='nearest')
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(
            output2, size=[output1.size(2), output1.size(3)], mode='nearest')
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]
