# The implementation is modified from torchvision
# under BSD-3-Clause License
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo


def load_weights_sequential(target, source_state, extra_chan=1):

    new_dict = OrderedDict()

    for k1, v1 in target.state_dict().items():
        if 'num_batches_tracked' not in k1:
            if k1 in source_state:
                tar_v = source_state[k1]

                if v1.shape != tar_v.shape:
                    # Init the new segmentation channel with zeros
                    # print(v1.shape, tar_v.shape)
                    c, _, w, h = v1.shape
                    pads = torch.zeros((c, extra_chan, w, h),
                                       device=tar_v.device)
                    nn.init.orthogonal_(pads)
                    tar_v = torch.cat([tar_v, pads], 1)

                new_dict[k1] = tar_v

    target.load_state_dict(new_dict, strict=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=True):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        bias=bias,
        dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 bias=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(
            inplanes, planes, stride=stride, dilation=dilation, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(
            planes, planes, stride=1, dilation=dilation, bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 dilation=1,
                 bias=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=bias)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=bias)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers=(3, 4, 23, 3), extra_chan=1, bias=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3 + extra_chan, 64, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bias=bias)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, bias=bias)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, bias=bias)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                try:
                    m.bias.data.zero_()
                except Exception:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    bias=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=bias),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation, bias)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, bias=bias))

        return nn.Sequential(*layers)


def resnet18(pretrained=True, extra_chan=0):
    model = ResNet(BasicBlock, [2, 2, 2, 2], extra_chan=extra_chan)
    if pretrained and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        load_weights_sequential(
            model,
            model_zoo.load_url(
                model_urls['resnet18'],
                model_dir='pretrain/resnet18-{}'.format(local_rank)),
            extra_chan)
    return model


def resnet50(pretrained=True, extra_chan=0):
    model = ResNet(Bottleneck, [3, 4, 6, 3], extra_chan=extra_chan, bias=False)

    if pretrained and torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        load_weights_sequential(
            model,
            model_zoo.load_url(
                model_urls['resnet50'],
                model_dir='pretrain/resnet50-{}'.format(local_rank)),
            extra_chan)
        print(torch.distributed.get_rank(), 'resnet 50 is loading...')
    return model
