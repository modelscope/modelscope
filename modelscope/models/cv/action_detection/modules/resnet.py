# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
from detectron2.modeling import Backbone


def conv1x3x3(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=stride,
        padding=(0, 1, 1),
        bias=False)


def conv3x3x3(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 3),
        stride=stride,
        padding=(1, 1, 1),
        bias=False)


def conv1x1x1(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x1x1(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 1, 1),
        stride=stride,
        padding=(1, 0, 0),
        bias=False)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 op='c2d',
                 downsample=None,
                 base_width=64,
                 norm_layer=None):
        super(BasicBlock3D, self).__init__()
        dilation = 1
        groups = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        stride = [stride] * 3 if isinstance(stride, int) else stride
        self.t_stride = stride[0]
        self.stride = stride
        stride = [1] + list(stride[1:])
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d = conv3x1x1(planes, planes)
        # self.conv2 = conv3x3(planes, planes)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.t_stride > 1:
            out = torch.max_pool3d(
                out, [self.t_stride, 1, 1], stride=[self.t_stride, 1, 1])
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1d(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    @property
    def out_channels(self):
        return self.conv2.out_channels


class Bottleneck3D(nn.Module):
    expansion = 2

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 op='c2d',
                 downsample=None,
                 base_width=64,
                 norm_layer=None):
        super(Bottleneck3D, self).__init__()
        self.op = op
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.))
        stride = [stride] * 3 if isinstance(stride, int) else stride
        self.conv1 = conv3x1x1(inplanes, width) if op == 'p3d' else conv1x1x1(
            inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width,
                               stride) if op == 'c3d' else conv1x3x3(
                                   width, width, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.op == 'tsm':
            out = self.tsm(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    @property
    def out_channels(self):
        return self.conv3.out_channels


class ResNet3D(Backbone):

    def __init__(self,
                 block,
                 layers,
                 ops,
                 t_stride,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 reduce_dim=0,
                 norm_layer=None):
        self.reduce_dim = reduce_dim
        self.num_classes = num_classes
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self._out_feature_strides = {'res3': 8, 'res4': 16, 'res5': 32}
        self._out_features = ['res3', 'res4', 'res5']
        self._out_feature_channels = {}
        self.outputs = {}
        self.inplanes = 64
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(
            3,
            self.inplanes, (1, 7, 7),
            stride=(t_stride[0], 2, 2),
            padding=(0, 3, 3),
            bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(t_stride[1], 2, 2),
            padding=(0, 1, 1))
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=(1, 1, 1), ops=ops[:layers[0]])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=(t_stride[2], 2, 2),
            ops=ops[layers[0]:][:layers[1]])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=(t_stride[3], 2, 2),
            ops=ops[sum(layers[:2], 0):][:layers[2]])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=(t_stride[4], 2, 2),
            ops=ops[sum(layers[:3], 0):][:layers[3]])
        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.spatial_atten = nn.Conv2d(2, 1, kernel_size=7, padding=3)
            self.drop = nn.Dropout(0.5)
            if reduce_dim > 0:
                self.rd_conv = nn.Conv2d(
                    512 * block.expansion, reduce_dim, kernel_size=1)
                self.clc = nn.Conv2d(reduce_dim, num_classes, kernel_size=1)
            else:
                self.clc = nn.Conv2d(
                    512 * block.expansion, num_classes, kernel_size=1)

        self._out_feature_channels['res3'] = self.layer2[-1].out_channels
        self._out_feature_channels['res4'] = self.layer3[-1].out_channels
        self._out_feature_channels['res5'] = self.layer4[-1].out_channels

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, ops, stride=(1, 1, 1)):
        norm_layer = self._norm_layer
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, ops[0], downsample,
                  self.base_width, norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    op=ops[i],
                    base_width=self.base_width,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.norm_x(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        self.outputs['res3'] = x.mean(dim=2)
        x = self.layer3(x)
        self.outputs['res4'] = x.mean(dim=2)
        x = self.layer4(x)  # N,C,T,H,W
        self.outputs['res5'] = x.mean(dim=2)
        if self.num_classes is not None:
            x = torch.mean(x, dim=2)  # 解决时间维度, N,C,H,W
            # spatial attention
            ftr = torch.cat(
                (x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)),
                dim=1)
            score = self.spatial_atten(ftr)  # N,1,H,W
            x = x * torch.sigmoid(score)  # N,C,H,W
            self.score = score

            x = self.avgpool(x)  # ,N,C,1,1
            if self.reduce_dim > 0:
                x = self.rd_conv(x)
                self.outputs['ftr'] = x.mean(dim=(2, 3))
        return x

    def logits(self, x):
        x = self.features(x)
        x = self.clc(x)
        return x

    def forward(self, x):
        ftr = self.features(x)
        if self.num_classes is not None:
            x = self.drop(ftr)
            x = self.clc(x)
            x = torch.mean(x, (2, 3))
            return x

        return self.outputs

    @torch.no_grad()
    def norm_x(self, x):
        m = x.new_tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1, 1])
        s = x.new_tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1, 1])
        x -= m
        x /= s
        return x


def resnet101_3d(ops, t_stride, num_class, reduce_dim=0):
    net = ResNet3D(
        Bottleneck3D, [3, 4, 23, 3],
        ops=ops,
        t_stride=t_stride,
        num_classes=num_class,
        reduce_dim=reduce_dim)
    return net


def resnet50_3d(ops, t_stride, num_class, reduce_dim=0):
    net = ResNet3D(
        Bottleneck3D, [3, 4, 6, 3],
        ops=ops,
        t_stride=t_stride,
        num_classes=num_class,
        reduce_dim=reduce_dim)
    return net


def resnet34_3d(ops, t_stride, num_class, reduce_dim=0):
    net = ResNet3D(
        BasicBlock3D, [3, 4, 6, 3],
        ops=ops,
        t_stride=t_stride,
        num_classes=num_class,
        reduce_dim=reduce_dim)
    return net


def resnet18_3d(ops, t_stride, num_class, reduce_dim=0):
    net = ResNet3D(
        BasicBlock3D, [2, 2, 2, 2],
        ops=ops,
        t_stride=t_stride,
        num_classes=num_class,
        reduce_dim=reduce_dim)
    return net
