# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

__all__ = [
    'ResNetFeatures', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]


class ResNetFeatures(ResNet):
    """
    Modified from torchvision.models.resnet, the only one difference is outputing layer4 feature in forward.
    """

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetFeatures,
              self).__init__(block, layers, num_classes, zero_init_residual,
                             groups, width_per_group,
                             replace_stride_with_dilation, norm_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature = self.layer4(x)

        x = self.avgpool(feature)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, feature


def resnet18(**kwargs):
    return ResNetFeatures(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNetFeatures(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNetFeatures(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNetFeatures(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNetFeatures(Bottleneck, [3, 8, 36, 3], **kwargs)
