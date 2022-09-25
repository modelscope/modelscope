# The implementation is adopted from TFace,made pubicly available under the Apache-2.0 license at
# https://github.com/Tencent/TFace/blob/master/recognition/torchkit/backbone/common.py
import torch
import torch.nn as nn
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module, ReLU,
                      Sigmoid)


def initialize_weights(modules):
    """ Weight initilize, conv2d and linear is initialized with kaiming_normal
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()


class Flatten(Module):
    """ Flat tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class SEModule(Module):
    """ SE block
    """

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels,
            channels // reduction,
            kernel_size=1,
            padding=0,
            bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction,
            channels,
            kernel_size=1,
            padding=0,
            bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x
