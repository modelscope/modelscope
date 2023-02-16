# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/modeling/meta_arch/gdl.py

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientDecoupleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None


class AffineLayer(nn.Module):

    def __init__(self, num_channels, bias=False):
        super(AffineLayer, self).__init__()
        weight = torch.FloatTensor(1, num_channels, 1, 1).fill_(1)
        self.weight = nn.Parameter(weight, requires_grad=True)

        self.bias = None
        if bias:
            bias = torch.FloatTensor(1, num_channels, 1, 1).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)

    def forward(self, X):
        out = X * self.weight.expand_as(X)
        if self.bias is not None:
            out = out + self.bias.expand_as(X)
        return out


def decouple_layer(x, _lambda):
    return GradientDecoupleLayer.apply(x, _lambda)
