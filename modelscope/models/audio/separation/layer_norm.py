# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)
# made publicly available under the MIT License
# at https://github.com/wangkenpu/Conv-TasNet-PyTorch/blob/64188ffa48971218fdd68b66906970f215d7eca2/model/layer_norm.py

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization."""

    def __init__(self, *args, **kwargs):
        super(CLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # LayerNorm
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample


class GLayerNorm(nn.Module):
    """Global Layer Normalization for TasNet."""

    def __init__(self, channels, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.norm_dim = channels
        self.gamma = nn.Parameter(torch.Tensor(channels))
        self.beta = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # Mean and variance [N, 1, 1]
        mean = torch.mean(sample, (1, 2), keepdim=True)
        var = torch.mean((sample - mean)**2, (1, 2), keepdim=True)
        sample = (sample
                  - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample
