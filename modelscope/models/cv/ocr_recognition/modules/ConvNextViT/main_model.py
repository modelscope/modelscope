# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from .convnext import convnext_tiny
from .vitstr import vitstr_tiny


class ConvNextViT(nn.Module):

    def __init__(self):
        super(ConvNextViT, self).__init__()
        self.cnn_model = convnext_tiny()
        self.vitstr = vitstr_tiny(num_tokens=7644)

    def forward(self, input):
        # RGB2GRAY
        input = input[:, 0:
                      1, :, :] * 0.2989 + input[:, 1:
                                                2, :, :] * 0.5870 + input[:, 2:
                                                                          3, :, :] * 0.1140
        features = self.cnn_model(input)
        output = self.vitstr(features)
        return output
