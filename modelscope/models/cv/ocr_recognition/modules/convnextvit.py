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
        """ Transformation stage """
        features = self.cnn_model(input)
        output = self.vitstr(features)
        return output
