# The implementation is adopted from CenseoQoE, made pubicly available under the MIT License at
# https://github.com/Tencent/CenseoQoE

import torch
from torch import nn

from . import backbones, heads


class CenseoIVQAModel(nn.Module):
    """
    A strong baseline model for image quality assessment.
    Its architecture is based on the modified resnet18 and reach SOTA interms of PLCC and SRCC.
    The reference papaer is https://arxiv.org/pdf/2111.07104.pdf
    """

    def __init__(self, pretrained=True):
        super(CenseoIVQAModel, self).__init__()
        input_channels = 3
        model_name = 'resnet18'
        self.backbone = getattr(backbones, model_name)(
            input_channels=input_channels,
            pretrained=pretrained,
            out_indices=(3, ),
            strides=(2, 2, 2))
        self.head = getattr(heads, 'SimpleHead')(
            self.backbone.ouput_dims, out_num=1)

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        out = torch.sigmoid(out)
        return out
