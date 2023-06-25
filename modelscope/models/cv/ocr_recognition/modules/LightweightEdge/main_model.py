# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict

import torch.nn as nn

from .nas_block import plnas_linear_mix_se


class LightweightEdge(nn.Module):
    """
        基于混合rep block的nas模型
        Args:
            input (tensor): batch of input images
    """

    def __init__(self):
        super(LightweightEdge, self).__init__()
        self.FeatureExtraction = plnas_linear_mix_se(3, 123)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1))  # Transform final (imgH/16-1) -> 1
        self.dropout = nn.Dropout(0.3)
        self.Prediction = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(123, 120)),
                ('bn', nn.BatchNorm1d(120)),
                ('fc2', nn.Linear(120, 7642)),
            ]))

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(
            visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        visual_feature = self.dropout(visual_feature)
        prediction = self.Prediction.fc1(visual_feature.contiguous())
        b, t, c = prediction.shape
        prediction = self.Prediction.bn(prediction.view(b * t,
                                                        c)).view(b, t, c)
        prediction = self.Prediction.fc2(prediction)

        return prediction
