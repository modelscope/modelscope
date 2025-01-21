# The implementation is based on
# "Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization",
# ICCV 2019, Seoul, paper available at https://arxiv.org/abs/1910.04562
# Poster available at https://chufengt.github.io/publication/pedestrian-attribute/iccv_poster_id2029.pdf

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks


def gem(x, p=3, eps=1e-6):
    return F.adaptive_avg_pool2d(F.relu(x + eps).pow(p), (1, 1)).pow(1. / p)


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(
            self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


@MODELS.register_module(
    Tasks.pedestrian_attribute_recognition,
    module_name=Models.pedestrian_attribute_recognition)
class PedestrainAttribute(TorchModel):
    """Pedestrain Attribute Recognition model.
    """

    def __init__(self, num_classes=51, **kwargs):
        """initialize the pedestrain attribute recognition model.

        Args:
            num_classes (int): the number of attributes.
        """
        super(PedestrainAttribute, self).__init__(**kwargs)
        model_ft = models.resnet50(pretrained=False)
        model_ft.avgpool = GeM()
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.fc = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512))
        self.fc.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        """
        Args:
            x (`torch.Tensor`): batched image tensor,
                shape of each tensor is [N, 3, H, W].

        Return:
            `labels [N, num_classes] of the attributes`
        """

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
