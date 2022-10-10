# The implementation is adopted from Pytorch_Retinaface, made pubicly available under the MIT License
# at https://github.com/biubug6/Pytorch_Retinaface/tree/master/models/retinaface.py
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.models._utils as _utils
import torchvision.models.detection.backbone_utils as backbone_utils

from .net import FPN, SSH, MobileNetV1


class ClassHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels,
            self.num_anchors * 2,
            kernel_size=(1, 1),
            stride=1,
            padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels,
            num_anchors * 4,
            kernel_size=(1, 1),
            stride=1,
            padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels,
            num_anchors * 10,
            kernel_size=(1, 1),
            stride=1,
            padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class RetinaFace(nn.Module):

    def __init__(self, cfg=None):
        """
        :param cfg:  Network related settings.
        """
        super(RetinaFace, self).__init__()
        backbone = None
        if cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=cfg['pretrain'])
        else:
            raise Exception('Invalid name')

        self.body = _utils.IntermediateLayerGetter(backbone,
                                                   cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(
            fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(
            fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(
            fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)],
            dim=1)
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)],
            dim=1)
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feat) for i, feat in enumerate(features)],
            dim=1)

        output = (bbox_regressions, F.softmax(classifications,
                                              dim=-1), ldm_regressions)
        return output
