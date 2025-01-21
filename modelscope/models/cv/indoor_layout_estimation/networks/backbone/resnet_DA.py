# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torchvision.models as models

from ..utils import StripPooling


class SPHead(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
            norm_layer(inter_channels), nn.ReLU(True))
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer,
                                        up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer,
                                        up_kwargs)
        self.score_layer = nn.Sequential(
            nn.Conv2d(
                inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
            norm_layer(inter_channels // 2), nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        return x


class ResnetDA(nn.Module):

    def __init__(self,
                 backbone='resnet50',
                 coco='',
                 input_extra=0,
                 input_height=512):
        super(ResnetDA, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=True)
        del self.encoder.fc, self.encoder.avgpool
        if coco:
            coco_pretrain = getattr(models.segmentation,
                                    coco)(pretrained=True).backbone
            self.encoder.load_state_dict(coco_pretrain.state_dict())
        self.out_channels = [256, 512, 1024, 2048]
        self.feat_heights = [input_height // 4 // (2**i) for i in range(4)]
        if int(backbone[6:]) < 50:
            self.out_channels = [_ // 4 for _ in self.out_channels]

        # Patch for extra input channel
        if input_extra > 0:
            ori_conv1 = self.encoder.conv1
            new_conv1 = nn.Conv2d(
                3 + input_extra,
                ori_conv1.out_channels,
                kernel_size=ori_conv1.kernel_size,
                stride=ori_conv1.stride,
                padding=ori_conv1.padding,
                bias=ori_conv1.bias)
            with torch.no_grad():
                for i in range(0, 3 + input_extra, 3):
                    n = new_conv1.weight[:, i:i + 3].shape[1]
                    new_conv1.weight[:, i:i + n] = ori_conv1.weight[:, :n]
            self.encoder.conv1 = new_conv1

        # Prepare for pre/pose down height filtering
        self.pre_down = None
        self.post_down = None
        # SPhead
        self.strip_pool1 = StripPooling(
            self.out_channels[0], [128, 256], (20, 12),
            norm_layer=nn.BatchNorm2d)
        self.strip_pool2 = StripPooling(
            self.out_channels[1], [64, 128], (20, 12),
            norm_layer=nn.BatchNorm2d)
        self.strip_pool3 = StripPooling(
            self.out_channels[2], [32, 64], (20, 12),
            norm_layer=nn.BatchNorm2d)
        self.strip_pool4 = StripPooling(
            self.out_channels[3], [16, 32], (10, 12),
            norm_layer=nn.BatchNorm2d)

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        if self.pre_down is not None:
            x = self.pre_down(x)
        x = self.encoder.layer1(x)
        x = self.strip_pool1(x)
        if self.post_down is not None:
            x = self.post_down(x)
        features.append(x)  # 1/4
        x = self.encoder.layer2(x)
        x = self.strip_pool2(x)
        features.append(x)  # 1/8
        x = self.encoder.layer3(x)
        x = self.strip_pool3(x)
        features.append(x)  # 1/16
        x = self.encoder.layer4(x)
        x = self.strip_pool4(x)
        features.append(x)  # 1/32
        return features
