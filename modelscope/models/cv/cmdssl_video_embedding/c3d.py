# Copyright 2022 Davide Abati.
# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

# The implementation here is modified based on c3d-pytorch,
# originally MIT License, Copyright (c) 2022 Davide Abati,
# and publicly available at https://github.com/DavideA/c3d-pytorch
""" C3D Model Architecture."""

import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class C3D(nn.Module):

    def __init__(self,
                 num_classes=1000,
                 dropout=0.5,
                 inplanes=3,
                 norm_layer=None,
                 last_pool=True):
        super(C3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not last_pool and num_classes is not None:
            raise ValueError('num_classes should be None when last_pool=False')

        self.conv1 = conv3x3x3(inplanes, 64)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = conv3x3x3(64, 128)
        self.bn2 = norm_layer(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = conv3x3x3(128, 256)
        self.bn3a = norm_layer(256)
        self.relu3a = nn.ReLU(inplace=True)

        self.conv3b = conv3x3x3(256, 256)
        self.bn3b = norm_layer(256)
        self.relu3b = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = conv3x3x3(256, 512)
        self.bn4a = norm_layer(512)
        self.relu4a = nn.ReLU(inplace=True)

        self.conv4b = conv3x3x3(512, 512)
        self.bn4b = norm_layer(512)
        self.relu4b = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = conv3x3x3(512, 512)
        self.bn5a = norm_layer(512)
        self.relu5a = nn.ReLU(inplace=True)

        self.conv5b = conv3x3x3(512, 512)
        self.bn5b = norm_layer(512)
        self.relu5b = nn.ReLU(inplace=True)
        self.pool5 = nn.AdaptiveAvgPool3d((1, 1, 1)) if last_pool else None

        if num_classes is None:
            self.dropout = None
            self.fc = None
        else:
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(512, num_classes)
        self.out_planes = 512

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.bn3a(x)
        x = self.relu3a(x)

        x = self.conv3b(x)
        x = self.bn3b(x)
        x = self.relu3b(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.bn4a(x)
        x = self.relu4a(x)

        x = self.conv4b(x)
        x = self.bn4b(x)
        x = self.relu4b(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.bn5a(x)
        x = self.relu5a(x)

        x = self.conv5b(x)
        x = self.bn5b(x)
        x = self.relu5b(x)

        if self.pool5:
            x = self.pool5(x)
            x = torch.flatten(x, 1)
            if self.dropout and self.fc:
                x = self.dropout(x)
                x = self.fc(x)

        return x
