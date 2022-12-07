# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import torch
import torch.nn.functional as F
from torch.nn import (AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear,
                      MaxPool2d, Module, Parameter, ReLU, Sequential)


class LandmarkConfidence(Module):

    def __init__(self, landmark_count=5):
        super(LandmarkConfidence, self).__init__()
        self.landmark_net = LandmarkNetD(landmark_count)
        self.landmark_net.eval()
        self.cls_net = ClassNet()
        self.cls_net.eval()
        self.rp_net = RespiratorNet()

    def forward(self, x):
        feat, nose_feat, lms = self.landmark_net(x)
        cls_respirator, nose = self.rp_net(feat, nose_feat)
        confidence = self.cls_net(feat)
        return confidence, lms, cls_respirator, nose


class FC(Module):

    def __init__(self, feat_dim=256, num_class=2):
        super(FC, self).__init__()
        self.weight = Parameter(
            torch.zeros(num_class, feat_dim, dtype=torch.float32))

    def forward(self, x):
        cos_theta = F.linear(x, self.weight)
        return F.softmax(cos_theta, dim=1)


class Flatten(Module):

    def forward(self, x):
        return torch.flatten(x, 1)


class RespiratorNet(Module):

    def __init__(self):
        super(RespiratorNet, self).__init__()
        self.conv1 = Sequential(
            Conv2d(48, 48, 3, 2, 1), BatchNorm2d(48), ReLU(True))
        self.conv2 = AdaptiveAvgPool2d(
            (1, 1)
        )  # Sequential(Conv2d(48, 48, 5, 1, 0), BatchNorm2d(48), ReLU(True))
        self.binary_cls = FC(feat_dim=48, num_class=2)
        self.nose_layer = Sequential(
            Conv2d(48, 64, 3, 1, 0), BatchNorm2d(64), ReLU(True),
            Conv2d(64, 64, 3, 1, 0), BatchNorm2d(64), ReLU(True), Flatten(),
            Linear(64, 96), ReLU(True), Linear(96, 6))

    def train(self, mode=True):
        self.conv1.train(mode)
        self.conv2.train(mode)
        # self.nose_feat.train(mode)
        self.nose_layer.train(mode)
        self.binary_cls.train(mode)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.conv2(x)
        cls = self.binary_cls(torch.flatten(x, 1))
        # loc = self.nose_feat(y)
        loc = self.nose_layer(y)
        return cls, loc


class ClassNet(Module):

    def __init__(self):
        super(ClassNet, self).__init__()
        self.conv1 = Sequential(
            Conv2d(48, 48, 3, 1, 1), BatchNorm2d(48), ReLU(True))
        self.conv2 = Sequential(
            Conv2d(48, 54, 3, 2, 1), BatchNorm2d(54), ReLU(True))
        self.conv3 = Sequential(
            Conv2d(54, 54, 5, 1, 0), BatchNorm2d(54), ReLU(True))
        self.fc1 = Sequential(Flatten(), Linear(54, 54), ReLU(True))
        self.fc2 = Linear(54, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.fc1(y)
        y = self.fc2(y)
        return y


class LandmarkNetD(Module):

    def __init__(self, landmark_count=5):
        super(LandmarkNetD, self).__init__()
        self.conv_pre = Sequential(
            Conv2d(3, 16, 5, 2, 0), BatchNorm2d(16), ReLU(True))
        self.pool_pre = MaxPool2d(2, 2)  # output is 29

        self.conv1 = Sequential(
            Conv2d(16, 32, 3, 1, 1), BatchNorm2d(32), ReLU(True),
            Conv2d(32, 32, 3, 1, 1), BatchNorm2d(32), ReLU(True))
        self.pool1 = MaxPool2d(2, 2)  # 14

        self.conv2 = Sequential(
            Conv2d(32, 48, 3, 1, 0), BatchNorm2d(48), ReLU(True),
            Conv2d(48, 48, 3, 1, 0), BatchNorm2d(48), ReLU(True))
        self.pool2 = MaxPool2d(2, 2)  # 5

        self.conv3 = Sequential(
            Conv2d(48, 80, 3, 1, 0), BatchNorm2d(80), ReLU(True),
            Conv2d(80, 80, 3, 1, 0), BatchNorm2d(80), ReLU(True))

        self.fc1 = Sequential(Linear(80, 128), ReLU(True))
        self.fc2 = Sequential(Linear(128, 128), ReLU(True))

        self.output = Linear(128, landmark_count * 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.conv_pre(x)
        y = self.pool_pre(y)
        y = self.conv1(y)
        y = self.pool1(y[:, :, :28, :28])
        feat = self.conv2(y)
        y2 = self.pool2(feat)
        y = self.conv3(y2)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.output(y)
        return feat, y2, y
