# Part of the implementation is borrowed and modified from pytorch-caffe-models,
# publicly available at https://github.com/crowsonkb/pytorch-caffe-models

import cv2
import numpy as np
import torch
import torch.nn as nn


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5,
                 pool_proj):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, ch1x1, 1)
        self.relu_1x1 = nn.ReLU(inplace=True)
        self.conv_3x3_reduce = nn.Conv2d(in_channels, ch3x3red, 1)
        self.relu_3x3_reduce = nn.ReLU(inplace=True)
        self.conv_3x3 = nn.Conv2d(ch3x3red, ch3x3, 3, padding=1)
        self.relu_3x3 = nn.ReLU(inplace=True)
        self.conv_5x5_reduce = nn.Conv2d(in_channels, ch5x5red, 1)
        self.relu_5x5_reduce = nn.ReLU(inplace=True)
        self.conv_5x5 = nn.Conv2d(ch5x5red, ch5x5, 5, padding=2)
        self.relu_5x5 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, 1)
        self.relu_pool_proj = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_1 = self.relu_1x1(self.conv_1x1(x))
        branch_2 = self.relu_3x3_reduce(self.conv_3x3_reduce(x))
        branch_2 = self.relu_3x3(self.conv_3x3(branch_2))
        branch_3 = self.relu_5x5_reduce(self.conv_5x5_reduce(x))
        branch_3 = self.relu_5x5(self.conv_5x5(branch_3))
        branch_4 = self.pool(x)
        branch_4 = self.relu_pool_proj(self.pool_proj(branch_4))
        return torch.cat([branch_1, branch_2, branch_3, branch_4], dim=1)


class GoogLeNet(nn.Sequential):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.norm1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.conv2_reduce = nn.Conv2d(64, 64, kernel_size=1)
        self.relu2_reduce = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.norm2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75)
        self.pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        self.loss3_classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.relu2_reduce(self.conv2_reduce(x))
        x = self.relu2(self.conv2(x))
        x = self.norm2(x)
        x = self.pool2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.pool3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.pool4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.pool5(x).flatten(1)
        return x


class bvlc_googlenet(nn.Module):

    def __init__(self, input_size=224):
        """model for the BVLC GoogLeNet, trained on ImageNet.
        URL: https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet"""
        super(bvlc_googlenet, self).__init__()

        self.model = GoogLeNet(num_classes=1000)

        self.input_size = input_size
        self.input_mean = (104.0, 117.0, 123.0)

    def forward(self, frame):
        x = cv2.resize(frame,
                       (self.input_size, self.input_size)).astype(np.float32)
        x = (x - self.input_mean).astype(np.float32)
        x = np.transpose(x, [2, 0, 1])

        x = np.expand_dims(x, 0)
        x = torch.from_numpy(x)
        if not next(self.model.parameters()).device.type == 'cpu':
            x = x.cuda()
        with torch.no_grad():
            frame_feat = self.model(x)
            if not frame_feat.device.type == 'cpu':
                frame_feat = frame_feat.cpu()
            frame_feat = frame_feat.numpy()
            frame_feat = frame_feat / np.linalg.norm(frame_feat)
        return frame_feat.reshape(-1)
