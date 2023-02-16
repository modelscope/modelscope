import time
from collections import defaultdict

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models


class DegradationIQA(nn.Module):

    def __init__(self):
        super(DegradationIQA, self).__init__()
        # [64, 128, 128]
        features = list(
            models.__dict__['resnet50'](pretrained=False).children())[:-2]
        features = nn.Sequential(*features)
        # features = list(models.__dict__['vgg16'](pretrained=True).children())[:-2][0][:-1]
        self.features = features
        self.lcn_radius = 7
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self.noise_regression = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self.blur_regression = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self.compression_regression = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self.bright_regression = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self.color_regression = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            # nn.BatchNorm1d(256),
            nn.Conv2d(256, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1, dilation=1))

        self._initialize_weights()
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def _initialize_weights(self):
        initialize_layers = [
            x for j in [
                self.classifier, self.noise_regression, self.blur_regression,
                self.compression_regression, self.bright_regression,
                self.color_regression
            ] for x in j
        ]
        for m in initialize_layers:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, require_map=False):

        for model in self.features:
            x = model(x)

        fea = x
        out_map = self.classifier(fea)
        noise_map = self.noise_regression(fea)
        blur_map = self.blur_regression(fea)
        comp_map = self.compression_regression(fea)
        bright_map = self.bright_regression(fea)
        color_map = self.color_regression(fea)
        out = torch.mean(torch.mean(out_map, dim=2), dim=2)
        noise_out = torch.mean(torch.mean(noise_map, dim=2), dim=2)
        blur_out = torch.mean(torch.mean(blur_map, dim=2), dim=2)
        comp_out = torch.mean(torch.mean(comp_map, dim=2), dim=2)
        bright_out = torch.mean(torch.mean(bright_map, dim=2), dim=2)
        color_out = torch.mean(torch.mean(color_map, dim=2), dim=2)

        if not require_map:
            return out, [noise_out, blur_out, comp_out, bright_out, color_out]
        else:
            return out, [
                noise_out, blur_out, comp_out, bright_out, color_out
            ], [noise_map, blur_map, comp_map, bright_map, color_map]
