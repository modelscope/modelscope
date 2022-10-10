# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np
import torch

from .model_resnet import FaceQuality, ResNet


class FQA(object):

    def __init__(self, backbone_path, quality_path, device='cuda', size=112):
        self.BACKBONE = ResNet(num_layers=100, feature_dim=512)
        self.QUALITY = FaceQuality(512 * 7 * 7)
        self.size = size
        self.device = device

        self.load_model(backbone_path, quality_path)

    def load_model(self, backbone_path, quality_path):
        checkpoint = torch.load(backbone_path, map_location='cpu')
        self.load_state_dict(self.BACKBONE, checkpoint)

        checkpoint = torch.load(quality_path, map_location='cpu')
        self.load_state_dict(self.QUALITY, checkpoint)

        self.BACKBONE.to(self.device)
        self.QUALITY.to(self.device)
        self.BACKBONE.eval()
        self.QUALITY.eval()

    def load_state_dict(self, model, state_dict):
        all_keys = {k for k in state_dict.keys()}
        for k in all_keys:
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict.pop(k)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def get_face_quality(self, img):
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).flip(1).to(
            self.device)
        img = (img - 127.5) / 128.0

        # extract features & predict quality
        with torch.no_grad():
            feature, fc = self.BACKBONE(img.to(self.device), True)
            s = self.QUALITY(fc)[0]

        return s.cpu().numpy()[0], feature.cpu().numpy()[0]
