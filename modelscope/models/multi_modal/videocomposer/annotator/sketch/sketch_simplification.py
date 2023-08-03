r"""PyTorch re-implementation adapted from the Lua code in ``https://github.com/bobbens/sketch_simplification''.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.multi_modal.videocomposer.utils.utils import \
    DOWNLOAD_TO_CACHE

__all__ = [
    'SketchSimplification', 'sketch_simplification_gan',
    'sketch_simplification_mse', 'sketch_to_pencil_v1', 'sketch_to_pencil_v2'
]


class SketchSimplification(nn.Module):
    r"""NOTE:
        1. Input image should has only one gray channel.
        2. Input image size should be divisible by 8.
        3. Sketch in the input/output image is in dark color while background in light color.
    """

    def __init__(self, mean, std):
        assert isinstance(mean, float) and isinstance(std, float)
        super(SketchSimplification, self).__init__()
        self.mean = mean
        self.std = std

        # layers
        self.layers = nn.Sequential(
            nn.Conv2d(1, 48, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(48, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 48, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        r"""x: [B, 1, H, W] within range [0, 1]. Sketch pixels in dark color.
        """
        x = (x - self.mean) / self.std
        return self.layers(x)


def sketch_simplification_gan(model_dir, pretrained=False):
    model = SketchSimplification(
        mean=0.9664114577640158, std=0.0858381272736797)
    if pretrained:
        model.load_state_dict(
            torch.load(
                os.path.join(model_dir, 'sketch_simplification_gan.pth'),
                map_location='cpu'))
    return model


def sketch_simplification_mse(pretrained=False):
    model = SketchSimplification(
        mean=0.9664423107454593, std=0.08583666033640507)
    if pretrained:
        model.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE(
                    'models/sketch_simplification/sketch_simplification_mse.pth'
                ),
                map_location='cpu'))
    return model


def sketch_to_pencil_v1(pretrained=False):
    model = SketchSimplification(
        mean=0.9817833515894078, std=0.0925009022585048)
    if pretrained:
        model.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE(
                    'models/sketch_simplification/sketch_to_pencil_v1.pth'),
                map_location='cpu'))
    return model


def sketch_to_pencil_v2(pretrained=False):
    model = SketchSimplification(
        mean=0.9851298627337799, std=0.07418377454883571)
    if pretrained:
        model.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE(
                    'models/sketch_simplification/sketch_to_pencil_v2.pth'),
                map_location='cpu'))
    return model
