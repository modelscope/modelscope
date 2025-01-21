# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/gaussian_maps.py
# Copyright © Alibaba, Inc. and its affiliates.

import math

import torch


def _gaussian_map(img, boxes, scale_splits=None, scale_ratios=None):
    g_maps = torch.zeros(*img.shape[1:]).to(img.device)
    height, width = img.shape[1], img.shape[2]

    x_range = torch.arange(0, height, 1).to(img.device)
    y_range = torch.arange(0, width, 1).to(img.device)
    xx, yy = torch.meshgrid(x_range, y_range)
    pos = torch.empty(xx.shape + (2, )).to(img.device)
    pos[:, :, 0] = xx
    pos[:, :, 1] = yy

    for j, box in enumerate(boxes):
        y1, x1, y2, x2 = box
        x, y, h, w = x1, y1, x2 - x1, y2 - y1
        mean_torch = torch.tensor([x + h // 2, y + w // 2]).to(img.device)
        if scale_ratios is None:
            scale_ratio = 1.0
        else:
            ratio_list = [0.2, 0.4, 0.6, 0.8, 1.0, 2, 4, 6, 8, 10]
            if h * w < scale_splits[0]:
                scale_ratio = ratio_list[scale_ratios[0]] * scale_splits[0] / (
                    h * w)
            elif h * w < scale_splits[1]:
                scale_ratio = ratio_list[scale_ratios[1]] * (
                    scale_splits[0] + scale_splits[1]) / 2.0 / (
                        h * w)
            elif h * w < scale_splits[2]:
                scale_ratio = ratio_list[scale_ratios[2]] * scale_splits[2] / (
                    h * w)
            else:
                scale_ratio = ratio_list[scale_ratios[2]]

        r_var = (scale_ratio * height * width / (2 * math.pi))**0.5
        var_x = torch.tensor([(h / height) * r_var],
                             dtype=torch.float32).to(img.device)
        var_y = torch.tensor([(w / width) * r_var],
                             dtype=torch.float32).to(img.device)
        g_map = torch.exp(-(
            ((xx.float() - mean_torch[0])**2 / (2.0 * var_x**2)
             + (yy.float() - mean_torch[1])**2 / (2.0 * var_y**2)))).to(
                 img.device)
        g_maps += g_map
    return g_maps


def _merge_gaussian(img, img_aug, boxes, scale_ratios, scale_splits):
    g_maps = _gaussian_map(img, boxes, scale_splits, scale_ratios)
    g_maps = g_maps.clamp(min=0, max=1.0)
    out = img * (1 - g_maps) + img_aug * g_maps
    return out
