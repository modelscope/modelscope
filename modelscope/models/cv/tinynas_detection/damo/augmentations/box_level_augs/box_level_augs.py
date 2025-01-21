# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/box_level_augs.py
# Copyright © Alibaba, Inc. and its affiliates.

import random

import numpy as np

from .color_augs import color_aug_func
from .geometric_augs import geometric_aug_func


def _box_sample_prob(bbox, scale_ratios_splits, box_prob=0.3):
    scale_ratios, scale_splits = scale_ratios_splits

    ratios = np.array(scale_ratios)
    ratios = ratios / ratios.sum()
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area == 0:
        return 0
    if area < scale_splits[0]:
        scale_ratio = ratios[0]
    elif area < scale_splits[1]:
        scale_ratio = ratios[1]
    else:
        scale_ratio = ratios[2]
    return box_prob * scale_ratio


def _box_aug_per_img(img,
                     target,
                     aug_type=None,
                     scale_ratios=None,
                     scale_splits=None,
                     img_prob=0.1,
                     box_prob=0.3,
                     level=1):
    if random.random() > img_prob:
        return img, target
    img /= 255.0

    tag = 'prob' if aug_type in geometric_aug_func else 'area'
    scale_ratios_splits = [scale_ratios[tag], scale_splits]
    if scale_ratios is None:
        box_sample_prob = [box_prob] * len(target.bbox)
    else:
        box_sample_prob = [
            _box_sample_prob(bbox, scale_ratios_splits, box_prob=box_prob)
            for bbox in target.bbox
        ]

    if aug_type in color_aug_func:
        img_aug = color_aug_func[aug_type](
            img, level, target, [scale_ratios['area'], scale_splits],
            box_sample_prob)
    elif aug_type in geometric_aug_func:
        img_aug, target = geometric_aug_func[aug_type](img, level, target,
                                                       box_sample_prob)
    else:
        raise ValueError('Unknown box-level augmentation function %s.' %
                         (aug_type))
    out = img_aug * 255.0

    return out, target


class Box_augs(object):

    def __init__(self, box_augs_dict, max_iters, scale_splits, box_prob=0.3):
        self.max_iters = max_iters
        self.box_prob = box_prob
        self.scale_splits = scale_splits
        self.policies = box_augs_dict['policies']
        self.scale_ratios = box_augs_dict['scale_ratios']

    def __call__(self, tensor, target, iteration):
        iter_ratio = float(iteration) / self.max_iters
        sub_policy = random.choice(self.policies)

        h, w = tensor.shape[-2:]
        ratio = min(h, w) / 800

        scale_splits = [area * ratio for area in self.scale_splits]
        if iter_ratio <= 1:
            tensor, _ = _box_aug_per_img(
                tensor,
                target,
                aug_type=sub_policy[0][0],
                scale_ratios=self.scale_ratios,
                scale_splits=scale_splits,
                img_prob=sub_policy[0][1] * iter_ratio,
                box_prob=self.box_prob,
                level=sub_policy[0][2])
            tensor, target = _box_aug_per_img(
                tensor,
                target,
                aug_type=sub_policy[1][0],
                scale_ratios=self.scale_ratios,
                scale_splits=scale_splits,
                img_prob=sub_policy[1][1] * iter_ratio,
                box_prob=self.box_prob,
                level=sub_policy[1][2])

        return tensor, target
