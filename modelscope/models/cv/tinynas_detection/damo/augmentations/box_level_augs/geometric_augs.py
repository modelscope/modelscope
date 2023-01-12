# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/box_level_augs/geometric_augs.py
# Copyright © Alibaba, Inc. and its affiliates.

import copy
import random

import torch
import torchvision.transforms as transforms

from .gaussian_maps import _gaussian_map

_MAX_LEVEL = 10.0
pixel_mean = [102.9801, 115.9465, 122.7717]


def scale_area(box, height, width, scale_ratio=1.0):
    y1, x1, y2, x2 = box
    h, w = x2 - x1, y2 - y1
    h_new, w_new = h * scale_ratio, w * scale_ratio
    x1, y1 = max(x1 + h / 2 - h_new / 2, 0), max(y1 + w / 2 - w_new / 2, 0)
    x2, y2 = min(x1 + h_new, height), min(y1 + w_new, width)
    box_new = torch.Tensor([y1, x1, y2, x2])
    return box_new


def _geometric_aug_func(x,
                        target,
                        angle=0,
                        translate=(0, 0),
                        scale=1,
                        shear=(0, 0),
                        hflip=False,
                        boxes_sample_prob=[],
                        scale_ratio=1.0):
    boxes_and_labels = [(target.bbox[i], target.extra_fields['labels'][i])
                        for i in range(len(target.bbox))
                        if random.random() < boxes_sample_prob[i]]
    boxes = [b_and_l[0] for b_and_l in boxes_and_labels]
    labels = [b_and_l[1] for b_and_l in boxes_and_labels]

    if random.random() < 0.5:
        angle *= -1
        translate = (-translate[0], -translate[1])
        shear = (-shear[0], -shear[1])

    height, width = x.shape[1], x.shape[2]

    x_crops = []
    boxes_crops = []
    boxes_new = []
    labels_new = []
    for i, box in enumerate(boxes):
        box_crop = scale_area(box, height, width, scale_ratio)
        y1, x1, y2, x2 = box_crop.long()

        x_crop = x[:, x1:x2, y1:y2]
        boxes_crops.append(box_crop)

        if x1 >= x2 or y1 >= y2:
            x_crops.append(x_crop)
            continue

        if hflip:
            x_crop = x_crop.flip(-1)
        elif translate[0] + translate[1] != 0:
            offset_y = (y2 + translate[0]).clamp(0, width).long().tolist() - y2
            offset_x = (x2 + translate[1]).clamp(0,
                                                 height).long().tolist() - x2
            if offset_x != 0 or offset_y != 0:
                offset = [offset_y, offset_x]
                boxes_new.append(box + torch.Tensor(offset * 2))
                labels_new.append(labels[i])
        else:
            x_crop = transforms.functional.to_pil_image(x_crop.cpu())
            x_crop = transforms.functional.affine(
                x_crop,
                angle,
                translate,
                scale,
                shear,
                resample=2,
                fillcolor=tuple([int(i) for i in pixel_mean]))
            x_crop = transforms.functional.to_tensor(x_crop).to(x.device)
        x_crops.append(x_crop)
    y = _transform(x, x_crops, boxes_crops, translate)

    if translate[0] + translate[1] != 0 and len(boxes_new) > 0:
        target.bbox = torch.cat((target.bbox, torch.stack(boxes_new)))
        target.extra_fields['labels'] = torch.cat(
            (target.extra_fields['labels'], torch.Tensor(labels_new).long()))

    return y, target


def _transform(x, x_crops, boxes_crops, translate=(0, 0)):
    y = copy.deepcopy(x)
    height, width = x.shape[1], x.shape[2]

    for i, box in enumerate(boxes_crops):
        y1_c, x1_c, y2_c, x2_c = boxes_crops[i].long()

        y1_c = (y1_c + translate[0]).clamp(0, width).long().tolist()
        x1_c = (x1_c + translate[1]).clamp(0, height).long().tolist()
        y2_c = (y2_c + translate[0]).clamp(0, width).long().tolist()
        x2_c = (x2_c + translate[1]).clamp(0, height).long().tolist()

        y_crop = copy.deepcopy(y[:, x1_c:x2_c, y1_c:y2_c])
        x_crop = x_crops[i][:, :y_crop.shape[1], :y_crop.shape[2]]

        if y_crop.shape[1] * y_crop.shape[2] == 0:
            continue

        g_maps = _gaussian_map(x_crop,
                               [[0, 0, y_crop.shape[2], y_crop.shape[1]]])
        _, _h, _w = y[:, x1_c:x2_c, y1_c:y2_c].shape
        y[:, x1_c:x1_c + x_crop.shape[1],
          y1_c:y1_c + x_crop.shape[2]] = g_maps * x_crop + (
              1 - g_maps) * y_crop[:, :x_crop.shape[1], :x_crop.shape[2]]
    return y


geometric_aug_func = {
    'hflip':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x, target, hflip=True, boxes_sample_prob=boxes_sample_probs),
    'rotate':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        level / _MAX_LEVEL * 30,
        boxes_sample_prob=boxes_sample_probs),
    'shearX':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        shear=(level / _MAX_LEVEL * 15, 0),
        boxes_sample_prob=boxes_sample_probs),
    'shearY':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        shear=(0, level / _MAX_LEVEL * 15),
        boxes_sample_prob=boxes_sample_probs),
    'translateX':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        translate=(level / _MAX_LEVEL * 120.0, 0),
        boxes_sample_prob=boxes_sample_probs),
    'translateY':
    lambda x, level, target, boxes_sample_probs: _geometric_aug_func(
        x,
        target,
        translate=(0, level / _MAX_LEVEL * 120.0),
        boxes_sample_prob=boxes_sample_probs)
}
