# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


def crop(image, target, region, delete=True):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target['size'] = torch.tensor([h, w])

    fields = ['labels', 'area']

    if 'boxes' in target:
        boxes = target['boxes']
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target['boxes'] = cropped_boxes.reshape(-1, 4)
        target['area'] = area
        fields.append('boxes')

    if 'polygons' in target:
        polygons = target['polygons']
        num_polygons = polygons.shape[0]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        start_coord = torch.cat([
            torch.tensor([j, i], dtype=torch.float32)
            for _ in range(polygons.shape[1] // 2)], dim=0)  # yapf: disable#
        cropped_boxes = polygons - start_coord
        cropped_boxes = torch.min(
            cropped_boxes.reshape(num_polygons, -1, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target['polygons'] = cropped_boxes.reshape(num_polygons, -1)
        fields.append('polygons')

    if 'masks' in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append('masks')

    # remove elements for which the boxes or masks that have zero area
    if delete and ('boxes' in target or 'masks' in target):
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if 'boxes' in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(
                cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep.tolist()]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor(
            [-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target['boxes'] = boxes

    if 'polygons' in target:
        polygons = target['polygons']
        num_polygons = polygons.shape[0]
        polygons = polygons.reshape(num_polygons, -1, 2) * torch.as_tensor(
            [-1, 1]) + torch.as_tensor([w, 0])
        target['polygons'] = polygons

    if 'masks' in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size

        if (w <= h and w == size) or (h <= w and h == size):
            if max_size is not None:
                max_size = int(max_size)
                h = min(h, max_size)
                w = min(w, max_size)
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        if max_size is not None:
            max_size = int(max_size)
            oh = min(oh, max_size)
            ow = min(ow, max_size)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size, interpolation=Image.BICUBIC)

    if target is None:
        return rescaled_image

    ratios = tuple(
        float(s) / float(s_orig)
        for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if 'boxes' in target:
        boxes = target['boxes']
        scaled_boxes = boxes * torch.as_tensor(
            [ratio_width, ratio_height, ratio_width, ratio_height])
        target['boxes'] = scaled_boxes

    if 'polygons' in target:
        polygons = target['polygons']
        scaled_ratio = torch.cat([
            torch.tensor([ratio_width, ratio_height])
            for _ in range(polygons.shape[1] // 2)], dim=0)  # yapf: disable
        scaled_polygons = polygons * scaled_ratio
        target['polygons'] = scaled_polygons

    if 'area' in target:
        area = target['area']
        scaled_area = area * (ratio_width * ratio_height)
        target['area'] = scaled_area

    h, w = size
    target['size'] = torch.tensor([h, w])

    if 'masks' in target:
        assert False

    return rescaled_image, target


class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target,
                    (crop_top, crop_left, crop_height, crop_width))


class ObjectCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size

        x0 = float(target['boxes'][0][0])
        y0 = float(target['boxes'][0][1])
        x1 = float(target['boxes'][0][2])
        y1 = float(target['boxes'][0][3])

        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        crop_left = max(
            center_x - crop_width / 2
            + min(image_width - center_x - crop_width / 2, 0), 0)
        crop_top = max(
            center_y - crop_height / 2
            + min(image_height - center_y - crop_height / 2, 0), 0)

        return crop(
            img,
            target, (crop_top, crop_left, crop_height, crop_width),
            delete=False)


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):

    def __init__(self, sizes, max_size=None, equal=False):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
        self.equal = equal

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        if self.equal:
            return resize(img, target, size, size)
        else:
            return resize(img, target, size, self.max_size)


class ToTensor(object):

    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize(object):

    def __init__(self, mean, std, max_image_size=512):
        self.mean = mean
        self.std = std
        self.max_image_size = max_image_size

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        # h, w = image.shape[-2:]
        h, w = target['size'][0], target['size'][1]
        if 'boxes' in target:
            boxes = target['boxes']
            boxes = boxes / self.max_image_size
            target['boxes'] = boxes
        if 'polygons' in target:
            polygons = target['polygons']
            scale = torch.cat([
                torch.tensor([w, h], dtype=torch.float32)
                for _ in range(polygons.shape[1] // 2)], dim=0)  # yapf: disable
            polygons = polygons / scale
            target['polygons'] = polygons
        return image, target


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=512, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor([output_size])
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target['size'] = scaled_size

        if 'boxes' in target:
            boxes = target['boxes']
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height])
            target['boxes'] = scaled_boxes

        if 'area' in target:
            area = target['area']
            scaled_area = area * (ratio_width * ratio_height)
            target['area'] = scaled_area

        if 'masks' in target:
            assert False
            masks = target['masks']
            # masks = interpolate(
            #     masks[:, None].float(), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ['labels', 'area']

        target = target.copy()
        target['size'] = torch.tensor([h, w])

        if 'boxes' in target:
            boxes = target['boxes']
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(
                cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :]
                    - cropped_boxes[:, 0, :]).prod(dim=1)
            target['boxes'] = cropped_boxes.reshape(-1, 4)
            target['area'] = area
            fields.append('boxes')

        if 'masks' in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append('masks')

        # remove elements for which the boxes or masks that have zero area
        if 'boxes' in target or 'masks' in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if 'boxes' in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(
                    cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep.tolist()]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if 'masks' in target:
            target['masks'] = torch.nn.functional.pad(
                target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])

        random_scale = torch.rand(1) * (
            self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.maximum(scaled_size / image_size[0],
                              scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(
            image, scaled_size.tolist(), interpolation=Image.BICUBIC)

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale >= 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - self.desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      self.desired_size[0].item(), self.desired_size[0].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            assert False
            padding = self.desired_size - scaled_size
            output_image = F.pad(scaled_image,
                                 [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class OriginLargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=512, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target['size'] = scaled_size

        if 'boxes' in target:
            boxes = target['boxes']
            scaled_boxes = boxes * torch.as_tensor(
                [ratio_width, ratio_height, ratio_width, ratio_height])
            target['boxes'] = scaled_boxes

        if 'area' in target:
            area = target['area']
            scaled_area = area * (ratio_width * ratio_height)
            target['area'] = scaled_area

        if 'masks' in target:
            assert False
            masks = target['masks']
            # masks = interpolate(
            #     masks[:, None].float(), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ['labels', 'area']

        target = target.copy()
        target['size'] = torch.tensor([h, w])

        if 'boxes' in target:
            boxes = target['boxes']
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(
                cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :]
                    - cropped_boxes[:, 0, :]).prod(dim=1)
            target['boxes'] = cropped_boxes.reshape(-1, 4)
            target['area'] = area
            fields.append('boxes')

        if 'masks' in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append('masks')

        # remove elements for which the boxes or masks that have zero area
        if 'boxes' in target or 'masks' in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if 'boxes' in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(
                    cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep.tolist()]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if 'masks' in target:
            target['masks'] = torch.nn.functional.pad(
                target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])

        out_desired_size = (self.desired_size * image_size
                            / max(image_size)).round().int()

        random_scale = torch.rand(1) * (
            self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0],
                              scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image,
                                 [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target
