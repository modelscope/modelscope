# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import random

import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    'Compose', 'Resize', 'Rescale', 'CenterCrop', 'CenterCropV2', 'RandomCrop',
    'RandomCropV2', 'RandomHFlip', 'GaussianBlur', 'ColorJitter', 'RandomGray',
    'ToTensor', 'Normalize', 'ResizeRandomCrop', 'ExtractResizeRandomCrop',
    'ExtractResizeAssignCrop'
]


def random_resize(img, size):
    img = [
        TF.resize(
            u,
            size,
            interpolation=random.choice([
                InterpolationMode.BILINEAR, InterpolationMode.BICUBIC,
                InterpolationMode.LANCZOS
            ])) for u in img
    ]
    return img


class CenterCropV3(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2),
                             resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)),
                         resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Compose(self.transforms[index])
        else:
            return self.transforms[index]

    def __len__(self):
        return len(self.transforms)

    def __call__(self, rgb):
        for t in self.transforms:
            rgb = t(rgb)
        return rgb


class Resize(object):

    def __init__(self, size=256):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, rgb):

        rgb = [u.resize(self.size, Image.BILINEAR) for u in rgb]
        return rgb


class Rescale(object):

    def __init__(self, size=256, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, rgb):
        w, h = rgb[0].size
        scale = self.size / min(w, h)
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        rgb = [u.resize((out_w, out_h), self.interpolation) for u in rgb]
        return rgb


class CenterCrop(object):

    def __init__(self, size=224):
        self.size = size

    def __call__(self, rgb):
        w, h = rgb[0].size
        assert min(w, h) >= self.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        rgb = [u.crop((x1, y1, x1 + self.size, y1 + self.size)) for u in rgb]
        return rgb


class ResizeRandomCrop(object):

    def __init__(self, size=256, size_short=292):
        self.size = size
        # self.min_area = min_area
        self.size_short = size_short

    def __call__(self, rgb):

        # consistent crop between rgb and m
        while min(rgb[0].size) >= 2 * self.size_short:
            rgb = [
                u.resize((u.width // 2, u.height // 2), resample=Image.BOX)
                for u in rgb
            ]
        scale = self.size_short / min(rgb[0].size)
        rgb = [
            u.resize((round(scale * u.width), round(scale * u.height)),
                     resample=Image.BICUBIC) for u in rgb
        ]
        out_w = self.size
        out_h = self.size
        w, h = rgb[0].size  # (518, 292)
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)

        rgb = [u.crop((x1, y1, x1 + out_w, y1 + out_h)) for u in rgb]

        return rgb


class ExtractResizeRandomCrop(object):

    def __init__(self, size=256, size_short=292):
        self.size = size
        self.size_short = size_short

    def __call__(self, rgb):

        # consistent crop between rgb and m
        while min(rgb[0].size) >= 2 * self.size_short:
            rgb = [
                u.resize((u.width // 2, u.height // 2), resample=Image.BOX)
                for u in rgb
            ]
        scale = self.size_short / min(rgb[0].size)
        rgb = [
            u.resize((round(scale * u.width), round(scale * u.height)),
                     resample=Image.BICUBIC) for u in rgb
        ]
        out_w = self.size
        out_h = self.size
        w, h = rgb[0].size  # (518, 292)
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)

        rgb = [u.crop((x1, y1, x1 + out_w, y1 + out_h)) for u in rgb]
        wh = [x1, y1, x1 + out_w, y1 + out_h]

        return rgb, wh


class ExtractResizeAssignCrop(object):

    def __init__(self, size=256, size_short=292):
        self.size = size
        self.size_short = size_short

    def __call__(self, rgb, wh):

        # consistent crop between rgb and m
        while min(rgb[0].size) >= 2 * self.size_short:
            rgb = [
                u.resize((u.width // 2, u.height // 2), resample=Image.BOX)
                for u in rgb
            ]
        scale = self.size_short / min(rgb[0].size)
        rgb = [
            u.resize((round(scale * u.width), round(scale * u.height)),
                     resample=Image.BICUBIC) for u in rgb
        ]

        rgb = [u.crop(wh) for u in rgb]
        rgb = [u.resize((self.size, self.size), Image.BILINEAR) for u in rgb]

        return rgb


class CenterCropV2(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # fast resize
        while min(img[0].size) >= 2 * self.size:
            img = [
                u.resize((u.width // 2, u.height // 2), resample=Image.BOX)
                for u in img
            ]
        scale = self.size / min(img[0].size)
        img = [
            u.resize((round(scale * u.width), round(scale * u.height)),
                     resample=Image.BICUBIC) for u in img
        ]

        # center crop
        x1 = (img[0].width - self.size) // 2
        y1 = (img[0].height - self.size) // 2
        img = [u.crop((x1, y1, x1 + self.size, y1 + self.size)) for u in img]
        return img


class RandomCrop(object):

    def __init__(self, size=224, min_area=0.4):
        self.size = size
        self.min_area = min_area

    def __call__(self, rgb):

        # consistent crop between rgb and m
        w, h = rgb[0].size
        area = w * h
        out_w, out_h = float('inf'), float('inf')
        while out_w > w or out_h > h:
            target_area = random.uniform(self.min_area, 1.0) * area
            aspect_ratio = random.uniform(3. / 4., 4. / 3.)
            out_w = int(round(math.sqrt(target_area * aspect_ratio)))
            out_h = int(round(math.sqrt(target_area / aspect_ratio)))
        x1 = random.randint(0, w - out_w)
        y1 = random.randint(0, h - out_h)

        rgb = [u.crop((x1, y1, x1 + out_w, y1 + out_h)) for u in rgb]
        rgb = [u.resize((self.size, self.size), Image.BILINEAR) for u in rgb]

        return rgb


class RandomCropV2(object):

    def __init__(self, size=224, min_area=0.4, ratio=(3. / 4., 4. / 3.)):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        self.min_area = min_area
        self.ratio = ratio

    def _get_params(self, img):
        width, height = img.size
        area = height * width

        for _ in range(10):
            target_area = random.uniform(self.min_area, 1.0) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(self.ratio)):
            w = width
            h = int(round(w / min(self.ratio)))
        elif (in_ratio > max(self.ratio)):
            h = height
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, rgb):
        i, j, h, w = self._get_params(rgb[0])
        rgb = [F.resized_crop(u, i, j, h, w, self.size) for u in rgb]
        return rgb


class RandomHFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb):
        if random.random() < self.p:
            rgb = [u.transpose(Image.FLIP_LEFT_RIGHT) for u in rgb]
        return rgb


class GaussianBlur(object):

    def __init__(self, sigmas=[0.1, 2.0], p=0.5):
        self.sigmas = sigmas
        self.p = p

    def __call__(self, rgb):
        if random.random() < self.p:
            sigma = random.uniform(*self.sigmas)
            rgb = [
                u.filter(ImageFilter.GaussianBlur(radius=sigma)) for u in rgb
            ]
        return rgb


class ColorJitter(object):

    def __init__(self,
                 brightness=0.4,
                 contrast=0.4,
                 saturation=0.4,
                 hue=0.1,
                 p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, rgb):
        if random.random() < self.p:
            brightness, contrast, saturation, hue = self._random_params()
            transforms = [
                lambda f: F.adjust_brightness(f, brightness),
                lambda f: F.adjust_contrast(f, contrast),
                lambda f: F.adjust_saturation(f, saturation),
                lambda f: F.adjust_hue(f, hue)
            ]
            random.shuffle(transforms)
            for t in transforms:
                rgb = [t(u) for u in rgb]

        return rgb

    def _random_params(self):
        brightness = random.uniform(
            max(0, 1 - self.brightness), 1 + self.brightness)
        contrast = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation = random.uniform(
            max(0, 1 - self.saturation), 1 + self.saturation)
        hue = random.uniform(-self.hue, self.hue)
        return brightness, contrast, saturation, hue


class RandomGray(object):

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, rgb):
        if random.random() < self.p:
            rgb = [u.convert('L').convert('RGB') for u in rgb]
        return rgb


class ToTensor(object):

    def __call__(self, rgb):
        rgb = torch.stack([F.to_tensor(u) for u in rgb], dim=0)
        return rgb


class Normalize(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, rgb):
        rgb = rgb.clone()
        rgb.clamp_(0, 1)
        if not isinstance(self.mean, torch.Tensor):
            self.mean = rgb.new_tensor(self.mean).view(-1)
        if not isinstance(self.std, torch.Tensor):
            self.std = rgb.new_tensor(self.std).view(-1)
        rgb.sub_(self.mean.view(1, -1, 1, 1)).div_(self.std.view(1, -1, 1, 1))
        return rgb
