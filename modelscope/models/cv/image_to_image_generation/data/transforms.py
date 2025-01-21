# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import math
import random

import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

__all__ = [
    'Identity', 'PadToSquare', 'RandomScale', 'RandomRotate',
    'RandomGaussianBlur', 'RandomCrop'
]


class Identity(object):

    def __call__(self, *args):
        if len(args) == 0:
            return None
        elif len(args) == 1:
            return args[0]
        else:
            return args


class PadToSquare(object):

    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w != h:
            if w > h:
                t = (w - h) // 2
                b = w - h - t
                padding = (0, t, 0, b)
            else:
                left = (h - w) // 2
                right = h - w - l
                padding = (left, 0, right, 0)
            img = TF.pad(img, padding, fill=self.fill)
        return img


class RandomScale(object):

    def __init__(self,
                 min_scale=0.5,
                 max_scale=2.0,
                 min_ratio=0.8,
                 max_ratio=1.25):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def __call__(self, img):
        w, h = img.size
        scale = 2**random.uniform(
            math.log2(self.min_scale), math.log2(self.max_scale))
        ratio = 2**random.uniform(
            math.log2(self.min_ratio), math.log2(self.max_ratio))
        ow = int(w * scale * math.sqrt(ratio))
        oh = int(h * scale / math.sqrt(ratio))
        img = img.resize((ow, oh), Image.BILINEAR)
        return img


class RandomRotate(object):

    def __init__(self,
                 min_angle=-10.0,
                 max_angle=10.0,
                 padding=(255, 255, 255),
                 p=0.5):
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.padding = padding
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            angle = random.uniform(self.min_angle, self.max_angle)
            img = img.rotate(angle, Image.BILINEAR, fillcolor=self.padding)
        return img


class RandomGaussianBlur(object):

    def __init__(self, radius=5, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        return img


class RandomCrop(object):

    def __init__(self, size, padding=(255, 255, 255)):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        # pad
        w, h = img.size
        pad_w = max(0, self.size - w)
        pad_h = max(0, self.size - h)
        if pad_w > 0 or pad_h > 0:
            half_w = pad_w // 2
            half_h = pad_h // 2
            pad = (half_w, half_h, pad_w - half_w, pad_h - half_h)
            img = TF.pad(img, pad, fill=self.padding)

        # crop
        w, h = img.size
        x1 = random.randint(0, w - self.size)
        y1 = random.randint(0, h - self.size)
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img
