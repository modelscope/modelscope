# The implementation here is modified based on BaSSL,
# originally Apache 2.0 License and publicly available at https://github.com/kakaobrain/bassl
import numbers
import os.path as osp
import random
from typing import List

import numpy as np
import torch
import torchvision.transforms as TF
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter


def get_transform(lst):
    assert len(lst) > 0
    transform_lst = []
    for item in lst:
        transform_lst.append(build_transform(item))
    transform = TF.Compose(transform_lst)
    return transform


def build_transform(cfg):
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')

    if type == 'VideoResizedCenterCrop':
        return VideoResizedCenterCrop(**cfg)
    elif type == 'VideoToTensor':
        return VideoToTensor(**cfg)
    elif type == 'VideoRandomResizedCrop':
        return VideoRandomResizedCrop(**cfg)
    elif type == 'VideoRandomHFlip':
        return VideoRandomHFlip()
    elif type == 'VideoRandomColorJitter':
        return VideoRandomColorJitter(**cfg)
    elif type == 'VideoRandomGaussianBlur':
        return VideoRandomGaussianBlur(**cfg)
    else:
        raise NotImplementedError


class VideoResizedCenterCrop(torch.nn.Module):

    def __init__(self, image_size, crop_size):
        self.tfm = TF.Compose([
            TF.Resize(size=image_size, interpolation=Image.BICUBIC),
            TF.CenterCrop(crop_size),
        ])

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        return [self.tfm(img) for img in imgmap]


class VideoToTensor(torch.nn.Module):

    def __init__(self, mean=None, std=None, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

        assert self.mean is not None
        assert self.std is not None

    def __to_tensor__(self, img):
        return F.to_tensor(img)

    def __normalize__(self, img):
        return F.normalize(img, self.mean, self.std, self.inplace)

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        return [self.__normalize__(self.__to_tensor__(img)) for img in imgmap]


class VideoRandomResizedCrop(torch.nn.Module):

    def __init__(self, size, bottom_area=0.2):
        self.p = 1.0
        self.interpolation = Image.BICUBIC
        self.size = size
        self.bottom_area = bottom_area

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.p:  # do RandomResizedCrop, consistent=True
            top, left, height, width = TF.RandomResizedCrop.get_params(
                imgmap[0],
                scale=(self.bottom_area, 1.0),
                ratio=(3 / 4.0, 4 / 3.0))
            return [
                F.resized_crop(
                    img=img,
                    top=top,
                    left=left,
                    height=height,
                    width=width,
                    size=(self.size, self.size),
                ) for img in imgmap
            ]
        else:
            return [
                F.resize(img=img, size=[self.size, self.size])
                for img in imgmap
            ]


class VideoRandomHFlip(torch.nn.Module):

    def __init__(self, consistent=True, command=None, seq_len=0):
        self.consistent = consistent
        if seq_len != 0:
            self.consistent = False
        if command == 'left':
            self.threshold = 0
        elif command == 'right':
            self.threshold = 1
        else:
            self.threshold = 0.5
        self.seq_len = seq_len

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if self.consistent:
            if random.random() < self.threshold:
                return [i.transpose(Image.FLIP_LEFT_RIGHT) for i in imgmap]
            else:
                return imgmap
        else:
            result = []
            for idx, i in enumerate(imgmap):
                if idx % self.seq_len == 0:
                    th = random.random()
                if th < self.threshold:
                    result.append(i.transpose(Image.FLIP_LEFT_RIGHT))
                else:
                    result.append(i)
            assert len(result) == len(imgmap)
            return result


class VideoRandomColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(
        self,
        brightness=0,
        contrast=0,
        saturation=0,
        hue=0,
        consistent=True,
        p=1.0,
        seq_len=0,
    ):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(
            hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)
        self.consistent = consistent
        self.threshold = p
        self.seq_len = seq_len

    def _check_input(self,
                     value,
                     name,
                     center=1,
                     bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    'If {} is a single number, it must be non negative.'.
                    format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError('{} values should be between {}'.format(
                    name, bound))
        else:
            raise TypeError(
                '{} should be a single number or a list/tuple with lenght 2.'.
                format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(
                TF.Lambda(
                    lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(
                TF.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(
                TF.Lambda(
                    lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(
                TF.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = TF.Compose(transforms)

        return transform

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.threshold:  # do ColorJitter
            if self.consistent:
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue)

                return [transform(i) for i in imgmap]
            else:
                if self.seq_len == 0:
                    return [
                        self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)(img)
                        for img in imgmap
                    ]
                else:
                    result = []
                    for idx, img in enumerate(imgmap):
                        if idx % self.seq_len == 0:
                            transform = self.get_params(
                                self.brightness,
                                self.contrast,
                                self.saturation,
                                self.hue,
                            )
                        result.append(transform(img))
                    return result

        else:
            return imgmap

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class VideoRandomGaussianBlur(torch.nn.Module):

    def __init__(self, radius_min=0.1, radius_max=2.0, p=0.5):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, imgmap):
        assert isinstance(imgmap, list)
        if random.random() < self.p:
            result = []
            for _, img in enumerate(imgmap):
                _radius = random.uniform(self.radius_min, self.radius_max)
                result.append(
                    img.filter(ImageFilter.GaussianBlur(radius=_radius)))
            return result
        else:
            return imgmap


def apply_transform(images, trans):
    return torch.stack(trans(images), dim=0)
