# The implementation is based on Facial-Expression-Recognition, available at
# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
import numbers
import types

import numpy as np
import torch
from PIL import Image


def to_tensor(pic):

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img.crop((j, i, j + tw, i + th))


def five_crop(img, size):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(
            size) == 2, 'Please provide only two dimensions (h, w) for size.'

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError(
            'Requested crop size {} is bigger than input size {}'.format(
                size, (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


class TenCrop(object):

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(
                size
            ) == 2, 'Please provide only two dimensions (h, w) for size.'
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        first_five = five_crop(img, self.size)

        if self.vertical_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        second_five = five_crop(img, self.size)

        return first_five + second_five


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):

    def __call__(self, pic):
        return to_tensor(pic)


class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)
