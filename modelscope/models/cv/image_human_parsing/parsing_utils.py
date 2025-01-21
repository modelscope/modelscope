# Part of the implementation is borrowed and modified from M2FP, made publicly available
# under the CC BY-NC 4.0 License at https://github.com/soeaver/M2FP
# Part of the implementation is borrowed and modified from Detectron2, made publicly available
# under the Apache-2.0 License at https://github.com/facebookresearch/detectron2

import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def center_to_target_size_test(img, target_size):
    src_h, src_w = img.shape[0], img.shape[1]
    trg_h, trg_w = target_size[1], target_size[0]

    new_h, new_w = 0, 0
    tfm_list = []
    if src_h > trg_h and src_w > trg_w:
        if src_h >= src_w:
            new_h = trg_h
            new_w = int(new_h * src_w / src_h)
            if new_w > trg_w:
                new_w = trg_w
                new_h = int(new_w * src_h / src_w)
        elif src_w > src_h:
            new_w = trg_w
            new_h = int(new_w * src_h / src_w)
            if new_h > trg_h:
                new_h = trg_h
                new_w = int(new_h * src_w / src_h)
        tfm_list.append(ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    elif src_h > trg_h and src_w <= trg_w:
        new_h = trg_h
        new_w = int(new_h * src_w / src_h)
        tfm_list.append(ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    elif src_h <= trg_h and src_w > trg_w:
        new_w = trg_w
        new_h = int(new_w * src_h / src_w)
        tfm_list.append(ResizeTransform(src_h, src_w, new_h, new_w))
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    else:
        new_h, new_w = src_h, src_w
        tfm_list.append(PadTransform(new_h, new_w, trg_h, trg_w))

    box = get_box(new_h, new_w, trg_h, trg_w)

    new_img = copy.deepcopy(img)
    for tfm in tfm_list:
        new_img = tfm.apply_image(new_img)

    return new_img, box


def get_box(src_h, src_w, trg_h, trg_w):
    assert src_h <= trg_h, 'expect src_h <= trg_h'
    assert src_w <= trg_w, 'expect src_w <= trg_w'

    x0 = int((trg_w - src_w) / 2)
    x1 = src_w + x0
    y0 = int((trg_h - src_h) / 2)
    y1 = src_h + y0

    box = [x0, y0, x1, y1]
    return box


class PadTransform:

    def __init__(self, src_h, src_w, trg_h, trg_w):
        super().__init__()
        assert src_h <= trg_h, 'expect src_h <= trg_h'
        assert src_w <= trg_w, 'expect src_w <= trg_w'

        self.src_h, self.src_w = src_h, src_w
        self.trg_h, self.trg_w = trg_h, trg_w
        self.pad_left = int((trg_w - src_w) / 2)
        self.pad_right = trg_w - src_w - self.pad_left
        self.pad_top = int((trg_h - src_h) / 2)
        self.pad_bottom = trg_h - src_h - self.pad_top

    def apply_image(self, img, pad_value=128):
        if self.pad_left == 0 and self.pad_top == 0:
            return img

        if len(img.shape) == 2:
            return np.pad(
                img, ((self.pad_top, self.pad_bottom),
                      (self.pad_left, self.pad_right)),
                'constant',
                constant_values=((pad_value, pad_value), (pad_value,
                                                          pad_value)))
        elif len(img.shape) == 3:
            return np.pad(
                img, ((self.pad_top, self.pad_bottom),
                      (self.pad_left, self.pad_right), (0, 0)),
                'constant',
                constant_values=((pad_value, pad_value),
                                 (pad_value, pad_value), (pad_value,
                                                          pad_value)))


class ResizeTransform:

    def __init__(self, h, w, new_h, new_w, interp=None):
        super().__init__()
        if interp is None:
            interp = Image.BILINEAR
        self.h, self.w = h, w
        self.new_h, self.new_w = new_h, new_w
        self.interp = interp

    def apply_image(self, img, interp=None):
        assert img.shape[:2] == (self.h, self.w)
        assert len(img.shape) <= 4
        interp_method = interp if interp is not None else self.interp

        if img.dtype == np.uint8:
            if len(img.shape) > 2 and img.shape[2] == 1:
                pil_image = Image.fromarray(img[:, :, 0], mode='L')
            else:
                pil_image = Image.fromarray(img)
            pil_image = pil_image.resize((self.new_w, self.new_h),
                                         interp_method)
            ret = np.asarray(pil_image)
            if len(img.shape) > 2 and img.shape[2] == 1:
                ret = np.expand_dims(ret, -1)
        else:
            # PIL only supports uint8
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            shape = list(img.shape)
            shape_4d = shape[:2] + [1] * (4 - len(shape)) + shape[2:]
            img = img.view(shape_4d).permute(2, 3, 0, 1)  # hw(c) -> nchw
            _PIL_RESIZE_TO_INTERPOLATE_MODE = {
                Image.NEAREST: 'nearest',
                Image.BILINEAR: 'bilinear',
                Image.BICUBIC: 'bicubic',
            }
            mode = _PIL_RESIZE_TO_INTERPOLATE_MODE[interp_method]
            align_corners = None if mode == 'nearest' else False
            img = F.interpolate(
                img, (self.new_h, self.new_w),
                mode=mode,
                align_corners=align_corners)
            shape[:2] = (self.new_h, self.new_w)
            ret = img.permute(2, 3, 0, 1).view(shape).numpy()  # nchw -> hw(c)

        return ret
