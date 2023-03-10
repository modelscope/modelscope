"""
Part of the implementation is borrowed and modified from LaMa,
publicly available at https://github.com/saic-mdal/lama
"""
import glob
import os.path as osp
from enum import Enum

import albumentations as A
import cv2
import numpy as np

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .aug import IAAAffine2, IAAPerspective2

LOGGER = get_logger()


class LinearRamp:

    def __init__(self, start_value=0, end_value=1, start_iter=-1, end_iter=0):
        self.start_value = start_value
        self.end_value = end_value
        self.start_iter = start_iter
        self.end_iter = end_iter

    def __call__(self, i):
        if i < self.start_iter:
            return self.start_value
        if i >= self.end_iter:
            return self.end_value
        part = (i - self.start_iter) / (self.end_iter - self.start_iter)
        return self.start_value * (1 - part) + self.end_value * part


class DrawMethod(Enum):
    LINE = 'line'
    CIRCLE = 'circle'
    SQUARE = 'square'


def make_random_superres_mask(shape,
                              min_step=2,
                              max_step=4,
                              min_width=1,
                              max_width=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    step_x = np.random.randint(min_step, max_step + 1)
    width_x = np.random.randint(min_width, min(step_x, max_width + 1))
    offset_x = np.random.randint(0, step_x)

    step_y = np.random.randint(min_step, max_step + 1)
    width_y = np.random.randint(min_width, min(step_y, max_width + 1))
    offset_y = np.random.randint(0, step_y)

    for dy in range(width_y):
        mask[offset_y + dy::step_y] = 1
    for dx in range(width_x):
        mask[:, offset_x + dx::step_x] = 1
    return mask[None, ...]


class RandomSuperresMaskGenerator:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, img, iter_i=None):
        return make_random_superres_mask(img.shape[1:], **self.kwargs)


def make_random_rectangle_mask(shape,
                               margin=10,
                               bbox_min_size=30,
                               bbox_max_size=100,
                               min_times=0,
                               max_times=3):
    height, width = shape
    mask = np.zeros((height, width), np.float32)
    bbox_max_size = min(bbox_max_size, height - margin * 2, width - margin * 2)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        box_width = np.random.randint(bbox_min_size, bbox_max_size)
        box_height = np.random.randint(bbox_min_size, bbox_max_size)
        start_x = np.random.randint(margin, width - margin - box_width + 1)
        start_y = np.random.randint(margin, height - margin - box_height + 1)
        mask[start_y:start_y + box_height, start_x:start_x + box_width] = 1
    return mask[None, ...]


class RandomRectangleMaskGenerator:

    def __init__(self,
                 margin=10,
                 bbox_min_size=30,
                 bbox_max_size=100,
                 min_times=0,
                 max_times=3,
                 ramp_kwargs=None):
        self.margin = margin
        self.bbox_min_size = bbox_min_size
        self.bbox_max_size = bbox_max_size
        self.min_times = min_times
        self.max_times = max_times
        self.ramp = LinearRamp(
            **ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (
            iter_i is not None) else 1
        cur_bbox_max_size = int(self.bbox_min_size + 1
                                + (self.bbox_max_size - self.bbox_min_size)
                                * coef)
        cur_max_times = int(self.min_times
                            + (self.max_times - self.min_times) * coef)
        return make_random_rectangle_mask(
            img.shape[1:],
            margin=self.margin,
            bbox_min_size=self.bbox_min_size,
            bbox_max_size=cur_bbox_max_size,
            min_times=self.min_times,
            max_times=cur_max_times)


def make_random_irregular_mask(shape,
                               max_angle=4,
                               max_len=60,
                               max_width=20,
                               min_times=0,
                               max_times=10,
                               draw_method=DrawMethod.LINE):
    draw_method = DrawMethod(draw_method)

    height, width = shape
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(min_times, max_times + 1)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len)
            brush_w = 5 + np.random.randint(max_width)
            end_x = np.clip(
                (start_x + length * np.sin(angle)).astype(np.int32), 0, width)
            end_y = np.clip(
                (start_y + length * np.cos(angle)).astype(np.int32), 0, height)
            if draw_method == DrawMethod.LINE:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0,
                         brush_w)
            elif draw_method == DrawMethod.CIRCLE:
                cv2.circle(
                    mask, (start_x, start_y),
                    radius=brush_w,
                    color=1.,
                    thickness=-1)
            elif draw_method == DrawMethod.SQUARE:
                radius = brush_w // 2
                mask[start_y - radius:start_y + radius,
                     start_x - radius:start_x + radius] = 1
            start_x, start_y = end_x, end_y
    return mask[None, ...]


class RandomIrregularMaskGenerator:

    def __init__(self,
                 max_angle=4,
                 max_len=60,
                 max_width=20,
                 min_times=0,
                 max_times=10,
                 ramp_kwargs=None,
                 draw_method=DrawMethod.LINE):
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.min_times = min_times
        self.max_times = max_times
        self.draw_method = draw_method
        self.ramp = LinearRamp(
            **ramp_kwargs) if ramp_kwargs is not None else None

    def __call__(self, img, iter_i=None, raw_image=None):
        coef = self.ramp(iter_i) if (self.ramp is not None) and (
            iter_i is not None) else 1
        cur_max_len = int(max(1, self.max_len * coef))
        cur_max_width = int(max(1, self.max_width * coef))
        cur_max_times = int(self.min_times + 1
                            + (self.max_times - self.min_times) * coef)
        return make_random_irregular_mask(
            img.shape[1:],
            max_angle=self.max_angle,
            max_len=cur_max_len,
            max_width=cur_max_width,
            min_times=self.min_times,
            max_times=cur_max_times,
            draw_method=self.draw_method)


class MixedMaskGenerator:

    def __init__(self,
                 irregular_proba=1 / 3,
                 irregular_kwargs=None,
                 box_proba=1 / 3,
                 box_kwargs=None,
                 segm_proba=1 / 3,
                 segm_kwargs=None,
                 squares_proba=0,
                 squares_kwargs=None,
                 superres_proba=0,
                 superres_kwargs=None,
                 outpainting_proba=0,
                 outpainting_kwargs=None,
                 invert_proba=0):
        self.probas = []
        self.gens = []

        if irregular_proba > 0:
            self.probas.append(irregular_proba)
            if irregular_kwargs is None:
                irregular_kwargs = {}
            else:
                irregular_kwargs = dict(irregular_kwargs)
            irregular_kwargs['draw_method'] = DrawMethod.LINE
            self.gens.append(RandomIrregularMaskGenerator(**irregular_kwargs))

        if box_proba > 0:
            self.probas.append(box_proba)
            if box_kwargs is None:
                box_kwargs = {}
            self.gens.append(RandomRectangleMaskGenerator(**box_kwargs))

        if squares_proba > 0:
            self.probas.append(squares_proba)
            if squares_kwargs is None:
                squares_kwargs = {}
            else:
                squares_kwargs = dict(squares_kwargs)
            squares_kwargs['draw_method'] = DrawMethod.SQUARE
            self.gens.append(RandomIrregularMaskGenerator(**squares_kwargs))

        if superres_proba > 0:
            self.probas.append(superres_proba)
            if superres_kwargs is None:
                superres_kwargs = {}
            self.gens.append(RandomSuperresMaskGenerator(**superres_kwargs))

        self.probas = np.array(self.probas, dtype='float32')
        self.probas /= self.probas.sum()
        self.invert_proba = invert_proba

    def __call__(self, img, iter_i=None, raw_image=None):
        kind = np.random.choice(len(self.probas), p=self.probas)
        gen = self.gens[kind]
        result = gen(img, iter_i=iter_i, raw_image=raw_image)
        if self.invert_proba > 0 and random.random() < self.invert_proba:
            result = 1 - result
        return result


def get_transforms(test_mode, out_size):
    if not test_mode:
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3), rotate=(-40, 40), shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    else:
        transform = A.Compose([
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.CenterCrop(height=out_size, width=out_size),
            A.ToFloat()
        ])
    return transform


@CUSTOM_DATASETS.register_module(
    Tasks.image_inpainting, module_name=Models.image_inpainting)
class ImageInpaintingDataset(TorchCustomDataset):

    def __init__(self, **kwargs):
        split_config = kwargs['split_config']
        LOGGER.info(kwargs)
        mode = kwargs.get('test_mode', False)

        self.data_root = next(iter(split_config.values()))
        if not osp.exists(self.data_root):
            self.data_root = osp.dirname(self.data_root)
            assert osp.exists(self.data_root)
        mask_gen_kwargs = kwargs.get('mask_gen_kwargs', {})
        out_size = kwargs.get('out_size', 256)
        self.mask_generator = MixedMaskGenerator(**mask_gen_kwargs)
        self.transform = get_transforms(mode, out_size)
        self.in_files = sorted(
            list(
                glob.glob(
                    osp.join(self.data_root, '**', '*.jpg'), recursive=True))
            + list(
                glob.glob(
                    osp.join(self.data_root, '**', '*.png'), recursive=True)))
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, index):
        path = self.in_files[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img, mask=mask)
