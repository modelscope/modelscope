# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np

from modelscope.fileio import File


def build_preprocess_transform(cfg):
    assert isinstance(cfg, dict)
    cfg = cfg.copy()
    type = cfg.pop('type')
    if type == 'LoadImageFromFile':
        return LoadImageFromFile(**cfg)
    elif type == 'LoadAnnotations':
        from mmdet.datasets.pipelines import LoadAnnotations
        return LoadAnnotations(**cfg)
    elif type == 'Resize':
        if 'img_scale' in cfg:
            if isinstance(cfg.img_scale[0], list):
                elems = []
                for elem in cfg.img_scale:
                    elems.append(tuple(elem))
                cfg.img_scale = elems
            else:
                cfg.img_scale = tuple(cfg.img_scale)
        from mmdet.datasets.pipelines import Resize
        return Resize(**cfg)
    elif type == 'RandomFlip':
        from mmdet.datasets.pipelines import RandomFlip
        return RandomFlip(**cfg)
    elif type == 'Normalize':
        from mmdet.datasets.pipelines import Normalize
        return Normalize(**cfg)
    elif type == 'Pad':
        from mmdet.datasets.pipelines import Pad
        return Pad(**cfg)
    elif type == 'DefaultFormatBundle':
        from mmdet.datasets.pipelines import DefaultFormatBundle
        return DefaultFormatBundle(**cfg)
    elif type == 'ImageToTensor':
        from mmdet.datasets.pipelines import ImageToTensor
        return ImageToTensor(**cfg)
    elif type == 'Collect':
        from mmdet.datasets.pipelines import Collect
        return Collect(**cfg)
    else:
        raise ValueError(f'preprocess transform \'{type}\' is not supported.')


class LoadImageFromFile:
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename", "ann_file", and "classes"). Added or updated keys are
    "filename", "ori_filename", "img", "img_shape", "ori_shape" (same as `img_shape`),
    "img_fields", "ann_file" (path to annotation file) and "classes".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
    """

    def __init__(self, to_float32=False, mode='rgb'):
        self.to_float32 = to_float32
        self.mode = mode

        from mmcv import imfrombytes

        self.imfrombytes = imfrombytes

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`ImageInstanceSegmentationCocoDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if 'img' in results and isinstance(results['img'], np.ndarray):
            img = results['img']
            filename = results['img_info']['filename']
        else:
            if results['img_prefix'] is not None:
                filename = osp.join(results['img_prefix'],
                                    results['img_info']['filename'])
            else:
                filename = results['img_info']['filename']

            img_bytes = File.read(filename)

            img = self.imfrombytes(img_bytes, 'color', 'bgr', backend='pillow')

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        results['ann_file'] = results['img_info']['ann_file']
        results['classes'] = results['img_info']['classes']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"mode='{self.mode}'")
        return repr_str
