# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import defaultdict

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.utils.constant import Tasks


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


@CUSTOM_DATASETS.register_module(
    Tasks.video_super_resolution, module_name=Models.real_basicvsr)
class VideoSuperResolutionDataset(TorchCustomDataset):
    """single video dataset for video super-resolution.
    """

    def __init__(self, dataset):
        frames_len = len(dataset)
        self.dataset = defaultdict(list)
        for i in range(frames_len):
            item_dict = dataset[i]
            frame_path = item_dict['LQ Frame:FILE']
            clip_num = item_dict['Clip Num']
            self.dataset[int(clip_num)].append(frame_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load frames. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        clip = self.dataset[index]
        frames = []

        for frame_path in clip:
            frame = default_loader(frame_path)
            # BGR to RGB, HWC to CHW, numpy to tensor
            frames.append(img2tensor(frame, bgr2rgb=True, float32=True))

        input = torch.stack(frames, dim=0)  # (T, C, H, W)
        return {'input': input, 'target': input}
