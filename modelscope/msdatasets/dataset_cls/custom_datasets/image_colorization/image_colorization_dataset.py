# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.utils.constant import Tasks


def default_loader(path):
    return cv2.imread(path).astype(np.float32) / 255.0


@CUSTOM_DATASETS.register_module(
    Tasks.image_colorization, module_name=Models.ddcolor)
class ImageColorizationDataset(TorchCustomDataset):
    """Image dataset for image colorization.
    """

    def __init__(self, dataset, opt, is_train):
        self.dataset = dataset
        self.opt = opt
        self.input_size = 256
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        item_dict = self.dataset[index]
        gt_path = item_dict['Image:FILE']
        img_gt = default_loader(gt_path)

        # rezise to 256
        img_gt = cv2.resize(img_gt, (self.input_size, self.input_size))

        # get lq
        img_l = cv2.cvtColor(img_gt, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate(
            (img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
        tensor_lq_rgb = torch.from_numpy(img_gray_rgb.transpose(
            (2, 0, 1))).float()
        tensor_lq = torch.from_numpy(img_l.transpose((2, 0, 1))).float()

        # get ab
        img_ab = cv2.cvtColor(img_gt, cv2.COLOR_BGR2Lab)[:, :, 1:]
        tensor_gt_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        # gt_bgr
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        gt_rgb = torch.from_numpy(img_gt_rgb.transpose((2, 0, 1))).float()

        if self.is_train:
            return {'input': tensor_lq_rgb, 'target': tensor_gt_ab}
        else:
            return {
                'input': tensor_lq_rgb,
                'target': tensor_gt_ab,
                'img_l': tensor_lq,
                'gt_rgb': gt_rgb
            }
