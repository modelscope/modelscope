# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.utils.constant import Tasks
from .data_utils import img2tensor, padding
from .transforms import augment, paired_random_crop


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


@TASK_DATASETS.register_module(
    Tasks.image_denoising, module_name=Models.nafnet)
class SiddImageDenoisingDataset(TorchTaskDataset):
    """Paired image dataset for image restoration.
    """

    def __init__(self, dataset, opt, is_train):
        self.dataset = dataset
        self.opt = opt
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        item_dict = self.dataset[index]
        gt_path = item_dict['Clean Image:FILE']
        img_gt = default_loader(gt_path)
        lq_path = item_dict['Noisy Image:FILE']
        img_lq = default_loader(lq_path)

        # augmentation for training
        if self.is_train:
            gt_size = self.opt.gt_size
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale=1)

            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt.use_flip,
                                     self.opt.use_rot)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        return {'input': img_lq, 'target': img_gt}
