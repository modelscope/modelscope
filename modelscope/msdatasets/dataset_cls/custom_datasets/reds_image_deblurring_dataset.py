# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from modelscope.metainfo import CustomDatasets
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.msdatasets.dataset_cls.custom_datasets.sidd_image_denoising.data_utils import (
    img2tensor, padding)
from modelscope.msdatasets.dataset_cls.custom_datasets.sidd_image_denoising.transforms import (
    augment, paired_random_crop)
from modelscope.utils.constant import Tasks


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


@CUSTOM_DATASETS.register_module(
    Tasks.image_deblurring, module_name=CustomDatasets.RedsDataset)
class RedsImageDeblurringDataset(TorchCustomDataset):
    """Paired image dataset for image restoration.
    """

    def __init__(self, dataset, opt, is_train):
        self.dataset = dataset
        self.opt = opt
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item_dict = self.dataset[index]
        hq_path = item_dict['LQ Frame:FILE']
        img_hq = default_loader(hq_path)
        lq_path = item_dict['HQ Frame:FILE']
        img_lq = default_loader(lq_path)

        # augmentation for training
        if self.is_train:
            gt_size = self.opt.gt_size
            # padding
            img_hq, img_lq = padding(img_hq, img_lq, gt_size)

            # random crop
            img_hq, img_lq = paired_random_crop(
                img_hq, img_lq, gt_size, scale=1)

            # flip, rotation
            img_hq, img_lq = augment([img_hq, img_lq], self.opt.use_flip,
                                     self.opt.use_rot)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_hq, img_lq = img2tensor([img_hq, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        return {'input': img_lq, 'target': img_hq}
