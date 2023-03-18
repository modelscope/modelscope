# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from modelscope.metainfo import CustomDatasets
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.utils.constant import Tasks
from .data_utils import img2tensor


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0


@CUSTOM_DATASETS.register_module(
    Tasks.image_portrait_enhancement, module_name=CustomDatasets.PairedDataset)
class ImagePortraitEnhancementDataset(TorchCustomDataset):
    """Paired image dataset for image portrait enhancement.
    """

    def __init__(self, dataset, is_train):
        self.dataset = dataset
        self.gt_size = 256
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        item_dict = self.dataset[index]
        gt_path = item_dict['hq:FILE']
        img_gt = default_loader(gt_path)
        lq_path = item_dict['lq:FILE']
        img_lq = default_loader(lq_path)

        gt_size = self.gt_size
        img_gt = cv2.resize(img_gt, (gt_size, gt_size))
        img_lq = cv2.resize(img_lq, (gt_size, gt_size))

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        return {'input': (img_lq - 0.5) / 0.5, 'target': (img_gt - 0.5) / 0.5}
