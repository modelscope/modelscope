# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.cv.bad_image_preprocessor import \
    BadImageDetectingPreprocessor
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    Tasks.bad_image_detecting, module_name=Models.bad_image_detecting)
class BadImageDetectingDataset(TorchTaskDataset):
    """Paired image dataset for bad image detecting.
    """

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt
        self.preprocessor = BadImageDetectingPreprocessor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load input video paths.
        item_dict = self.dataset[index]
        iterm_label = item_dict['label']

        img = LoadImage.convert_to_ndarray(input)
        img = self.preprocessor(img)

        return {'input': img['input'], 'target': iterm_label}
