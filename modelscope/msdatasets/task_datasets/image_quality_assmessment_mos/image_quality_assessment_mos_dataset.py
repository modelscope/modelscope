# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.preprocessors.cv import ImageQualityAssessmentMosPreprocessor
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    Tasks.image_quality_assessment_mos,
    module_name=Models.image_quality_assessment_mos)
class ImageQualityAssessmentMosDataset(TorchTaskDataset):
    """Paired image dataset for image quality assessment mos.
    """

    def __init__(self,
                 dataset,
                 opt,
                 preprocessor=ImageQualityAssessmentMosPreprocessor()):
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.opt = opt
        self.scale = 0.2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load input video paths.
        item_dict = self.dataset[index]
        iterm_mos = float(item_dict['mos']) * self.scale

        img = self.preprocessor(item_dict['image:FILE'])

        return {'input': img['input'].squeeze(0), 'target': iterm_mos}
