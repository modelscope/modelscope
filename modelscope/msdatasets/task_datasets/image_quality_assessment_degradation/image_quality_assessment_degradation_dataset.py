# Copyright (c) Alibaba, Inc. and its affiliates.

import cv2
import numpy as np
from torchvision import transforms

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    Tasks.image_quality_assessment_degradation,
    module_name=Models.image_quality_assessment_degradation)
class ImageQualityAssessmentDegradationDataset(TorchTaskDataset):
    """Paired image dataset for image quality assessment degradation.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load input video paths.
        item_dict = self.dataset[index]
        item_id = item_dict['image:FILE'].split('/')[-1].split('_')[0]
        item_degree = item_dict['degree']
        item_distortion_type = '%02d' % item_dict['degradation_category']

        img = LoadImage.convert_to_img(item_dict['image:FILE'])
        w, h = img.size
        if h * w < 1280 * 720:
            img = transforms.functional.resize(img, 720)
        test_transforms = transforms.Compose([transforms.ToTensor()])
        img = test_transforms(img)

        return {
            'input': img,
            'item_id': item_id,
            'target': item_degree,
            'distortion_type': item_distortion_type
        }
