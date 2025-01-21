# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.cv.bad_image_detecting_preprocessor import \
    BadImageDetectingPreprocessor
from modelscope.utils.constant import Tasks


@CUSTOM_DATASETS.register_module(
    Tasks.bad_image_detecting, module_name=Models.bad_image_detecting)
class BadImageDetectingDataset(TorchCustomDataset):
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
        iterm_label = item_dict['category']
        # print(input)
        img = LoadImage.convert_to_ndarray(item_dict['image:FILE'])
        img = self.preprocessor(img)
        return {
            'input': img['input'].squeeze(0),
            OutputKeys.LABEL: iterm_label
        }
