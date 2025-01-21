# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Models
from modelscope.msdatasets.dataset_cls.custom_datasets import (
    CUSTOM_DATASETS, TorchCustomDataset)
from modelscope.utils.constant import Tasks


@CUSTOM_DATASETS.register_module(
    Tasks.video_stabilization, module_name=Models.video_stabilization)
class VideoStabilizationDataset(TorchCustomDataset):
    """Paired video dataset for video stabilization.
    """

    def __init__(self, dataset, opt):
        self.dataset = dataset
        self.opt = opt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        # Load input video paths.
        item_dict = self.dataset[index]
        input_path = item_dict['input_video:FILE']

        return {'input': input_path}
