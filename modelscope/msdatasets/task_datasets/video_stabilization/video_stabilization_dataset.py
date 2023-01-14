# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    Tasks.video_stabilization, module_name=Models.video_stabilization)
class VideoStabilizationDataset(TorchTaskDataset):
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
