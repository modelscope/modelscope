# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.segmentation import SegDataset as _SegDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.image_segmentation, module_name=Datasets.SegDataset)
class SegDataset(_SegDataset):
    """EasyCV dataset for Sementic segmentation.
    For more details, please refer to :
    https://github.com/alibaba/EasyCV/blob/master/easycv/datasets/segmentation/raw.py .

    Args:
        data_source: Data source config to parse input data.
        pipeline: Sequence of transform object or config dict to be composed.
        ignore_index (int): Label index to be ignored.
        profiling: If set True, will print transform time.
    """
