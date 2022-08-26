# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.classification import ClsDataset as _ClsDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.image_classification, module_name=Datasets.ClsDataset)
class ClsDataset(_ClsDataset):
    """EasyCV dataset for classification.
    For more details, please refer to :
    https://github.com/alibaba/EasyCV/blob/master/easycv/datasets/classification/raw.py .

    Args:
        data_source: Data source config to parse input data.
        pipeline: Sequence of transform object or config dict to be composed.
    """
