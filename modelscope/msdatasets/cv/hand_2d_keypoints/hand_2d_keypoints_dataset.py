# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.pose import \
    HandCocoWholeBodyDataset as _HandCocoWholeBodyDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.cv.easycv_base import EasyCVBaseDataset
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.hand_2d_keypoints,
    module_name=Datasets.HandCocoWholeBodyDataset)
class HandCocoWholeBodyDataset(EasyCVBaseDataset, _HandCocoWholeBodyDataset):
    """EasyCV dataset for human hand 2d keypoints.

    Args:
        split_config (dict): Dataset root path from MSDataset, e.g.
            {"train":"local cache path"} or {"evaluation":"local cache path"}.
        preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. Not support yet.
        mode: Training or Evaluation.
    """

    def __init__(self,
                 split_config=None,
                 preprocessor=None,
                 mode=None,
                 *args,
                 **kwargs) -> None:
        EasyCVBaseDataset.__init__(
            self,
            split_config=split_config,
            preprocessor=preprocessor,
            mode=mode,
            args=args,
            kwargs=kwargs)
        _HandCocoWholeBodyDataset.__init__(self, *args, **kwargs)
