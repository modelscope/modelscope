# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.face import FaceKeypointDataset as _FaceKeypointDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.face_2d_keypoints,
    module_name=Datasets.Face2dKeypointsDataset)
class FaceKeypointDataset(_FaceKeypointDataset):
    """EasyCV dataset for face 2d keypoints."""
