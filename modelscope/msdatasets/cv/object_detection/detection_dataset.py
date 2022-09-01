# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.detection import DetDataset as _DetDataset
from easycv.datasets.detection import \
    DetImagesMixDataset as _DetImagesMixDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.task_datasets import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.image_object_detection, module_name=Datasets.DetDataset)
class DetDataset(_DetDataset):
    """EasyCV dataset for object detection.
    For more details, please refer to https://github.com/alibaba/EasyCV/blob/master/easycv/datasets/detection/raw.py .

    Args:
        data_source: Data source config to parse input data.
        pipeline: Transform config list
        profiling: If set True, will print pipeline time
        classes: A list of class names, used in evaluation for result and groundtruth visualization
    """


@TASK_DATASETS.register_module(
    group_key=Tasks.image_object_detection,
    module_name=Datasets.DetImagesMixDataset)
class DetImagesMixDataset(_DetImagesMixDataset):
    """EasyCV dataset for object detection, a wrapper of multiple images mixed dataset.
    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process. At the same time, we provide the `dynamic_scale` parameter
    to dynamically change the output image size.
    output boxes format: cx, cy, w, h

    For more details, please refer to https://github.com/alibaba/EasyCV/blob/master/easycv/datasets/detection/mix.py .

    Args:
        data_source (:obj:`DetSourceCoco`): Data source config to parse input data.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        label_padding: out labeling padding [N, 120, 5]
    """
