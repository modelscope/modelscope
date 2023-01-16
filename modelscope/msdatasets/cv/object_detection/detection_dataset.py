# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

from easycv.datasets.detection import DetDataset as _DetDataset
from easycv.datasets.detection import \
    DetImagesMixDataset as _DetImagesMixDataset

from modelscope.metainfo import Datasets
from modelscope.msdatasets.cv.easycv_base import EasyCVBaseDataset
from modelscope.msdatasets.task_datasets import TASK_DATASETS
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    group_key=Tasks.image_object_detection, module_name=Datasets.DetDataset)
@TASK_DATASETS.register_module(
    group_key=Tasks.image_segmentation, module_name=Datasets.DetDataset)
class DetDataset(EasyCVBaseDataset, _DetDataset):
    """EasyCV dataset for object detection.
    For more details, please refer to https://github.com/alibaba/EasyCV/blob/master/easycv/datasets/detection/raw.py .

    Args:
        split_config (dict): Dataset root path from MSDataset, e.g.
            {"train":"local cache path"} or {"evaluation":"local cache path"}.
        preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. Not support yet.
        mode: Training or Evaluation.
        data_source: Data source config to parse input data.
        pipeline: Transform config list
        profiling: If set True, will print pipeline time
        classes: A list of class names, used in evaluation for result and groundtruth visualization
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
        _DetDataset.__init__(self, *args, **kwargs)


@TASK_DATASETS.register_module(
    group_key=Tasks.image_object_detection,
    module_name=Datasets.DetImagesMixDataset)
@TASK_DATASETS.register_module(
    group_key=Tasks.domain_specific_object_detection,
    module_name=Datasets.DetImagesMixDataset)
class DetImagesMixDataset(EasyCVBaseDataset, _DetImagesMixDataset):
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
        split_config (dict): Dataset root path from MSDataset, e.g.
            {"train":"local cache path"} or {"evaluation":"local cache path"}.
        preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied. Not support yet.
        mode: Training or Evaluation.
        data_source (:obj:`DetSourceCoco`): Data source config to parse input data.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        dynamic_scale (tuple[int], optional): The image scale can be changed
            dynamically. Default to None.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
        label_padding: out labeling padding [N, 120, 5]
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
        _DetImagesMixDataset.__init__(self, *args, **kwargs)
