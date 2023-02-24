# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_driving_perception import (
    ImageDrivingPerceptionPreprocessor, driving_area_mask, lane_line_mask,
    non_max_suppression, scale_coords, split_for_trace_model)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_driving_perception,
    module_name=Pipelines.yolopv2_image_driving_percetion_bdd100k)
class ImageDrivingPerceptionPipeline(Pipeline):
    """ Image Driving Perception Pipeline. Given a image,
    pipeline will detects cars, and segments both lane lines and drivable areas.
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> image_driving_perception_pipeline = pipeline(Tasks.image_driving_perception,
                                                        model='damo/cv_yolopv2_image-driving-perception_bdd100k')
    >>> image_driving_perception_pipeline(img_path)
    {
        'boxes': array([[1.0000e+00, 2.8600e+02, 4.0700e+02, 6.2600e+02],
                        [8.8200e+02, 2.9600e+02, 1.0910e+03, 4.4700e+02],
                        [3.7200e+02, 2.7500e+02, 5.2100e+02, 3.5500e+02],
                        ...,
                        [7.8600e+02, 2.8100e+02, 8.0400e+02, 3.0800e+02],
                        [5.7000e+02, 2.8000e+02, 5.9400e+02, 3.0000e+02],
                        [7.0500e+02, 2.7800e+02, 7.2100e+02, 2.9000e+02]], dtype=float32)
        'masks': [
                    array([[0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            ...,
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0]], dtype=int32),
                    array([[0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            ...,
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0],
                            [0, 0, 0, ..., 0, 0, 0]], dtype=int32)
                ]
    }
    >>> #
    ```
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` and 'preprocessor' to create a image driving percetion pipeline for prediction
        """
        super().__init__(model=model, auto_collate=True, **kwargs)
        if self.preprocessor is None:
            self.preprocessor = ImageDrivingPerceptionPreprocessor()
        logger.info('load model done')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results_dict = {
            OutputKeys.BOXES: [],
            OutputKeys.MASKS: [],
        }

        pred = split_for_trace_model(inputs['pred'], inputs['anchor_grid'])

        # Apply NMS
        pred = non_max_suppression(pred)

        h, w = inputs['ori_img_shape']
        da_seg_mask = driving_area_mask(
            inputs['driving_area_mask'], out_shape=(h, w))
        ll_seg_mask = lane_line_mask(
            inputs['lane_line_mask'], out_shape=(h, w))

        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to (h, w)
                det[:, :4] = scale_coords(inputs['img_hw'], det[:, :4],
                                          (h, w)).round()

        results_dict[OutputKeys.BOXES] = det[:, :4].cpu().numpy()
        results_dict[OutputKeys.MASKS].append(da_seg_mask)
        results_dict[OutputKeys.MASKS].append(ll_seg_mask)
        return results_dict
