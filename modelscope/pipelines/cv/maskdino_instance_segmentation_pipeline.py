# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch
import torchvision.transforms as T

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_instance_segmentation import (
    MaskDINOSwinModel, get_maskdino_ins_seg_result)
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_segmentation,
    module_name=Pipelines.maskdino_instance_segmentation)
class MaskDINOInstanceSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[MaskDINOSwinModel, str],
                 preprocessor: Optional = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a MaskDINO instance segmentation
        pipeline for prediction

        Args:
            model (MaskDINOSwinModel | str): a model instance
            preprocessor (None): a preprocessor instance
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()

    def get_preprocess_shape(self, oldh, oldw, short_edge_length, max_size):
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        image = LoadImage.convert_to_img(input)
        w, h = image.size[:2]
        new_h, new_w = self.get_preprocess_shape(h, w, 800, 1333)
        test_transforms = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
        ])
        image = test_transforms(image)
        dataset_dict = {}
        dataset_dict['height'] = h
        dataset_dict['width'] = w
        dataset_dict['image'] = image
        result = {'batched_inputs': [dataset_dict]}
        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.model(input)
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        result = get_maskdino_ins_seg_result(
            inputs['eval_result'][0]['instances'],
            class_names=self.model.model.classes)
        return result
