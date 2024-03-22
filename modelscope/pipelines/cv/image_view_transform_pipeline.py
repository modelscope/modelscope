# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_view_transform import \
    image_view_transform_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_view_transform, module_name=Pipelines.image_view_transform)
class ImageViewTransformPipeline(Pipeline):
    r""" Image View Transform Pipeline.
    Examples:
    >>> image_view_transform = pipeline(Tasks.image_view_transform,
    >>>.                                  model='damo/image_view_transform', revision='v1.0.0')
    >>> input_images = {'source_img_path': '/your_path/image_view_transform_source_img.jpg',
    >>>                 'target_view_path': '/your_path/image_view_transform_target_view.txt'}
    >>> result = image_view_transform(input_images)
    >>> result[OutputKeys.OUTPUT_IMG]
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create image view translation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        self.model_path = model
        logger.info('load model done')
        if torch.cuda.is_available():
            self.device = 'cuda'
            logger.info('Use GPU')
        else:
            self.device = 'cpu'
            logger.info('Use CPU')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        image_view_transform_imgs = image_view_transform_infer.infer(
            self.model, self.model_path, input['source_img'],
            input['target_view'], self.device)
        return {OutputKeys.OUTPUT_IMGS: image_view_transform_imgs}
