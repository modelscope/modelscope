# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_control_3d_portrait,
    module_name=Pipelines.image_control_3d_portrait)
class ImageControl3dPortraitPipeline(Pipeline):
    """ Image control 3d portrait synthesis pipeline
    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> image_control_3d_portrait = pipeline(Tasks.image_control_3d_portrait,
                'damo/cv_vit_image-control-3d-portrait-synthesis')
    >>> image_control_3d_portrait({
            'image_path': 'input.jpg', # input image path (str)
            'save_dir': 'save_dir', # save dir path (str)
        })
    >>>
    ```
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create image_control_3D_portrait pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        logger.info('image control 3D portrait synthesis model init done')

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        image_path = input['image']
        save_dir = input['save_dir']
        self.model.inference(image_path, save_dir)
        return {OutputKeys.OUTPUT: 'Done'}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
