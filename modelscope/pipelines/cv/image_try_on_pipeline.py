# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_try_on import try_on_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_try_on, module_name=Pipelines.image_try_on)
class SALForImageTryOnPipeline(Pipeline):
    r""" Image Try On Pipeline.
    Examples:
    >>> image_try_on = pipeline(Tasks.image_try_on, model='damo/cv_SAL-VTON_virtual-try-on', revision='v1.0.1')
    >>> input_images = {'person_input_path': '/your_path/image_try_on_person.jpg',
    >>>                 'garment_input_path': '/your_path/image_try_on_garment.jpg',
    >>>                 'mask_input_path': '/your_path/image_try_on_mask.jpg'}
    >>> result = image_try_on(input_images)
    >>> result[OutputKeys.OUTPUT_IMG]
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create image try on pipeline for prediction
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

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        try_on_image = try_on_infer.infer(self.model, self.model_path,
                                          input['person_input_path'],
                                          input['garment_input_path'],
                                          input['mask_input_path'],
                                          self.device)
        return {OutputKeys.OUTPUT_IMG: try_on_image}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
