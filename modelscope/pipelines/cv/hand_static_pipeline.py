# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Any, Dict

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models.cv.hand_static import hand_model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.hand_static, module_name=Pipelines.hand_static)
class HandStaticPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create hand static pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        return img

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        result = hand_model.infer(input, self.model, self.device)
        return {OutputKeys.OUTPUT: result}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
