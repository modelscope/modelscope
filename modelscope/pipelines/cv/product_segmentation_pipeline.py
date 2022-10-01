# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.models.cv.product_segmentation import seg_infer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.product_segmentation, module_name=Pipelines.product_segmentation)
class F3NetForProductSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create product segmentation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        mask = seg_infer.inference(self.model, self.device,
                                   input['input_path'])
        return {OutputKeys.MASKS: mask}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
