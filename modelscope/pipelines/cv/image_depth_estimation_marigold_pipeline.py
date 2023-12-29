# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Union

from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_depth_estimation, module_name=Pipelines.image_depth_estimation_marigold)
class ImageDepthEstimationMarigoldPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        logger.info('depth estimation marigold model, pipeline init')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        print('pipeline preprocess')
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        self.input_image = Image.open(input)
        print('load', input, self.input_image.size)

        results = self.model.inference(self.input_image)
        return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = self.model.postprocess(inputs)
        depths = results[OutputKeys.DEPTHS]
        depths_color = results[OutputKeys.DEPTHS_COLOR]
        outputs = {
            OutputKeys.DEPTHS: depths,
            OutputKeys.DEPTHS_COLOR: depths_color
        }
        return outputs
