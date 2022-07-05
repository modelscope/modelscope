from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ..base import Model, Pipeline
from ..builder import PIPELINES
from ..outputs import OutputKeys

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_to_image_synthesis,
    module_name=Pipelines.text_to_image_synthesis)
class TextToImageSynthesisPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        device_id = 0 if torch.cuda.is_available() else -1
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model, device_id=device_id)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError(
                f'expecting a Model instance or str, but get {type(model)}.')

        super().__init__(model=pipe_model, **kwargs)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model.generate(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {OutputKeys.OUTPUT_IMG: inputs}
