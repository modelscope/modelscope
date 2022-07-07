from typing import Any, Dict

from modelscope.metainfo import Pipelines
from modelscope.pipelines.base import Input
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ..base import Model, Pipeline
from ..builder import PIPELINES

logger = get_logger()


@PIPELINES.register_module(
    Tasks.multi_modal_embedding, module_name=Pipelines.multi_modal_embedding)
class MultiModalEmbeddingPipeline(Pipeline):

    def __init__(self, model: str, device_id: int = -1):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError('model must be a single str')

        super().__init__(model=pipe_model)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        return input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
