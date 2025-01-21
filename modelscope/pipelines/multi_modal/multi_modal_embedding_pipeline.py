# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.clip.model import CLIPForMultiModalEmbedding
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.multi_modal import CLIPPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_text_retrieval, module_name=Pipelines.multi_modal_embedding)
@PIPELINES.register_module(
    Tasks.multi_modal_embedding, module_name=Pipelines.multi_modal_embedding)
class MultiModalEmbeddingPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        if preprocessor is None:
            if isinstance(self.model, CLIPForMultiModalEmbedding):
                self.preprocessor = CLIPPreprocessor(self.model.model_dir)
            else:
                raise NotImplementedError

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(self.preprocess(input))

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
