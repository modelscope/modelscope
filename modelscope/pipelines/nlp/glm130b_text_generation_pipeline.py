# Copyright (c) 2022 Zhipu.AI

from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models.nlp import GLM130bForTextGeneration
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    group_key=Tasks.text_generation,
    module_name=Pipelines.glm130b_text_generation)
class GLM130bTextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[GLM130bForTextGeneration, str],
                 preprocessor: [Preprocessor] = None,
                 *args,
                 **kwargs):
        model = GLM130bForTextGeneration(model) if isinstance(model,
                                                              str) else model
        self.model = model
        super().__init__(model=model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> str:
        return inputs

    # define the forward pass
    def forward(self, inputs: str, **forward_params) -> Dict[str, Any]:
        return self.model(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
