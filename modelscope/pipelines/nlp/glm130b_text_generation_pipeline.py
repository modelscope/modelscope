# Copyright (c) 2022 Zhipu.AI

from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models.nlp import GLM130bForTextGeneration
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    group_key=Tasks.text_generation,
    module_name=Pipelines.glm130b_text_generation)
class GLM130bTextGenerationPipeline(Pipeline):

    def __init__(self, model: Union[GLM130bForTextGeneration, str], *args,
                 **kwargs):
        model = GLM130bForTextGeneration(model) if isinstance(model,
                                                              str) else model
        self.model = model

    def __call__(self, input: str, **forward_params) -> Dict[str, Any]:
        return self.model(input)

    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        """This method will not be called.
        """
        return input
