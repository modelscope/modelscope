# Copyright (c) 2022 Zhipu.AI

from typing import Any, Dict, Union

from modelscope.metainfo import Pipelines
from modelscope.models.nlp import ChatGLM6bForTextGeneration
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    group_key=Tasks.text_generation,
    module_name=Pipelines.chatglm6b_text_generation)
class ChatGLM6bTextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[ChatGLM6bForTextGeneration, str],
                 preprocessor: [Preprocessor] = None,
                 *args,
                 **kwargs):
        model = ChatGLM6bForTextGeneration(model) if isinstance(model,
                                                               str) else model
        self.model = model
        self.model.eval()

        super().__init__(model=model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        return self.model(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input