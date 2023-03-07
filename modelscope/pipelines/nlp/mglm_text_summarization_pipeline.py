# Copyright (c) 2022 Zhipu.AI

import os
from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp import MGLMForTextSummarization
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import (MGLMSummarizationPreprocessor,
                                      Preprocessor)
from modelscope.utils.constant import Tasks

__all__ = ['MGLMTextSummarizationPipeline']


@PIPELINES.register_module(
    group_key=Tasks.text_summarization,
    module_name=Pipelines.mglm_text_summarization)
class MGLMTextSummarizationPipeline(Pipeline):

    def __init__(self,
                 model: Union[MGLMForTextSummarization, str],
                 preprocessor: Optional[Preprocessor] = None,
                 *args,
                 **kwargs):
        model = MGLMForTextSummarization(model) if isinstance(model,
                                                              str) else model
        self.model = model
        self.model.eval()
        if preprocessor is None:
            preprocessor = MGLMSummarizationPreprocessor()
        from modelscope.utils.torch_utils import _find_free_port
        os.environ['MASTER_PORT'] = str(_find_free_port())
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    # define the forward pass
    def forward(self, inputs: Union[Dict, str],
                **forward_params) -> Dict[str, Any]:
        inputs = {'text': inputs} if isinstance(inputs, str) else inputs
        return self.model.generate(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
