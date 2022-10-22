from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp import mGlmForTextSummarization
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks

__all__ = ['mglmTextSummarizationPipeline']


@PIPELINES.register_module(
    group_key=Tasks.summarization,
    module_name=Pipelines.mglm_text_summarization)
class mglmTextSummarizationPipeline(Pipeline):

    def __init__(self, model: Union[mGlmForTextSummarization, str], *args,
                 **kwargs):
        model = mGlmForTextSummarization(model) if isinstance(model,
                                                              str) else model
        self.model = model
        self.model.eval()
        super().__init__(model=model, **kwargs)

    # define the forward pass
    def forward(self, inputs: Union[Dict, str],
                **forward_params) -> Dict[str, Any]:
        inputs = {'text': inputs} if isinstance(inputs, str) else inputs
        return self.model.generate(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
