# Copyright (c) 2022 Zhipu.AI

from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp import CodeGeeXForCodeTranslation
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import CodeGeeXPreprocessor, Preprocessor
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    group_key=Tasks.code_translation,
    module_name=Pipelines.codegeex_code_translation)
class CodeGeeXCodeTranslationPipeline(Pipeline):

    def __init__(self,
                 model: Union[CodeGeeXForCodeTranslation, str],
                 preprocessor: [Preprocessor] = None,
                 *args,
                 **kwargs):
        model = CodeGeeXForCodeTranslation(model) if isinstance(model,
                                                                str) else model
        self.model = model
        self.model.eval()
        self.model.half()
        self.model.cuda()
        if preprocessor is None:
            preprocessor = CodeGeeXPreprocessor()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    # define the forward pass
    def forward(self, inputs: Union[Dict], **forward_params) -> Dict[str, Any]:
        # check input format
        for para in ['prompt', 'source language', 'target language']:
            if para not in inputs:
                return ('please check your input format.')
        return self.model(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
