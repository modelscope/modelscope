# Copyright (c) 2022 Zhipu.AI

from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.models.nlp import TXLForFastPoem
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor, TXLFastPoemPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['TXLFastPoemPipeline']


@PIPELINES.register_module(
    group_key=Tasks.fast_poem, module_name=Pipelines.txl_fast_poem)
class TXLFastPoemPipeline(Pipeline):

    def __init__(self,
                 model: Union[TXLForFastPoem, str],
                 preprocessor: [Preprocessor] = None,
                 *args,
                 **kwargs):
        model = TXLForFastPoem(model) if isinstance(model, str) else model
        self.model = model
        self.model.eval()
        if preprocessor is None:
            preprocessor = TXLFastPoemPreprocessor()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    # define the forward pass
    def forward(self, inputs: Union[Dict, str],
                **forward_params) -> Dict[str, Any]:
        if isinstance(inputs, str):
            inputs = {
                'title': inputs,
                'author': '李白',
                'desc': '寂寞',
                'lycr': 7,
                'senlength': 4
            }
        else:
            if 'title' not in inputs:
                inputs['title'] = '月光'
            if 'author' not in inputs:
                inputs['author'] = '李白'
            if 'desc' not in inputs:
                inputs['desc'] = '寂寞'
            if 'lycr' not in inputs:
                inputs['lycr'] = 7
            if 'senlength' not in inputs:
                inputs['senlength'] = 4

        return self.model.generate(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
