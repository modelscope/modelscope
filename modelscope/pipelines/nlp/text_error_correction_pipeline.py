# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import BartForTextErrorCorrection
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TextErrorCorrectionPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['TextErrorCorrectionPipeline']


@PIPELINES.register_module(
    Tasks.text_error_correction, module_name=Pipelines.text_error_correction)
class TextErrorCorrectionPipeline(Pipeline):

    def __init__(
            self,
            model: Union[BartForTextErrorCorrection, str],
            preprocessor: Optional[TextErrorCorrectionPreprocessor] = None,
            **kwargs):
        """use `model` and `preprocessor` to create a nlp text correction pipeline.

        Args:
            model (BartForTextErrorCorrection): A model instance, or a model local dir, or a model id in the model hub.
            preprocessor (TextErrorCorrectionPreprocessor): An optional preprocessor instance.

        Example:
        >>> from modelscope.pipelines import pipeline
        >>> pipeline_ins = pipeline(
        >>>    task='text-error-correction', model='damo/nlp_bart_text-error-correction_chinese')
        >>> sentence1 = '随着中国经济突飞猛近，建造工业与日俱增'
        >>> print(pipeline_ins(sentence1))

        To view other examples plese check the tests/pipelines/test_text_error_correction.py.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

        if preprocessor is None:
            self.preprocessor = TextErrorCorrectionPreprocessor(
                self.model.model_dir)
        self.vocab = self.preprocessor.vocab

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        """
        Args:
            inputs (Dict[str, Tensor])
            Example:
                {
                    'predictions': Tensor([1377, 4959, 2785, 6392...]), # tokens need to be decode by tokenizer
                }
        Returns:
            Dict[str, str]
            Example:
            {
                'output': '随着中国经济突飞猛进，建造工业与日俱增'
            }


        """

        pred_str = self.vocab.string(
            inputs['predictions'],
            '@@',
            extra_symbols_to_ignore={self.vocab.pad()})

        return {OutputKeys.OUTPUT: ''.join(pred_str.split())}
