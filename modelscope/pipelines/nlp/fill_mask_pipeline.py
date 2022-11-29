# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Optional, Union

import numpy as np

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Tasks

__all__ = ['FillMaskPipeline']


@PIPELINES.register_module(Tasks.fill_mask, module_name=Pipelines.fill_mask)
@PIPELINES.register_module(
    Tasks.fill_mask, module_name=Pipelines.fill_mask_ponet)
class FillMaskPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 first_sequence: str = 'sentence',
                 **kwargs):
        """The inference pipeline for all the fill mask sub-tasks.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            preprocessor (`Preprocessor`, `optional`): A Preprocessor instance.
            first_sequence (`str`， `optional`): The key to read the sentence in.
            sequence_length (`int`， `optional`): Max sequence length in the user's custom scenario, default 128.

            NOTE1: Inputs of type 'str' are also supported. In this scenario, the 'first_sequence'
            param will have no effect.

            Example1:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('fill-mask', model='damo/nlp_structbert_fill-mask_english-large')
            >>> input = 'Everything in [MASK] you call reality is really [MASK] a reflection of your [MASK].'
            >>> print(pipeline_ins(input))
            Example2:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('fill-mask', model='damo/nlp_ponet_fill-mask_english-base')
            >>> input = 'Everything in [MASK] you call reality is really [MASK] a reflection of your [MASK].'
            >>> print(pipeline_ins(input))

            NOTE2: Please pay attention to the model's special tokens.
            If bert based model(bert, structbert, etc.) is used, the mask token is '[MASK]'.
            If the xlm-roberta(xlm-roberta, veco, etc.) based model is used, the mask token is '<mask>'.
            To view other examples plese check the tests/pipelines/test_fill_mask.py.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir,
                first_sequence=first_sequence,
                second_sequence=None,
                sequence_length=kwargs.pop('sequence_length', 128))
            assert hasattr(
                self.preprocessor, 'mask_id'
            ), 'The input preprocessor should have the mask_id attribute.'

        self.model.eval()

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        return self.model(**inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): The model outputs.
            The output should follow some rules:
                1. Values can be retrieved by keys(dict-like, or the __getitem__ method is overriden)
                2. 'logits' and 'input_ids' key exists.
            Models in modelscope will return the output dataclass `modelscope.outputs.FillMaskModelOutput`.
        Returns:
            Dict[str, str]: the prediction results
        """
        logits = inputs[OutputKeys.LOGITS].detach().cpu().numpy()
        input_ids = inputs[OutputKeys.INPUT_IDS].detach().cpu().numpy()
        pred_ids = np.argmax(logits, axis=-1)
        rst_ids = np.where(input_ids == self.preprocessor.mask_id, pred_ids,
                           input_ids)

        pred_strings = []
        for ids in rst_ids:  # batch
            pred_string = self.preprocessor.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)
            pred_strings.append(pred_string)

        return {OutputKeys.TEXT: pred_strings}
