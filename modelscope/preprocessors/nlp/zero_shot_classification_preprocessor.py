# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from .transformers_tokenizer import NLPTokenizer


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.zero_shot_cls_tokenizer)
class ZeroShotClassificationTransformersPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in zero shot classification.
    """

    def __init__(self,
                 model_dir: str,
                 first_sequence=None,
                 mode=ModeKeys.INFERENCE,
                 max_length=None,
                 use_fast=None,
                 **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 512)
        kwargs.pop('sequence_length', None)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        self.first_sequence = first_sequence
        super().__init__(mode=mode)

    def __call__(self,
                 data: Union[str, Dict],
                 hypothesis_template: str,
                 candidate_labels: list,
                 padding=True,
                 truncation=True,
                 truncation_strategy='only_first',
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str or dict): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, dict):
            data = data.get(self.first_sequence)

        pairs = [[data, hypothesis_template.format(label)]
                 for label in candidate_labels]

        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self._mode == ModeKeys.INFERENCE else None

        features = self.nlp_tokenizer(
            pairs,
            padding=padding,
            truncation=truncation,
            truncation_strategy=truncation_strategy,
            **kwargs)
        return features
