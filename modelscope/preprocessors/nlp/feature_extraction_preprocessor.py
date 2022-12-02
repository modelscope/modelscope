# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Tuple, Union

import numpy as np

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from .transformers_tokenizer import NLPTokenizer
from .utils import parse_text_and_label


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.feature_extraction)
class FeatureExtractionTransformersPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str = None,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 mode: str = ModeKeys.INFERENCE,
                 max_length: int = None,
                 use_fast: bool = None,
                 **kwargs):
        """The preprocessor for feature extraction task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(mode)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)

    def __call__(self, data: Union[str, Tuple, Dict],
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a, text_b, _ = parse_text_and_label(data, self.mode,
                                                 self.first_sequence,
                                                 self.second_sequence)
        output = self._tokenize_text(text_a, text_b, **kwargs)
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        return output

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        return self.nlp_tokenizer(sequence1, sequence2, **kwargs)
