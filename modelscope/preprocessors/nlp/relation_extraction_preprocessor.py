# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert
from .nlp_base import NLPBasePreprocessor


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.re_tokenizer)
class RelationExtractionPreprocessor(NLPBasePreprocessor):
    """The relation extraction preprocessor used in normal RE task.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """

        super().__init__(model_dir, *args, **kwargs)

        self.model_dir: str = model_dir
        self.sequence_length = kwargs.pop('sequence_length', 512)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True)

    @type_assert(object, str)
    def __call__(self, data: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        # preprocess the data for the model input
        text = data
        output = self.tokenizer([text], return_tensors='pt')
        return {
            'text': text,
            'input_ids': output['input_ids'],
            'attention_mask': output['attention_mask'],
            'offsets': output[0].offsets
        }
