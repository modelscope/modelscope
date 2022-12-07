# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger
from .transformers_tokenizer import NLPTokenizer
from .utils import labels_to_id, parse_text_and_label

logger = get_logger()


class TextClassificationPreprocessorBase(Preprocessor):

    def __init__(
        self,
        model_dir=None,
        first_sequence: str = None,
        second_sequence: str = None,
        label: str = 'label',
        label2id: Dict = None,
        mode: str = ModeKeys.INFERENCE,
    ):
        """The base class for the text classification preprocessor.

        Args:
            model_dir(str, `optional`): The model dir used to parse the label mapping, can be None.
            first_sequence(str, `optional`): The key of the first sequence.
            second_sequence(str, `optional`): The key of the second sequence.
            label(str, `optional`): The keys of the label columns, default is `label`
            label2id: (dict, `optional`): The optional label2id mapping
            mode: The mode for the preprocessor
        """
        super().__init__(mode)
        self.model_dir = model_dir
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.label = label
        self.label2id = label2id
        if self.label2id is None and self.model_dir is not None:
            self.label2id = parse_label_mapping(self.model_dir)

        logger.info(f'The key of sentence1: {self.first_sequence}, '
                    f'The key of sentence2: {self.second_sequence}, '
                    f'The key of label: {self.label}')
        if self.first_sequence is None:
            logger.warning('[Important] first_sequence attribute is not set, '
                           'this will cause an error if your input is a dict.')

    @property
    def id2label(self):
        """Return the id2label mapping according to the label2id mapping.

        @return: The id2label mapping if exists.
        """
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def __call__(self, data: Union[str, Tuple, Dict],
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (tuple): [sentence1, sentence2]
                sentence1 (str): a sentence
                    Example:
                        'you are so handsome.'
                sentence2 (str): a sentence
                    Example:
                        'you are so beautiful.'
        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a, text_b, labels = parse_text_and_label(data, self.mode,
                                                      self.first_sequence,
                                                      self.second_sequence,
                                                      self.label)
        output = self._tokenize_text(text_a, text_b, **kwargs)
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        labels_to_id(labels, output, self.label2id)
        return output

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        raise NotImplementedError()


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.nli_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_sim_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.bert_seq_cls_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_cls_tokenizer)
class TextClassificationTransformersPreprocessor(
        TextClassificationPreprocessorBase):

    def _tokenize_text(self, sequence1, sequence2=None, **kwargs):
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        return self.nlp_tokenizer(sequence1, sequence2, **kwargs)

    def __init__(self,
                 model_dir=None,
                 first_sequence: str = None,
                 second_sequence: str = None,
                 label: Union[str, List] = 'label',
                 label2id: Dict = None,
                 mode: str = ModeKeys.INFERENCE,
                 max_length: int = None,
                 use_fast: bool = None,
                 **kwargs):
        """The tokenizer preprocessor used in sequence classification.

        Args:
            use_fast: Whether to use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(model_dir, first_sequence, second_sequence, label,
                         label2id, mode)
