# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type
from .transformers_tokenizer import NLPTokenizer


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_embedding)
class SentenceEmbeddingTransformersPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in sentence embedding.
    """

    def __init__(self,
                 model_dir: str,
                 first_sequence='source_sentence',
                 second_sequence='sentences_to_compare',
                 mode=ModeKeys.INFERENCE,
                 use_fast: bool = True,
                 max_length: int = None,
                 **kwargs):
        """The preprocessor for sentence embedding task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir used to initialize the tokenizer.
            first_sequence: The key of the first sequence.
            second_sequence: The key of the second sequence.
            mode: The mode for the preprocessor.
            use_fast: Use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        model_type = None
        self.max_length = max_length
        if model_dir is not None:
            model_type = get_model_type(model_dir)
        self.nlp_tokenizer = NLPTokenizer(
            model_dir, model_type, use_fast=use_fast, tokenize_kwargs=kwargs)
        super().__init__(mode=mode)

    def __call__(self,
                 data: Dict,
                 padding=True,
                 truncation=True,
                 **kwargs) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict:
                keys: the source sentence and the sentences to compare
                values: list of sentences
                Example:
                    {"source_sentence": ["how long it take to get a master's degree"],
                     "sentences_to_compare": ["On average, students take about 18 to 24 months
                     to complete a master's degree.",
                     "On the other hand, some students prefer to go at a slower pace
                     and choose to take several years to complete their studies.",
                     "It can take anywhere from two semesters"]}
        Returns:
            Dict[str, Any]: the preprocessed data
        """
        source_sentences = data[self.first_sequence]
        if self.second_sequence in data:
            if isinstance(source_sentences[0], list):
                source_sentences = [source_sentences[0]]
            compare_sentences = data[self.second_sequence]
        else:
            compare_sentences = None
        if 'return_tensors' not in kwargs:
            kwargs[
                'return_tensors'] = 'pt' if self.mode == ModeKeys.INFERENCE else None
        query_inputs = self.nlp_tokenizer(
            source_sentences, padding=padding, truncation=truncation, **kwargs)
        tokenized_inputs = {'query': query_inputs, 'docs': None}
        if compare_sentences is not None and len(compare_sentences) > 0:
            tokenized_inputs['docs'] = self.nlp_tokenizer(
                compare_sentences,
                padding=padding,
                truncation=truncation,
                **kwargs)
        return tokenized_inputs
