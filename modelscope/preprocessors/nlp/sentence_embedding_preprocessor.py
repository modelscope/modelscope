# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_embedding)
class SentenceEmbeddingPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in sentence embedding.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, mode=mode, **kwargs)

    def __call__(self, data: Union[str, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data Dict:
                keys: "source_sentence" && "sentences_to_compare"
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
        source_sentence = data['source_sentence']
        compare_sentences = data['sentences_to_compare']
        sentences = []
        sentences.append(source_sentence[0])
        for sent in compare_sentences:
            sentences.append(sent)

        tokenized_inputs = self.tokenizer(
            sentences,
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            padding=True,
            truncation=True)
        return tokenized_inputs
