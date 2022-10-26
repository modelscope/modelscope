# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from transformers import AutoTokenizer

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_ranking)
class TextRankingPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in passage ranking model.
    """

    def __init__(self,
                 model_dir: str,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        super().__init__(model_dir, mode=mode, *args, **kwargs)
        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'source_sentence')
        self.second_sequence = kwargs.pop('second_sequence',
                                          'sentences_to_compare')
        self.sequence_length = kwargs.pop('sequence_length', 128)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    @type_assert(object, (str, tuple, Dict))
    def __call__(self, data: Union[tuple, Dict]) -> Dict[str, Any]:
        if isinstance(data, tuple):
            sentence1, sentence2 = data
        elif isinstance(data, dict):
            sentence1 = data.get(self.first_sequence)
            sentence2 = data.get(self.second_sequence)
        if isinstance(sentence2, str):
            sentence2 = [sentence2]
        if isinstance(sentence1, str):
            sentence1 = [sentence1]
        sentence1 = sentence1 * len(sentence2)

        max_seq_length = self.sequence_length
        feature = self.tokenizer(
            sentence1,
            sentence2,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt')
        if 'labels' in data:
            labels = data['labels']
            feature['labels'] = labels
        if 'qid' in data:
            qid = data['qid']
            feature['qid'] = qid
        return feature
