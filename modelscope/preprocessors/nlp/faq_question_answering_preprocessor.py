# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.faq_question_answering_preprocessor)
class FaqQuestionAnsweringTransformersPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 tokenizer='BertTokenizer',
                 query_set='query_set',
                 support_set='support_set',
                 query_label='query_label',
                 label_in_support_set='label',
                 text_in_support_set='text',
                 sequence_length=None,
                 **kwargs):
        """The preprocessor for Faq QA task, based on transformers' tokenizer.

        Args:
            model_dir: The model dir containing the essential files to build the tokenizer.
            mode: The mode for this preprocessor.
            tokenizer: The tokenizer type used, supported types are `BertTokenizer`
                and `XLMRobertaTokenizer`, default `BertTokenizer`.
            query_set: The key for the query_set.
            support_set: The key for the support_set.
            label_in_support_set: The key for the label_in_support_set.
            text_in_support_set: The key for the text_in_support_set.
            sequence_length: The sequence length for the preprocessor.
        """
        super().__init__(mode)
        if tokenizer == 'XLMRoberta':
            from transformers import XLMRobertaTokenizer
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
        else:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        if sequence_length is not None:
            self.max_len = sequence_length
        else:
            self.max_len = kwargs.get('max_seq_length', 50)
        self.label_dict = None
        self.query_label = query_label
        self.query_set = query_set
        self.support_set = support_set
        self.label_in_support_set = label_in_support_set
        self.text_in_support_set = text_in_support_set

    def pad(self, samples, max_len):
        result = []
        for sample in samples:
            pad_len = max_len - len(sample[:max_len])
            result.append(sample[:max_len]
                          + [self.tokenizer.pad_token_id] * pad_len)
        return result

    def set_label_dict(self, label_dict):
        self.label_dict = label_dict

    def get_label(self, label_id):
        assert self.label_dict is not None and label_id < len(self.label_dict)
        return self.label_dict[label_id]

    def encode_plus(self, text):
        return [
            self.tokenizer.cls_token_id
        ] + self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(text)) + [self.tokenizer.sep_token_id]

    @type_assert(object, Dict)
    def __call__(self, data: Dict[str, Any],
                 **preprocessor_param) -> Dict[str, Any]:
        invoke_mode = preprocessor_param.get('mode', None)
        if self.mode in (ModeKeys.TRAIN,
                         ModeKeys.EVAL) and invoke_mode != ModeKeys.INFERENCE:
            return data
        tmp_max_len = preprocessor_param.get(
            'sequence_length',
            preprocessor_param.get('max_seq_length', self.max_len))
        queryset = data[self.query_set]
        if not isinstance(queryset, list):
            queryset = [queryset]
        supportset = data[self.support_set]
        supportset = sorted(
            supportset, key=lambda d: d[self.label_in_support_set])

        queryset_tokenized = [self.encode_plus(text) for text in queryset]
        supportset_tokenized = [
            self.encode_plus(item[self.text_in_support_set])
            for item in supportset
        ]

        max_len = max(
            [len(seq) for seq in queryset_tokenized + supportset_tokenized])
        max_len = min(tmp_max_len, max_len)
        queryset_padded = self.pad(queryset_tokenized, max_len)
        supportset_padded = self.pad(supportset_tokenized, max_len)

        supportset_labels_ori = [
            item[self.label_in_support_set] for item in supportset
        ]
        label_dict = []
        for label in supportset_labels_ori:
            if label not in label_dict:
                label_dict.append(label)
        self.set_label_dict(label_dict)
        supportset_labels_ids = [
            label_dict.index(label) for label in supportset_labels_ori
        ]

        query_atttention_mask = torch.ne(
            torch.tensor(queryset_padded, dtype=torch.int32),
            self.tokenizer.pad_token_id)
        support_atttention_mask = torch.ne(
            torch.tensor(supportset_padded, dtype=torch.int32),
            self.tokenizer.pad_token_id)

        result = {
            'query': torch.LongTensor(queryset_padded),
            'support': torch.LongTensor(supportset_padded),
            'query_attention_mask': query_atttention_mask,
            'support_attention_mask': support_atttention_mask,
            'support_labels': torch.LongTensor(supportset_labels_ids)
        }
        if self.query_label in data:
            query_label = data[self.query_label]
            query_label_ids = [
                label_dict.index(label) for label in query_label
            ]
            result['labels'] = torch.LongTensor(query_label_ids)
        return result

    def batch_encode(self, sentence_list: list, max_length=None):
        if not max_length:
            max_length = self.max_len
        return self.tokenizer.batch_encode_plus(
            sentence_list, padding=True, max_length=max_length)
