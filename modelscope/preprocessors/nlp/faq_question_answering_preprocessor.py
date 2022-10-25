# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import Fields, ModeKeys, ModelFile
from modelscope.utils.type_assert import type_assert
from .nlp_base import NLPBasePreprocessor


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.faq_question_answering_preprocessor)
class FaqQuestionAnsweringPreprocessor(NLPBasePreprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        super(FaqQuestionAnsweringPreprocessor, self).__init__(
            model_dir, mode=ModeKeys.INFERENCE, **kwargs)
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        preprocessor_config = Config.from_file(
            os.path.join(model_dir, ModelFile.CONFIGURATION)).get(
                ConfigFields.preprocessor, {})
        self.MAX_LEN = preprocessor_config.get('max_seq_length', 50)
        self.label_dict = None

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
        TMP_MAX_LEN = preprocessor_param.get('max_seq_length', self.MAX_LEN)
        queryset = data['query_set']
        if not isinstance(queryset, list):
            queryset = [queryset]
        supportset = data['support_set']
        supportset = sorted(supportset, key=lambda d: d['label'])

        queryset_tokenized = [self.encode_plus(text) for text in queryset]
        supportset_tokenized = [
            self.encode_plus(item['text']) for item in supportset
        ]

        max_len = max(
            [len(seq) for seq in queryset_tokenized + supportset_tokenized])
        max_len = min(TMP_MAX_LEN, max_len)
        queryset_padded = self.pad(queryset_tokenized, max_len)
        supportset_padded = self.pad(supportset_tokenized, max_len)

        supportset_labels_ori = [item['label'] for item in supportset]
        label_dict = []
        for label in supportset_labels_ori:
            if label not in label_dict:
                label_dict.append(label)
        self.set_label_dict(label_dict)
        supportset_labels_ids = [
            label_dict.index(label) for label in supportset_labels_ori
        ]
        return {
            'query': queryset_padded,
            'support': supportset_padded,
            'support_labels': supportset_labels_ids
        }

    def batch_encode(self, sentence_list: list, max_length=None):
        if not max_length:
            max_length = self.MAX_LEN
        return self.tokenizer.batch_encode_plus(
            sentence_list, padding=True, max_length=max_length)
