# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert
from .nlp_base import NLPBasePreprocessor, NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp,
    module_name=Preprocessors.word_segment_text_to_label_preprocessor)
class WordSegmentationBlankSetToLabelPreprocessor(NLPBasePreprocessor):
    """The preprocessor used to turn a single sentence to a labeled token-classification dict.
    """

    def __init__(self, **kwargs):
        self.first_sequence: str = kwargs.pop('first_sequence', 'tokens')
        self.label = kwargs.pop('label', OutputKeys.LABELS)

    def __call__(self, data: str) -> Union[Dict[str, Any], Tuple]:
        data = data.split(' ')
        data = list(filter(lambda x: len(x) > 0, data))

        def produce_train_sample(words):
            chars = []
            labels = []
            for word in words:
                chars.extend(list(word))
                if len(word) == 1:
                    labels.append('S-CWS')
                else:
                    labels.extend(['B-CWS'] + ['I-CWS'] * (len(word) - 2)
                                  + ['E-CWS'])
            assert len(chars) == len(labels)
            return chars, labels

        chars, labels = produce_train_sample(data)
        return {
            self.first_sequence: chars,
            self.label: labels,
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.ner_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.token_cls_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sequence_labeling_tokenizer)
class TokenClassificationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in normal NER task.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        self.sequence_length = kwargs['max_length']
        self.label_all_tokens = kwargs.pop('label_all_tokens', False)
        super().__init__(model_dir, mode=mode, **kwargs)

        if 'is_split_into_words' in kwargs:
            self.is_split_into_words = kwargs.pop('is_split_into_words')
        else:
            self.is_split_into_words = self.tokenizer.init_kwargs.get(
                'is_split_into_words', False)
        if 'label2id' in kwargs:
            kwargs.pop('label2id')

    @type_assert(object, (str, dict))
    def __call__(self, data: Union[dict, str]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        # preprocess the data for the model input
        text = None
        labels_list = None
        if isinstance(data, str):
            # for inference inputs without label
            text = data
            self.tokenize_kwargs['add_special_tokens'] = False
        elif isinstance(data, dict):
            # for finetune inputs with label
            text = data.get(self.first_sequence)
            labels_list = data.get(self.label)
            if isinstance(text, list):
                self.tokenize_kwargs['is_split_into_words'] = True

        input_ids = []
        label_mask = []
        offset_mapping = []
        token_type_ids = []
        if self.is_split_into_words and self._mode == ModeKeys.INFERENCE:
            for offset, token in enumerate(list(text)):
                subtoken_ids = self.tokenizer.encode(token,
                                                     **self.tokenize_kwargs)
                if len(subtoken_ids) == 0:
                    subtoken_ids = [self.tokenizer.unk_token_id]
                input_ids.extend(subtoken_ids)
                label_mask.extend([1] + [0] * (len(subtoken_ids) - 1))
                offset_mapping.extend([(offset, offset + 1)])
        else:
            if self.tokenizer.is_fast:
                encodings = self.tokenizer(
                    text, return_offsets_mapping=True, **self.tokenize_kwargs)
                attention_mask = encodings['attention_mask']
                token_type_ids = encodings['token_type_ids']
                input_ids = encodings['input_ids']
                word_ids = encodings.word_ids()
                for i in range(len(word_ids)):
                    if word_ids[i] is None:
                        label_mask.append(0)
                    elif word_ids[i] == word_ids[i - 1]:
                        label_mask.append(0)
                        offset_mapping[-1] = (
                            offset_mapping[-1][0],
                            encodings['offset_mapping'][i][1])
                    else:
                        label_mask.append(1)
                        offset_mapping.append(encodings['offset_mapping'][i])
            else:
                encodings = self.tokenizer(
                    text, add_special_tokens=False, **self.tokenize_kwargs)
                input_ids = encodings['input_ids']
                label_mask, offset_mapping = self.get_label_mask_and_offset_mapping(
                    text)

        if self._mode == ModeKeys.INFERENCE:
            if len(input_ids) >= self.sequence_length - 2:
                input_ids = input_ids[:self.sequence_length - 2]
                label_mask = label_mask[:self.sequence_length - 2]
            input_ids = [self.tokenizer.cls_token_id
                         ] + input_ids + [self.tokenizer.sep_token_id]
            label_mask = [0] + label_mask + [0]
            attention_mask = [1] * len(input_ids)
            offset_mapping = offset_mapping[:sum(label_mask)]

            if not self.is_transformer_based_model:
                input_ids = input_ids[1:-1]
                attention_mask = attention_mask[1:-1]
                label_mask = label_mask[1:-1]

            input_ids = torch.tensor(input_ids).unsqueeze(0)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0)
            label_mask = torch.tensor(
                label_mask, dtype=torch.bool).unsqueeze(0)

            # the token classification
            output = {
                'text': text,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'label_mask': label_mask,
                'offset_mapping': offset_mapping
            }
        else:
            output = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'label_mask': label_mask,
            }

            # align the labels with tokenized text
            if labels_list is not None:
                assert self.label2id is not None
                # Map that sends B-Xxx label to its I-Xxx counterpart
                b_to_i_label = []
                label_enumerate_values = [
                    k for k, v in sorted(
                        self.label2id.items(), key=lambda item: item[1])
                ]
                for idx, label in enumerate(label_enumerate_values):
                    if label.startswith('B-') and label.replace(
                            'B-', 'I-') in label_enumerate_values:
                        b_to_i_label.append(
                            label_enumerate_values.index(
                                label.replace('B-', 'I-')))
                    else:
                        b_to_i_label.append(idx)

                label_row = [self.label2id[lb] for lb in labels_list]
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label_row[word_idx])
                    else:
                        if self.label_all_tokens:
                            label_ids.append(b_to_i_label[label_row[word_idx]])
                        else:
                            label_ids.append(-100)
                    previous_word_idx = word_idx
                labels = label_ids
                output['labels'] = labels
            output = {
                k: np.array(v) if isinstance(v, list) else v
                for k, v in output.items()
            }
        return output

    def get_tokenizer_class(self):
        tokenizer_class = self.tokenizer.__class__.__name__
        if tokenizer_class.endswith(
                'Fast') and tokenizer_class != 'PreTrainedTokenizerFast':
            tokenizer_class = tokenizer_class[:-4]
        return tokenizer_class

    def get_label_mask_and_offset_mapping(self, text):
        label_mask = []
        offset_mapping = []
        tokens = self.tokenizer.tokenize(text)
        offset = 0
        if self.get_tokenizer_class() == 'BertTokenizer':
            for token in tokens:
                is_start = (token[:2] != '##')
                if is_start:
                    label_mask.append(True)
                else:
                    token = token[2:]
                    label_mask.append(False)
                start = offset + text[offset:].index(token)
                end = start + len(token)
                if is_start:
                    offset_mapping.append((start, end))
                else:
                    offset_mapping[-1] = (offset_mapping[-1][0], end)
                offset = end
        elif self.get_tokenizer_class() == 'XLMRobertaTokenizer':
            last_is_blank = False
            for token in tokens:
                is_start = (token[0] == '▁')
                if is_start:
                    token = token[1:]
                    label_mask.append(True)
                    if len(token) == 0:
                        last_is_blank = True
                        continue
                else:
                    label_mask.append(False)
                start = offset + text[offset:].index(token)
                end = start + len(token)
                if last_is_blank or is_start:
                    offset_mapping.append((start, end))
                else:
                    offset_mapping[-1] = (offset_mapping[-1][0], end)
                offset = end
                last_is_blank = False
        else:
            raise NotImplementedError

        return label_mask, offset_mapping
