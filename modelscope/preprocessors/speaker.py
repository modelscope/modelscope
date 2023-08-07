# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple, Union

import torch

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.preprocessors.nlp.text_classification_preprocessor import \
    TextClassificationPreprocessorBase
from modelscope.preprocessors.nlp.token_classification_preprocessor import (
    NLPTokenizerForLSTM, TokenClassificationPreprocessorBase)
from modelscope.preprocessors.nlp.transformers_tokenizer import NLPTokenizer
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.sen_cls_tokenizer)
class SpeakerDiarizationDialogueDetectionPreprocessor(
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
                 keep_original_columns=None,
                 **kwargs):
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
                         label2id, mode, keep_original_columns)


@PREPROCESSORS.register_module(
    Fields.audio, module_name=Preprocessors.token_cls_tokenizer)
class SpeakerDiarizationSemanticSpeakerTurnDetectionPreprocessor(
        TokenClassificationPreprocessorBase):

    def __init__(self,
                 model_dir: str = None,
                 first_sequence: str = 'text',
                 label: str = 'label',
                 label2id: Dict = None,
                 label_all_tokens: bool = False,
                 mode: str = ModeKeys.INFERENCE,
                 max_length=None,
                 use_fast=None,
                 keep_original_columns=None,
                 return_text=True,
                 **kwargs):
        super().__init__(model_dir, first_sequence, label, label2id,
                         label_all_tokens, mode, keep_original_columns,
                         return_text)
        model_type = None
        if model_dir is not None:
            model_type = get_model_type(model_dir)

        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs[
            'max_length'] = max_length if max_length is not None else kwargs.get(
                'sequence_length', 128)
        kwargs.pop('sequence_length', None)
        kwargs['add_special_tokens'] = model_type != 'lstm'
        self.nlp_tokenizer = NLPTokenizerForLSTM(
            model_dir=model_dir,
            model_type=model_type,
            use_fast=use_fast,
            tokenize_kwargs=kwargs)

    def _tokenize_text(self, text: Union[str, List[str]], **kwargs):
        tokens = text

        if self.mode != ModeKeys.INFERENCE:
            assert isinstance(tokens, list), 'Input needs to be lists in training and evaluating,' \
                'because the length of the words and the labels need to be equal.'
        is_split_into_words = self.nlp_tokenizer.get_tokenizer_kwarg(
            'is_split_into_words', False)
        if is_split_into_words:
            # for supporting prompt seperator, should split twice. [SEP] for default.
            sep_idx = tokens.find('[SEP]')
            if sep_idx == -1 or self.is_lstm_model:
                tokens = list(tokens)
            else:
                tmp_tokens = []
                tmp_tokens.extend(list(tokens[:sep_idx]))
                tmp_tokens.append('[SEP]')
                tmp_tokens.extend(list(tokens[sep_idx + 5:]))
                tokens = tmp_tokens

        if is_split_into_words and self.mode == ModeKeys.INFERENCE:
            encodings, word_ids = self._tokenize_text_by_words(
                tokens, **kwargs)
        elif self.nlp_tokenizer.tokenizer.is_fast:
            encodings, word_ids = self._tokenize_text_with_fast_tokenizer(
                tokens, **kwargs)
        else:
            encodings, word_ids = self._tokenize_text_with_slow_tokenizer(
                tokens, **kwargs)

        sep_idx = -1
        for idx, token_id in enumerate(encodings['input_ids']):
            if token_id == self.nlp_tokenizer.tokenizer.sep_token_id:
                sep_idx = idx
                break
        if sep_idx != -1:
            for i in range(sep_idx, len(encodings['label_mask'])):
                encodings['label_mask'][i] = False

        if self.mode == ModeKeys.INFERENCE:
            for key in encodings.keys():
                encodings[key] = torch.tensor(encodings[key]).unsqueeze(0)
        else:
            encodings.pop('offset_mapping', None)
        return encodings, word_ids

    def _tokenize_text_by_words(self, tokens, **kwargs):
        input_ids = []
        label_mask = []
        offset_mapping = []
        attention_mask = []

        for offset, token in enumerate(tokens):
            subtoken_ids = self.nlp_tokenizer.tokenizer.encode(
                token, add_special_tokens=False)
            if len(subtoken_ids) == 0:
                subtoken_ids = [self.nlp_tokenizer.tokenizer.unk_token_id]
            input_ids.extend(subtoken_ids)
            attention_mask.extend([1] * len(subtoken_ids))
            label_mask.extend([True] + [False] * (len(subtoken_ids) - 1))
            offset_mapping.extend([(offset, offset + 1)])

        padding = kwargs.get('padding',
                             self.nlp_tokenizer.get_tokenizer_kwarg('padding'))
        max_length = kwargs.get(
            'max_length',
            kwargs.get('sequence_length',
                       self.nlp_tokenizer.get_tokenizer_kwarg('max_length')))
        special_token = 1 if self.nlp_tokenizer.get_tokenizer_kwarg(
            'add_special_tokens') else 0
        if len(label_mask) > max_length - 2 * special_token:
            label_mask = label_mask[:(max_length - 2 * special_token)]
            input_ids = input_ids[:(max_length - 2 * special_token)]
        offset_mapping = offset_mapping[:sum(label_mask)]
        if padding == 'max_length':
            label_mask = [False] * special_token + label_mask + \
                [False] * (max_length - len(label_mask) - special_token)
            offset_mapping = offset_mapping + [(0, 0)] * (
                max_length - len(offset_mapping))
            input_ids = [self.nlp_tokenizer.tokenizer.cls_token_id] * special_token + input_ids + \
                [self.nlp_tokenizer.tokenizer.sep_token_id] * special_token + \
                [self.nlp_tokenizer.tokenizer.pad_token_id] * (max_length - len(input_ids) - 2 * special_token)
            attention_mask = attention_mask + [1] * (
                special_token * 2) + [0] * (
                    max_length - len(attention_mask) - 2 * special_token)
        else:
            label_mask = [False] * special_token + label_mask + \
                [False] * special_token
            input_ids = [self.nlp_tokenizer.tokenizer.cls_token_id] * special_token + input_ids + \
                [self.nlp_tokenizer.tokenizer.sep_token_id] * special_token
            attention_mask = attention_mask + [1] * (special_token * 2)

        encodings = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_mask': label_mask,
            'offset_mapping': offset_mapping,
        }
        return encodings, None

    def _tokenize_text_with_fast_tokenizer(self, tokens, **kwargs):
        is_split_into_words = isinstance(tokens, list)
        encodings = self.nlp_tokenizer(
            tokens,
            return_offsets_mapping=True,
            is_split_into_words=is_split_into_words,
            **kwargs)
        label_mask = []
        word_ids = encodings.word_ids()
        offset_mapping = []
        for i in range(len(word_ids)):
            if word_ids[i] is None:
                label_mask.append(False)
            elif word_ids[i] == word_ids[i - 1]:
                label_mask.append(False)
                if not is_split_into_words:
                    offset_mapping[-1] = (offset_mapping[-1][0],
                                          encodings['offset_mapping'][i][1])
            else:
                label_mask.append(True)
                if is_split_into_words:
                    offset_mapping.append((word_ids[i], word_ids[i] + 1))
                else:
                    offset_mapping.append(encodings['offset_mapping'][i])

        padding = self.nlp_tokenizer.get_tokenizer_kwarg('padding')
        if padding == 'max_length':
            offset_mapping = offset_mapping + [(0, 0)] * (
                len(label_mask) - len(offset_mapping))
        encodings['offset_mapping'] = offset_mapping
        encodings['label_mask'] = label_mask
        return encodings, word_ids

    def _tokenize_text_with_slow_tokenizer(self, tokens, **kwargs):
        assert self.mode == ModeKeys.INFERENCE and isinstance(tokens, str), \
            'Slow tokenizer now only support str input in inference mode. If you are training models, ' \
            'please consider using the fast tokenizer.'
        word_ids = None
        encodings = self.nlp_tokenizer(
            tokens, is_split_into_words=False, **kwargs)
        tokenizer_name = self.nlp_tokenizer.get_tokenizer_class()
        method = 'get_label_mask_and_offset_mapping_' + tokenizer_name
        if not hasattr(self, method):
            raise RuntimeError(
                f'No `{method}` method defined for '
                f'tokenizer {tokenizer_name}, please use a fast tokenizer instead, or '
                f'try to implement a `{method}` method')
        label_mask, offset_mapping = getattr(self, method)(tokens)
        padding = kwargs.get('padding',
                             self.nlp_tokenizer.get_tokenizer_kwarg('padding'))
        max_length = kwargs.get(
            'max_length', self.nlp_tokenizer.get_tokenizer_kwarg('max_length'))
        special_token = 1 if kwargs.get(
            'add_special_tokens',
            self.nlp_tokenizer.get_tokenizer_kwarg(
                'add_special_tokens')) else 0
        if len(label_mask) > max_length - 2 * special_token:
            label_mask = label_mask[:(max_length - 2 * special_token)]
        offset_mapping = offset_mapping[:sum(label_mask)]

        if padding == 'max_length':
            label_mask = [False] * special_token + label_mask + \
                [False] * (max_length - len(label_mask) - special_token)
            offset_mapping = offset_mapping + [(0, 0)] * (
                max_length - len(offset_mapping))
        else:
            label_mask = [False] * special_token + label_mask + \
                [False] * special_token
        encodings['offset_mapping'] = offset_mapping
        encodings['label_mask'] = label_mask
        return encodings, word_ids

    def get_label_mask_and_offset_mapping_BertTokenizer(self, text):
        label_mask = []
        offset_mapping = []
        tokens = self.nlp_tokenizer.tokenizer.tokenize(text)
        offset = 0
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

        return label_mask, offset_mapping

    def get_label_mask_and_offset_mapping_XLMRobertaTokenizer(self, text):
        label_mask = []
        offset_mapping = []
        tokens = self.nlp_tokenizer.tokenizer.tokenize(text)
        offset = 0
        last_is_blank = False
        for token in tokens:
            is_start = (token[0] == '_')
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
        return label_mask, offset_mapping
