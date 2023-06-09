# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from modelscope.metainfo import Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger
from modelscope.utils.type_assert import type_assert
from .transformers_tokenizer import NLPTokenizer
from .utils import parse_text_and_label

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.nlp,
    module_name=Preprocessors.word_segment_text_to_label_preprocessor)
class WordSegmentationBlankSetToLabelPreprocessor(Preprocessor):
    """The preprocessor used to turn a single sentence to a labeled token-classification dict.
    """

    def __init__(self, generated_sentence='tokens', generated_label='labels'):
        super().__init__()
        self.generated_sentence = generated_sentence
        self.generated_label = generated_label

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
            self.generated_sentence: chars,
            self.generated_label: labels,
        }


class TokenClassificationPreprocessorBase(Preprocessor):

    def __init__(self,
                 model_dir: str = None,
                 first_sequence: str = None,
                 label: str = 'label',
                 label2id: Dict = None,
                 label_all_tokens: bool = False,
                 mode: str = ModeKeys.INFERENCE,
                 keep_original_columns: List[str] = None,
                 return_text: bool = True):
        """The base class for all the token-classification tasks.

        Args:
            model_dir: The model dir to build the the label2id mapping.
                If None, user need to pass in the `label2id` param.
            first_sequence: The key for the text(token) column if input type is a dict.
            label: The key for the label column if input type is a dict and the mode is `training` or `evaluation`.
            label2id: The label2id mapping, if not provided, you need to specify the model_dir to search the mapping
                from config files.
            label_all_tokens: If label exists in the dataset, the preprocessor will try to label the tokens.
                If label_all_tokens is true, all non-initial sub-tokens will get labels like `I-xxx`,
                or else the labels will be filled with -100, default False.
            mode: The preprocessor mode.
            keep_original_columns(List[str], `optional`): The original columns to keep,
                only available when the input is a `dict`, default None
            return_text: Whether to return `text` field in inference mode, default: True.
        """
        super().__init__(mode)
        self.model_dir = model_dir
        self.first_sequence = first_sequence
        self.label = label
        self.label2id = label2id
        self.label_all_tokens = label_all_tokens
        self.keep_original_columns = keep_original_columns
        self.return_text = return_text
        if self.label2id is None and self.model_dir is not None:
            self.label2id = parse_label_mapping(self.model_dir)

    @property
    def id2label(self):
        """Return the id2label mapping according to the label2id mapping.

        @return: The id2label mapping if exists.
        """
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def labels_to_id(self, labels_list, word_ids):
        # align the labels with tokenized text
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
                    label_enumerate_values.index(label.replace('B-', 'I-')))
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
        return label_ids

    def _tokenize_text(self, sequence1, **kwargs):
        """Tokenize the text.

        Args:
            sequence1: The first sequence.
            sequence2: The second sequence which may be None.

        Returns:
            The encoded sequence.
        """
        raise NotImplementedError()

    @type_assert(object, (str, tuple, dict))
    def __call__(self, data: Union[dict, tuple, str],
                 **kwargs) -> Dict[str, Any]:
        text, _, label = parse_text_and_label(
            data, self.mode, self.first_sequence, label=self.label)
        outputs, word_ids = self._tokenize_text(text, **kwargs)
        if label is not None:
            label_ids = self.labels_to_id(label, word_ids)
            outputs[OutputKeys.LABELS] = label_ids
        outputs = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in outputs.items()
        }
        if self.keep_original_columns and isinstance(data, dict):
            for column in self.keep_original_columns:
                outputs[column] = data[column]
        if self.mode == ModeKeys.INFERENCE and self.return_text:
            outputs['text'] = text
        return outputs


class NLPTokenizerForLSTM(NLPTokenizer):

    def build_tokenizer(self):
        if self.model_type == 'lstm':
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(
                self.model_dir, use_fast=self.use_fast, tokenizer_type='bert')
        else:
            return super().build_tokenizer()

    def get_tokenizer_class(self):
        tokenizer_class = self.tokenizer.__class__.__name__
        if tokenizer_class.endswith(
                'Fast') and tokenizer_class != 'PreTrainedTokenizerFast':
            tokenizer_class = tokenizer_class[:-4]
        return tokenizer_class


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.ner_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.token_cls_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sequence_labeling_tokenizer)
class TokenClassificationTransformersPreprocessor(
        TokenClassificationPreprocessorBase):
    """The tokenizer preprocessor used in normal NER task.
    """

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
        """

        Args:
            use_fast: Whether to use the fast tokenizer or not.
            max_length: The max sequence length which the model supported,
                will be passed into tokenizer as the 'max_length' param.
            **kwargs: Extra args input into the tokenizer's __call__ method.
        """
        super().__init__(model_dir, first_sequence, label, label2id,
                         label_all_tokens, mode, keep_original_columns,
                         return_text)
        self.is_lstm_model = 'lstm' in model_dir
        model_type = None
        if self.is_lstm_model:
            model_type = 'lstm'
        elif model_dir is not None:
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

        # modify label mask, mask all prompt tokens (tokens after sep token)
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
        return label_mask, offset_mapping
