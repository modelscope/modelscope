# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from transformers import AutoTokenizer

from modelscope.metainfo import Models, Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Fields, InputFields, ModeKeys
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS

__all__ = [
    'Tokenize', 'SequenceClassificationPreprocessor',
    'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
    'PairSentenceClassificationPreprocessor',
    'SingleSentenceClassificationPreprocessor', 'FillMaskPreprocessor',
    'ZeroShotClassificationPreprocessor', 'NERPreprocessor',
    'TextErrorCorrectionPreprocessor'
]


@PREPROCESSORS.register_module(Fields.nlp)
class Tokenize(Preprocessor):

    def __init__(self, tokenizer_name) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(data, str):
            data = {InputFields.text: data}
        token_dict = self._tokenizer(data[InputFields.text])
        data.update(token_dict)
        return data


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.bert_seq_cls_tokenizer)
class SequenceClassificationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        from easynlp.modelzoo import AutoTokenizer
        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.sequence_length = kwargs.pop('sequence_length', 128)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        print(f'this is the tokenzier {self.tokenizer}')
        self.label2id = parse_label_mapping(self.model_dir)

    @type_assert(object, (str, tuple, Dict))
    def __call__(self, data: Union[str, tuple, Dict]) -> Dict[str, Any]:
        feature = super().__call__(data)
        if isinstance(data, str):
            new_data = {self.first_sequence: data}
        elif isinstance(data, tuple):
            sentence1, sentence2 = data
            new_data = {
                self.first_sequence: sentence1,
                self.second_sequence: sentence2
            }
        else:
            new_data = data

        # preprocess the data for the model input

        rst = {
            'id': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
        }

        max_seq_length = self.sequence_length

        text_a = new_data[self.first_sequence]
        text_b = new_data.get(self.second_sequence, None)
        feature = self.tokenizer(
            text_a,
            text_b,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length)

        rst['id'].append(new_data.get('id', str(uuid.uuid4())))
        rst['input_ids'].append(feature['input_ids'])
        rst['attention_mask'].append(feature['attention_mask'])
        rst['token_type_ids'].append(feature['token_type_ids'])

        return rst


class NLPTokenizerPreprocessorBase(Preprocessor):

    def __init__(self, model_dir: str, pair: bool, mode: str, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(**kwargs)
        self.model_dir: str = model_dir
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
        self.second_sequence = kwargs.pop('second_sequence', 'second_sequence')
        self.pair = pair
        self._mode = mode
        self.label = kwargs.pop('label', OutputKeys.LABEL)
        self.label2id = None
        if 'label2id' in kwargs:
            self.label2id = kwargs.pop('label2id')
        if self.label2id is None:
            self.label2id = parse_label_mapping(self.model_dir)

        self.tokenize_kwargs = kwargs
        self.tokenizer = self.build_tokenizer(model_dir)

    @property
    def id2label(self):
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def build_tokenizer(self, model_dir):
        model_type = get_model_type(model_dir)
        if model_type in (Models.structbert, Models.gpt3, Models.palm):
            from modelscope.models.nlp.structbert import SbertTokenizerFast
            return SbertTokenizerFast.from_pretrained(model_dir)
        elif model_type == Models.veco:
            from modelscope.models.nlp.veco import VecoTokenizerFast
            return VecoTokenizerFast.from_pretrained(model_dir)
        else:
            return AutoTokenizer.from_pretrained(model_dir)

    def __call__(self, data: Union[str, Tuple, Dict]) -> Dict[str, Any]:
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

        text_a, text_b, labels = self.parse_text_and_label(data)
        output = self.tokenizer(
            text_a,
            text_b,
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            **self.tokenize_kwargs)
        self.labels_to_id(labels, output)
        return output

    def parse_text_and_label(self, data):
        text_a, text_b, labels = None, None, None
        if isinstance(data, str):
            text_a = data
        elif isinstance(data, tuple) or isinstance(data, list):
            if len(data) == 3:
                text_a, text_b, labels = data
            elif len(data) == 2:
                if self.pair:
                    text_a, text_b = data
                else:
                    text_a, labels = data
        elif isinstance(data, dict):
            text_a = data.get(self.first_sequence)
            text_b = data.get(self.second_sequence)
            labels = data.get(self.label)

        return text_a, text_b, labels

    def labels_to_id(self, labels, output):

        def label_can_be_mapped(label):
            return isinstance(label, str) or isinstance(label, int)

        if labels is not None:
            if isinstance(labels, Iterable) and all([label_can_be_mapped(label) for label in labels]) \
                    and self.label2id is not None:
                output[OutputKeys.LABEL] = [
                    self.label2id[str(label)] for label in labels
                ]
            elif label_can_be_mapped(labels) and self.label2id is not None:
                output[OutputKeys.LABEL] = self.label2id[str(labels)]
            else:
                output[OutputKeys.LABEL] = labels


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.nli_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_sim_tokenizer)
class PairSentenceClassificationPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=True, mode=mode, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_cls_tokenizer)
class SingleSentenceClassificationPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.zero_shot_cls_tokenizer)
class ZeroShotClassificationPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        self.sequence_length = kwargs.pop('sequence_length', 512)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)

    def __call__(self, data: Union[str, Dict], hypothesis_template: str,
                 candidate_labels: list) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str or dict): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, dict):
            data = data.get(self.first_sequence)

        pairs = [[data, hypothesis_template.format(label)]
                 for label in candidate_labels]

        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.sequence_length,
            truncation_strategy='only_first',
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None)
        return features


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_gen_tokenizer)
class TextGenerationPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self,
                 model_dir: str,
                 tokenizer=None,
                 mode=ModeKeys.INFERENCE,
                 **kwargs):
        self.tokenizer = self.build_tokenizer(
            model_dir) if tokenizer is None else tokenizer
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     False)
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)

    @staticmethod
    def get_roberta_tokenizer_dir(model_dir: str) -> Optional[str]:
        import os
        for name in os.listdir(model_dir):
            full_name = os.path.join(model_dir, name)
            if 'roberta' in name and os.path.isdir(full_name):
                return full_name

    def build_tokenizer(self, model_dir: str):
        roberta_tokenizer_dir = self.get_roberta_tokenizer_dir(model_dir)
        if roberta_tokenizer_dir:
            from transformers import RobertaTokenizer
            return RobertaTokenizer.from_pretrained(
                roberta_tokenizer_dir, do_lower_case=False)
        return super().build_tokenizer(model_dir)

    def __call__(self, data: Union[Dict, str]) -> Dict[str, Any]:
        if self._mode == ModeKeys.INFERENCE:
            return super().__call__(data)
        src_txt = data['src_txt']
        tgt_txt = data['tgt_txt']
        src_rst = super().__call__(src_txt)
        tgt_rst = super().__call__(tgt_txt)

        return {
            'src': src_rst['input_ids'],
            'tgt': tgt_rst['input_ids'],
            'mask_src': src_rst['attention_mask']
        }


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.fill_mask)
class FillMaskPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp,
    module_name=Preprocessors.word_segment_text_to_label_preprocessor)
class WordSegmentationBlankSetToLabelPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.first_sequence: str = kwargs.pop('first_sequence',
                                              'first_sequence')
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
    Fields.nlp, module_name=Preprocessors.token_cls_tokenizer)
class TokenClassificationPreprocessor(NLPTokenizerPreprocessorBase):

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        self.label_all_tokens = kwargs.pop('label_all_tokens', False)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)

    def __call__(self, data: Union[str, Dict]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """

        text_a = None
        labels_list = None
        if isinstance(data, str):
            text_a = data
        elif isinstance(data, dict):
            text_a = data.get(self.first_sequence)
            labels_list = data.get(self.label)

        if isinstance(text_a, str):
            text_a = text_a.replace(' ', '').strip()

        tokenized_inputs = self.tokenizer(
            [t for t in text_a],
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None,
            is_split_into_words=True,
            **self.tokenize_kwargs)

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
            word_ids = tokenized_inputs.word_ids()
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
            tokenized_inputs['labels'] = labels
            # new code end

        if self._mode == ModeKeys.INFERENCE:
            tokenized_inputs[OutputKeys.TEXT] = text_a
        return tokenized_inputs


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.ner_tokenizer)
class NERPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.sequence_length = kwargs.pop('sequence_length', 512)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True)
        self.is_split_into_words = self.tokenizer.init_kwargs.get(
            'is_split_into_words', False)

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
        if self.is_split_into_words:
            input_ids = []
            label_mask = []
            offset_mapping = []
            for offset, token in enumerate(list(data)):
                subtoken_ids = self.tokenizer.encode(
                    token, add_special_tokens=False)
                if len(subtoken_ids) == 0:
                    subtoken_ids = [self.tokenizer.unk_token_id]
                input_ids.extend(subtoken_ids)
                label_mask.extend([1] + [0] * (len(subtoken_ids) - 1))
                offset_mapping.extend([(offset, offset + 1)]
                                      + [(offset + 1, offset + 1)]
                                      * (len(subtoken_ids) - 1))
            if len(input_ids) >= self.sequence_length - 2:
                input_ids = input_ids[:self.sequence_length - 2]
                label_mask = label_mask[:self.sequence_length - 2]
                offset_mapping = offset_mapping[:self.sequence_length - 2]
            input_ids = [self.tokenizer.cls_token_id
                         ] + input_ids + [self.tokenizer.sep_token_id]
            label_mask = [0] + label_mask + [0]
            attention_mask = [1] * len(input_ids)
        else:
            encodings = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.sequence_length,
                return_offsets_mapping=True)
            input_ids = encodings['input_ids']
            attention_mask = encodings['attention_mask']
            word_ids = encodings.word_ids()
            label_mask = []
            offset_mapping = []
            for i in range(len(word_ids)):
                if word_ids[i] is None:
                    label_mask.append(0)
                elif word_ids[i] == word_ids[i - 1]:
                    label_mask.append(0)
                    offset_mapping[-1] = (offset_mapping[-1][0],
                                          encodings['offset_mapping'][i][1])
                else:
                    label_mask.append(1)
                    offset_mapping.append(encodings['offset_mapping'][i])
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_mask': label_mask,
            'offset_mapping': offset_mapping
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.text_error_correction)
class TextErrorCorrectionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        from fairseq.data import Dictionary
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        self.vocab = Dictionary.load(osp.join(model_dir, 'dict.src.txt'))

    def __call__(self, data: str) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    '随着中国经济突飞猛近，建造工业与日俱增'
        Returns:
            Dict[str, Any]: the preprocessed data
            Example:
            {'net_input':
                {'src_tokens':tensor([1,2,3,4]),
                'src_lengths': tensor([4])}
            }
        """

        text = ' '.join([x for x in data])
        inputs = self.vocab.encode_line(
            text, append_eos=True, add_if_not_exist=False)
        lengths = inputs.size()
        sample = dict()
        sample['net_input'] = {'src_tokens': inputs, 'src_lengths': lengths}
        return sample
