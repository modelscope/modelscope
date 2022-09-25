# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import re
import uuid
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from transformers import AutoTokenizer, BertTokenizerFast

from modelscope.metainfo import Models, Preprocessors
from modelscope.models.nlp.structbert import SbertTokenizerFast
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import Fields, InputFields, ModeKeys, ModelFile
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp import import_external_nltk_data
from modelscope.utils.type_assert import type_assert

logger = get_logger()

__all__ = [
    'Tokenize', 'SequenceClassificationPreprocessor',
    'TextGenerationPreprocessor', 'TokenClassificationPreprocessor',
    'PairSentenceClassificationPreprocessor',
    'SingleSentenceClassificationPreprocessor', 'FillMaskPreprocessor',
    'ZeroShotClassificationPreprocessor', 'NERPreprocessor',
    'SentenceEmbeddingPreprocessor', 'PassageRankingPreprocessor',
    'FaqQuestionAnsweringPreprocessor', 'SequenceLabelingPreprocessor',
    'RelationExtractionPreprocessor', 'DocumentSegmentationPreprocessor',
    'FillMaskPoNetPreprocessor'
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
        """preprocess the data

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
        """The NLP tokenizer preprocessor base class.

        Any nlp preprocessor which uses the hf tokenizer can inherit from this class.

        Args:
            model_dir (str): The local model path
            first_sequence: The key for the first sequence
            second_sequence: The key for the second sequence
            label: The label key
            label2id: An optional label2id mapping, the class will try to call utils.parse_label_mapping
                if this mapping is not supplied.
            pair (bool): Pair sentence input or single sentence input.
            mode: Run this preprocessor in either 'train'/'eval'/'inference' mode
            kwargs: These kwargs will be directly fed into the tokenizer.
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
        """Return the id2label mapping according to the label2id mapping.

        @return: The id2label mapping if exists.
        """
        if self.label2id is not None:
            return {id: label for label, id in self.label2id.items()}
        return None

    def build_tokenizer(self, model_dir):
        """Build a tokenizer by the model type.

        NOTE: This default implementation only returns slow tokenizer, because the fast tokenizers have a
        multi-thread problem.

        @param model_dir:  The local model dir.
        @return: The initialized tokenizer.
        """

        model_type = get_model_type(model_dir)
        if model_type in (Models.structbert, Models.gpt3, Models.palm,
                          Models.plug):
            from modelscope.models.nlp.structbert import SbertTokenizer
            return SbertTokenizer.from_pretrained(model_dir, use_fast=False)
        elif model_type == Models.veco:
            from modelscope.models.nlp.veco import VecoTokenizer
            return VecoTokenizer.from_pretrained(model_dir)
        elif model_type == Models.deberta_v2:
            from modelscope.models.nlp.deberta_v2 import DebertaV2Tokenizer
            return DebertaV2Tokenizer.from_pretrained(model_dir)
        else:
            return AutoTokenizer.from_pretrained(model_dir, use_fast=False)

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
        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }
        self.labels_to_id(labels, output)
        return output

    def parse_text_and_label(self, data):
        """Parse the input and return the sentences and labels.

        When input type is tuple or list and its size is 2:
        If the pair param is False, data will be parsed as the first_sentence and the label,
        else it will be parsed as the first_sentence and the second_sentence.

        @param data: The input data.
        @return: The sentences and labels tuple.
        """
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
        """Turn the labels to id with the type int or float.

        If the original label's type is str or int, the label2id mapping will try to convert it to the final label.
        If the original label's type is float, or the label2id mapping does not exist,
        the original label will be returned.

        @param labels: The input labels.
        @param output: The label id.
        @return: The final labels.
        """

        def label_can_be_mapped(label):
            return isinstance(label, str) or isinstance(label, int)

        if labels is not None:
            if isinstance(labels, Iterable) and all([label_can_be_mapped(label) for label in labels]) \
                    and self.label2id is not None:
                output[OutputKeys.LABELS] = [
                    self.label2id[str(label)] for label in labels
                ]
            elif label_can_be_mapped(labels) and self.label2id is not None:
                output[OutputKeys.LABELS] = self.label2id[str(labels)]
            else:
                output[OutputKeys.LABELS] = labels


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.passage_ranking)
class PassageRankingPreprocessor(NLPTokenizerPreprocessorBase):
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
        super().__init__(model_dir, pair=True, mode=mode, *args, **kwargs)
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


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.nli_tokenizer)
@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_sim_tokenizer)
class PairSentenceClassificationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in pair sentence classification.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=True, mode=mode, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sen_cls_tokenizer)
class SingleSentenceClassificationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in single sentence classification.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sentence_embedding)
class SentenceEmbeddingPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in sentence embedding.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get(
            'padding', False if mode == ModeKeys.INFERENCE else 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 128)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)

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


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.zero_shot_cls_tokenizer)
class ZeroShotClassificationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in zero shot classification.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        """preprocess the data

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
    """The tokenizer preprocessor used in text generation.
    """

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
        src_rst = super().__call__(data['src_txt'])
        src_input_ids = src_rst['input_ids']
        src_attention_mask = src_rst['attention_mask']
        if 'tgt_txt' in data:
            labels = super().__call__(data['tgt_txt'])['input_ids']
        else:
            labels = src_input_ids[1:]
            src_input_ids = src_input_ids[:-1]
            src_attention_mask = src_attention_mask[:-1]

        return {
            'input_ids': src_input_ids,
            'attention_mask': src_attention_mask,
            'labels': labels,
        }


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.fill_mask)
class FillMaskPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in MLM task.
    """

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
    """The preprocessor used to turn a single sentence to a labeled token-classification dict.
    """

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
    """The tokenizer preprocessor used in normal token classification task.
    """

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
    """The tokenizer preprocessor used in normal NER task.

    NOTE: This preprocessor may be merged with the TokenClassificationPreprocessor in the next edition.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.sequence_length = kwargs.pop('sequence_length', 512)
        self.is_transformer_based_model = 'lstm' not in model_dir
        if self.is_transformer_based_model:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, use_fast=True)
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(
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

        if not self.is_transformer_based_model:
            input_ids = input_ids[1:-1]
            attention_mask = attention_mask[1:-1]
            label_mask = label_mask[1:-1]
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_mask': label_mask,
            'offset_mapping': offset_mapping
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.sequence_labeling_tokenizer)
class SequenceLabelingPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in normal NER task.

    NOTE: This preprocessor may be merged with the TokenClassificationPreprocessor in the next edition.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.sequence_length = kwargs.pop('sequence_length', 512)

        if 'lstm' in model_dir or 'gcnn' in model_dir:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                model_dir, use_fast=False)
        elif 'structbert' in model_dir:
            self.tokenizer = SbertTokenizerFast.from_pretrained(
                model_dir, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir, use_fast=False)
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

        if not self.is_transformer_based_model:
            input_ids = input_ids[1:-1]
            attention_mask = attention_mask[1:-1]
            label_mask = label_mask[1:-1]
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label_mask': label_mask,
            'offset_mapping': offset_mapping
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.re_tokenizer)
class RelationExtractionPreprocessor(Preprocessor):
    """The tokenizer preprocessor used in normal RE task.

    NOTE: This preprocessor may be merged with the TokenClassificationPreprocessor in the next edition.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.sequence_length = kwargs.pop('sequence_length', 512)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=True)

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
        output = self.tokenizer([text], return_tensors='pt')
        return {
            'text': text,
            'input_ids': output['input_ids'],
            'attention_mask': output['attention_mask'],
            'offsets': output[0].offsets
        }


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.faq_question_answering_preprocessor)
class FaqQuestionAnsweringPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        super(FaqQuestionAnsweringPreprocessor, self).__init__(
            model_dir, pair=False, mode=ModeKeys.INFERENCE, **kwargs)
        import os
        from transformers import BertTokenizer

        from modelscope.utils.config import Config
        from modelscope.utils.constant import ModelFile
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


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.document_segmentation)
class DocumentSegmentationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, config, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """

        super().__init__(*args, **kwargs)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_dir,
            use_fast=True,
        )
        self.question_column_name = 'labels'
        self.context_column_name = 'sentences'
        self.example_id_column_name = 'example_id'
        self.label_to_id = {'B-EOP': 0, 'O': 1}
        self.target_specical_ids = set()
        self.target_specical_ids.add(self.tokenizer.eos_token_id)
        self.max_seq_length = config.max_position_embeddings
        self.label_list = ['B-EOP', 'O']

    def __call__(self, examples) -> Dict[str, Any]:
        questions = examples[self.question_column_name]
        contexts = examples[self.context_column_name]
        example_ids = examples[self.example_id_column_name]
        num_examples = len(questions)

        sentences = []
        for sentence_list in contexts:
            sentence_list = [_ + '[EOS]' for _ in sentence_list]
            sentences.append(sentence_list)

        try:
            tokenized_examples = self.tokenizer(
                sentences,
                is_split_into_words=True,
                add_special_tokens=False,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
        except Exception as e:
            logger.error(e)
            return {}

        segment_ids = []
        token_seq_labels = []
        for example_index in range(num_examples):
            example_input_ids = tokenized_examples['input_ids'][example_index]
            example_labels = questions[example_index]
            example_labels = [
                self.label_to_id[_] if _ in self.label_to_id else -100
                for _ in example_labels
            ]
            example_token_labels = []
            segment_id = []
            cur_seg_id = 1
            for token_index in range(len(example_input_ids)):
                if example_input_ids[token_index] in self.target_specical_ids:
                    example_token_labels.append(example_labels[cur_seg_id - 1])
                    segment_id.append(cur_seg_id)
                    cur_seg_id += 1
                else:
                    example_token_labels.append(-100)
                    segment_id.append(cur_seg_id)

            segment_ids.append(segment_id)
            token_seq_labels.append(example_token_labels)

        tokenized_examples['segment_ids'] = segment_ids
        tokenized_examples['token_seq_labels'] = token_seq_labels

        new_segment_ids = []
        new_token_seq_labels = []
        new_input_ids = []
        new_token_type_ids = []
        new_attention_mask = []
        new_example_ids = []
        new_sentences = []

        for example_index in range(num_examples):
            example_input_ids = tokenized_examples['input_ids'][example_index]
            example_token_type_ids = tokenized_examples['token_type_ids'][
                example_index]
            example_attention_mask = tokenized_examples['attention_mask'][
                example_index]
            example_segment_ids = tokenized_examples['segment_ids'][
                example_index]
            example_token_seq_labels = tokenized_examples['token_seq_labels'][
                example_index]
            example_sentences = contexts[example_index]
            example_id = example_ids[example_index]
            example_total_num_sentences = len(questions[example_index])
            example_total_num_tokens = len(
                tokenized_examples['input_ids'][example_index])
            accumulate_length = [
                i for i, x in enumerate(tokenized_examples['input_ids']
                                        [example_index])
                if x == self.tokenizer.eos_token_id
            ]
            samples_boundary = []
            left_index = 0
            sent_left_index = 0
            sent_i = 0

            # for sent_i, length in enumerate(accumulate_length):
            while sent_i < len(accumulate_length):
                length = accumulate_length[sent_i]
                right_index = length + 1
                sent_right_index = sent_i + 1
                if right_index - left_index >= self.max_seq_length - 1 or right_index == example_total_num_tokens:
                    samples_boundary.append([left_index, right_index])

                    sample_input_ids = [
                        self.tokenizer.cls_token_id
                    ] + example_input_ids[left_index:right_index]
                    sample_input_ids = sample_input_ids[:self.max_seq_length]

                    sample_token_type_ids = [
                        0
                    ] + example_token_type_ids[left_index:right_index]
                    sample_token_type_ids = sample_token_type_ids[:self.
                                                                  max_seq_length]

                    sample_attention_mask = [
                        1
                    ] + example_attention_mask[left_index:right_index]
                    sample_attention_mask = sample_attention_mask[:self.
                                                                  max_seq_length]

                    sample_segment_ids = [
                        0
                    ] + example_segment_ids[left_index:right_index]
                    sample_segment_ids = sample_segment_ids[:self.
                                                            max_seq_length]

                    sample_token_seq_labels = [
                        -100
                    ] + example_token_seq_labels[left_index:right_index]
                    sample_token_seq_labels = sample_token_seq_labels[:self.
                                                                      max_seq_length]

                    if sent_right_index - 1 == sent_left_index:
                        left_index = right_index
                        sample_input_ids[-1] = self.tokenizer.eos_token_id
                        sample_token_seq_labels[-1] = -100
                    else:
                        left_index = accumulate_length[sent_i - 1] + 1
                        if sample_token_seq_labels[-1] != -100:
                            sample_token_seq_labels[-1] = -100

                    if sent_right_index - 1 == sent_left_index or right_index == example_total_num_tokens:
                        sample_sentences = example_sentences[
                            sent_left_index:sent_right_index]
                        sent_left_index = sent_right_index
                        sent_i += 1
                    else:
                        sample_sentences = example_sentences[
                            sent_left_index:sent_right_index - 1]
                        sent_left_index = sent_right_index - 1

                    if (len([_ for _ in sample_token_seq_labels if _ != -100
                             ])) != len(sample_sentences) - 1 and (len([
                                 _
                                 for _ in sample_token_seq_labels if _ != -100
                             ])) != len(sample_sentences):
                        tmp = []
                        for w_i, w, l in zip(
                                sample_input_ids,
                                self.tokenizer.decode(sample_input_ids).split(
                                    ' '), sample_token_seq_labels):
                            tmp.append((w_i, w, l))
                    while len(sample_input_ids) < self.max_seq_length:
                        sample_input_ids.append(self.tokenizer.pad_token_id)
                        sample_token_type_ids.append(0)
                        sample_attention_mask.append(0)
                        sample_segment_ids.append(example_total_num_sentences
                                                  + 1)
                        sample_token_seq_labels.append(-100)

                    new_input_ids.append(sample_input_ids)
                    new_token_type_ids.append(sample_token_type_ids)
                    new_attention_mask.append(sample_attention_mask)
                    new_segment_ids.append(sample_segment_ids)
                    new_token_seq_labels.append(sample_token_seq_labels)
                    new_example_ids.append(example_id)
                    new_sentences.append(sample_sentences)
                else:
                    sent_i += 1
                    continue

        output_samples = {}

        output_samples['input_ids'] = new_input_ids
        output_samples['token_type_ids'] = new_token_type_ids
        output_samples['attention_mask'] = new_attention_mask

        output_samples['segment_ids'] = new_segment_ids
        output_samples['example_id'] = new_example_ids
        output_samples['labels'] = new_token_seq_labels
        output_samples['sentences'] = new_sentences

        return output_samples


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.fill_mask_ponet)
class FillMaskPoNetPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in MLM task.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        kwargs['truncation'] = kwargs.get('truncation', True)
        kwargs['padding'] = kwargs.get('padding', 'max_length')
        kwargs['max_length'] = kwargs.pop('sequence_length', 512)
        kwargs['return_token_type_ids'] = kwargs.get('return_token_type_ids',
                                                     True)
        super().__init__(model_dir, pair=False, mode=mode, **kwargs)

        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.language = self.cfg.model.get('language', 'en')
        if self.language == 'en':
            from nltk.tokenize import sent_tokenize
            import_external_nltk_data(
                osp.join(model_dir, 'nltk_data'), 'tokenizers/punkt')
        elif self.language in ['zh', 'cn']:

            def sent_tokenize(para):
                para = re.sub(r'([。！!？\?])([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\.{6})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'(\…{2})([^”’])', r'\1\n\2', para)  # noqa *
                para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2',
                              para)  # noqa *
                para = para.rstrip()
                return [_ for _ in para.split('\n') if _]
        else:
            raise NotImplementedError

        self.sent_tokenize = sent_tokenize
        self.max_length = kwargs['max_length']

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
        max_seq_length = self.max_length

        if text_b is None:
            segment_ids = []
            seg_lens = list(
                map(
                    len,
                    self.tokenizer(
                        self.sent_tokenize(text_a),
                        add_special_tokens=False,
                        truncation=True)['input_ids']))
            segment_id = [0] + sum(
                [[i] * sl for i, sl in enumerate(seg_lens, start=1)], [])
            segment_id = segment_id[:max_seq_length - 1]
            segment_ids.append(segment_id + [segment_id[-1] + 1]
                               * (max_seq_length - len(segment_id)))
            output['segment_ids'] = segment_ids

        output = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in output.items()
        }

        self.labels_to_id(labels, output)
        return output
