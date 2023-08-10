# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import logging

from modelscope.metainfo import Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config
from modelscope.utils.constant import ConfigFields, Fields, ModeKeys, ModelFile

logger = logging.get_logger(__name__)
MULTI_SEP_TOKENS_TOKENIZERS_SET = {'roberta', 'camembert', 'bart', 'mpnet'}


@PREPROCESSORS.register_module(
    Fields.nlp,
    module_name=Preprocessors.machine_reading_comprehension_for_ner)
class MachineReadingComprehensionForNERPreprocessor(Preprocessor):
    '''
    Preprocessor for Pretrained Machiner Reader (PMR) finetuned on Named Entity Recognition (NER)

    '''

    def __init__(self, model_dir, label2query=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, use_fast=False)

        if label2query is None:
            config_path = os.path.join(model_dir, ModelFile.CONFIGURATION)
            config = Config.from_file(config_path)
            self.label2query = config[ConfigFields.preprocessor].label2query
        else:
            self.label2query = label2query

    def __call__(self, data: str):
        all_data = []
        for label in self.label2query:
            all_data.append({
                'context': data,
                'end_position': [],
                'entity_label': label,
                'impossible': False,
                'qas_id': '',
                'query': self.label2query[label],
                'span_position': [],
                'start_position': []
            })

        all_data = self.prompt(all_data)
        output = []
        for data in all_data:
            output.append(self.encode(data))
        output = collate_to_max_length_roberta(output)

        output = {
            'input_ids': output[0],
            'attention_mask': output[1],
            'token_type_ids': output[2],
        }

        return output

    def prompt(self, all_data, var=0):
        new_datas = []
        for data in all_data:
            label = data['entity_label']
            details = data['query']
            context = data['context']
            start_positions = data['start_position']
            end_positions = data['end_position']
            words = context.split()
            assert len(words) == len(context.split(' '))
            if var == 0:
                query = '"{}". {}'.format(label, details)  # ori
            elif var == 1:
                query = 'What are the "{}" entity, where {}'.format(
                    label, details)  # variant 1
            elif var == 2:
                query = 'Identify the spans (if any) related to "{}" entity. Details: {}'.format(
                    label, details)  # variant 2
            span_positions = {
                '{};{}'.format(start_positions[i], end_positions[i]):
                ' '.join(words[start_positions[i]:end_positions[i] + 1])
                for i in range(len(start_positions))
            }
            new_data = {
                'context': words,
                'end_position': end_positions,
                'entity_label': label,
                'impossible': data['impossible'],
                'qas_id': data['qas_id'],
                'query': query,
                'span_position': span_positions,
                'start_position': start_positions,
            }
            new_datas.append(new_data)
        return new_datas

    def encode(self, data, max_length=512, max_query_length=64):

        tokenizer = self.tokenizer

        query = data['query']
        context = data['context']
        start_positions = data['start_position']
        end_positions = data['end_position']

        tokenizer_type = type(tokenizer).__name__.replace('Tokenizer',
                                                          '').lower()
        sequence_added_tokens = (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
            + 1 if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET else
            tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(context):
            orig_to_tok_index.append(len(all_doc_tokens))
            if tokenizer.__class__.__name__ in [
                    'RobertaTokenizer',
                    'LongformerTokenizer',
                    'BartTokenizer',
                    'RobertaTokenizerFast',
                    'LongformerTokenizerFast',
                    'BartTokenizerFast',
            ]:
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            elif tokenizer.__class__.__name__ in ['BertTokenizer']:
                sub_tokens = tokenizer.tokenize(token)
            elif tokenizer.__class__.__name__ in ['BertWordPieceTokenizer']:
                sub_tokens = tokenizer.encode(
                    token, add_special_tokens=False).tokens
            else:
                sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_positions = [orig_to_tok_index[x] for x in start_positions]
        tok_end_positions = []
        for x in end_positions:
            if x < len(context) - 1:
                tok_end_positions.append(orig_to_tok_index[x + 1] - 1)
            else:
                tok_end_positions.append(len(all_doc_tokens) - 1)

        truncation = TruncationStrategy.ONLY_SECOND.value
        padding_strategy = 'do_not_pad'

        truncated_query = tokenizer.encode(
            query,
            add_special_tokens=False,
            truncation=True,
            max_length=max_query_length)
        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            truncated_query,
            all_doc_tokens,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_token_type_ids=True,
        )
        tokens = encoded_dict['input_ids']
        type_ids = encoded_dict['token_type_ids']
        attn_mask = encoded_dict['attention_mask']

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. special tokens
        doc_offset = len(truncated_query) + sequence_added_tokens
        new_start_positions = [
            x + doc_offset for x in tok_start_positions
            if (x + doc_offset) < max_length - 1
        ]
        new_end_positions = [
            x + doc_offset if
            (x + doc_offset) < max_length - 1 else max_length - 2
            for x in tok_end_positions
        ]
        new_end_positions = new_end_positions[:len(new_start_positions)]

        label_mask = [0] * doc_offset + [1] * (len(tokens) - doc_offset
                                               - 1) + [0]

        assert all(label_mask[p] != 0 for p in new_start_positions)
        assert all(label_mask[p] != 0 for p in new_end_positions)

        assert len(label_mask) == len(tokens)

        seq_len = len(tokens)
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)
        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(attn_mask),
            torch.LongTensor(type_ids),
            torch.LongTensor(label_mask),
            match_labels,
        ]


def collate_to_max_length_roberta(batch):
    """
    adapted form https://github.com/ShannonAI/mrc-for-flat-nested-ner
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask,
            end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(4):
        if field_idx == 0:
            pad_output = torch.full([batch_size, max_length],
                                    1,
                                    dtype=batch[0][field_idx].dtype)
        else:
            pad_output = torch.full([batch_size, max_length],
                                    0,
                                    dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][:data.shape[0]] = data
        output.append(pad_output)

    pad_match_labels = torch.zeros([batch_size, max_length, max_length],
                                   dtype=torch.long)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][4]
        pad_match_labels[sample_idx, :data.shape[1], :data.shape[1]] = data
    output.append(pad_match_labels)

    return output
