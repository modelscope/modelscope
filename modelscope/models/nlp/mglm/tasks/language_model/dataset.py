# Copyright (c) 2022 Zhipu.AI

import math
from bisect import bisect_right
from itertools import accumulate

import json
import numpy as np
import torch
from tasks.data_utils import build_input_from_ids, num_special_tokens_to_add
from tasks.language_model.detokenizer import get_detokenizer
from utils import print_rank_0


class LMDataset(torch.utils.data.Dataset):

    def __init__(self, args, documents, tokenizer, num_original_tokens,
                 num_tokenized_tokens):
        self.args = args
        self.documents = documents
        self.max_seq_len = args.seq_length - 1
        self.tokenizer = tokenizer
        self.overalapping_eval = args.overlapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.max_seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        # remove first sequence tokens
        targets = [
            max(len(tokens) - self.max_seq_len, 0) for tokens in self.documents
        ]
        self.num_sequences = [
            max(math.ceil(target / self.overalapping_eval) + 1, 1)
            for target in targets
        ]
        self.weights = list(accumulate(self.num_sequences))
        self.left_weights = [0] + self.weights[:-1]
        self.unidirectional = args.unidirectional
        self.block_lm = args.block_lm
        mask_token = 'gMASK' if args.task_mask else 'MASK'
        self.mask_id = self.tokenizer.get_command(mask_token).Id

    def __len__(self):
        return sum(self.num_sequences)

    def __getitem__(self, idx):
        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.max_seq_len
        tokens = self.documents[document_idx][start_idx:end_idx]
        if self.block_lm:
            if idx == 0 or self.unidirectional:
                prompt, text = tokens[:1], tokens[1:]
            else:
                prompt_length = self.max_seq_len - self.overalapping_eval
                prompt, text = tokens[:prompt_length], tokens[prompt_length:]
            prompt = prompt + [self.mask_id]
            num_special_tokens = num_special_tokens_to_add(
                prompt,
                None,
                text,
                add_cls=True,
                add_sep=False,
                add_piece=True,
                add_eos=False)
            data = build_input_from_ids(
                prompt,
                None,
                text,
                self.max_seq_len + num_special_tokens + 1,
                self.tokenizer,
                args=self.args,
                add_cls=True,
                add_sep=False,
                add_piece=True,
                add_eos=False,
                mask_id=self.mask_id)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if idx != 0 and self.unidirectional:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                loss_masks[:-self.overalapping_eval] = 0
            return {
                'text': np.array(ids, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_masks, dtype=np.int64),
                'position_id': np.array(position_ids, dtype=np.int64)
            }
        else:
            loss_masks = [1] * len(tokens)
            if len(tokens) < self.max_seq_len:
                tokens = tokens + [0] * (self.max_seq_len - len(tokens))
                loss_masks = loss_masks + [0] * (
                    self.max_seq_len - len(loss_masks))
            if idx != 0:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                loss_masks[:-self.overalapping_eval] = 0
            return {
                'text': np.array(tokens, dtype=np.int64),
                'loss_mask': np.array(loss_masks, dtype=np.int64)
            }


class LambadaDataset(torch.utils.data.Dataset):

    def __init__(self, args, tokenizer, strict=True):
        data_path = args.valid_data[0]
        print_rank_0(
            '> building lambada dataset from {} ...'.format(data_path))
        self.args = args
        self.max_seq_length = args.seq_length
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.get_command('pad').Id
        self.strict = strict
        self.block_lm = args.block_lm
        self.unidirectional = args.unidirectional
        mask_token = 'gMASK' if args.task_mask else 'MASK'
        self.mask_id = self.tokenizer.get_command(mask_token).Id

        self.tokens = []
        self.labels = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.EncodeAsIds(text).tokenization
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.EncodeAsIds(
            text[:start_idx].strip()).tokenization
        last_token = self.tokenizer.EncodeAsIds(' ' + last_token).tokenization
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens, answer = self.tokens[idx], self.labels[idx]
        if self.block_lm:
            if self.unidirectional:
                tokens, answer_tokens = tokens[:1], tokens[1:] + answer
            else:
                answer_tokens = answer
            tokens = tokens + [self.mask_id]
            num_special_tokens = num_special_tokens_to_add(
                tokens,
                None,
                answer_tokens,
                add_cls=True,
                add_sep=False,
                add_piece=True)
            left_shift = len(tokens) + len(
                answer_tokens) + num_special_tokens - self.max_seq_length
            if left_shift > 0:
                tokens = tokens[left_shift:]
            data = build_input_from_ids(
                tokens,
                None,
                answer_tokens,
                self.max_seq_length,
                self.tokenizer,
                args=self.args,
                add_cls=True,
                add_sep=False,
                add_piece=True,
                mask_id=self.mask_id)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if self.unidirectional:
                loss_masks = np.array(loss_masks, dtype=np.int64)
                last_index = len(loss_masks)
                while loss_masks[last_index - 1] == 0:
                    last_index -= 1
                loss_masks[:last_index - len(answer)] = 0
            return {
                'text': np.array(ids, dtype=np.int64),
                'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64),
                'loss_mask': np.array(loss_masks, dtype=np.int64),
                'position_id': np.array(position_ids, dtype=np.int64)
            }
        else:
            left_shift = len(tokens) - self.max_seq_length
            if left_shift > 0:
                tokens = tokens[left_shift:]
            ids = tokens + answer
            if len(ids) < self.max_seq_length:
                ids = ids + [0] * (self.max_seq_length - len(ids))
            loss_masks = [0] * len(tokens) + [1] * len(answer)
            if len(loss_masks) < self.max_seq_length:
                loss_masks = loss_masks + [0] * (
                    self.max_seq_length - len(loss_masks))
            return {
                'text': np.array(ids, dtype=np.int64),
                'loss_mask': np.array(loss_masks, dtype=np.int64)
            }


def build_lambada_dataset(tokenizer, args):
    """Build lambada dataset."""
    assert len(args.valid_data) == 1
    val_dataset = LambadaDataset(args, tokenizer, strict=True)
    print_rank_0(' > found {} samples, {} label tokens.'.format(
        len(val_dataset), sum(map(len, val_dataset.labels))))
    return val_dataset


def build_lm_dataset(tokenizer, args):
    documents = []
    num_tokens, num_original_tokens = 0, 0
    with open(args.valid_data[0], encoding='utf-8') as file:
        for line in file:
            tokens = tokenizer.EncodeAsIds(line.strip()).tokenization
            num_tokens += len(tokens)
            num_original_tokens += len(line.strip().split(' '))
            documents.append(tokens)
    val_dataset = LMDataset(args, documents, tokenizer, num_original_tokens,
                            num_tokens)
    print_rank_0(
        ' > number of document: {}, number of original tokens {}, number of detokenized tokens: {}'
        .format(len(documents), num_original_tokens, num_tokens))
    return val_dataset


def build_wikitext103_dataset(tokenizer, args):
    """"""

    assert len(args.valid_data) == 1
    with open(args.valid_data[0], 'rb') as reader:
        entire_data = reader.read().decode('utf-8')
    num_original_tokens = len(entire_data.strip().split(' '))
    entire_data = get_detokenizer('wikitext')(entire_data)
    print_rank_0(entire_data[:1024])
    tokenized_data = tokenizer.EncodeAsIds(entire_data).tokenization
    num_tokenized_tokens = len(tokenized_data)

    val_dataset = LMDataset(args, [tokenized_data], tokenizer,
                            num_original_tokens, num_tokenized_tokens)
    print_rank_0(' > number of original tokens: {}, number of detokenized '
                 'tokens: {}'.format(num_original_tokens,
                                     num_tokenized_tokens))
    return val_dataset
