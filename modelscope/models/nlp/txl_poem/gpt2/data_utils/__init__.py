# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""utils for creating datasets"""
import math
import os
import time

from . import corpora
from .datasets import (ConcatDataset, GPT2Dataset, ShuffleDataset,
                       SplitDataset, XLDataset, bert_sentencepair_dataset,
                       csv_dataset, json_dataset, split_ds)
from .lazy_loader import LazyLoader, LazyWriter, exists_lazy
from .samplers import DistributedBatchSampler
from .tokenization import (BertWordPieceTokenizer, CharacterLevelTokenizer,
                           CommandToken, GPT2BPETokenizer, Tokenization,
                           Tokenizer, make_tokenizer)

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2


def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0])
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split) / sum(split) != 1.


def get_ext(path):
    """gets path extension"""
    return os.path.splitext(path)[1]


def get_dataset(name, tokenizer, pre_tokenize, local_rank):
    """gets dataset object based on keyword args and file at `path`"""
    if supported_corpus(name):
        dataset = corpora.NAMED_CORPORA[name]
        path = dataset.PATH
        if issubclass(dataset, corpora.PromptReader):
            if not (exists_lazy(path, data_type='prompt')
                    and exists_lazy(path, data_type='text')):
                # create cached version of dataset for lazy loading if it doesn't exist
                if local_rank == 0:
                    prompt_writer = LazyWriter(
                        path, data_type='prompt', is_array=pre_tokenize)
                    text_writer = LazyWriter(
                        path, data_type='text', is_array=pre_tokenize)
                    writers = {'prompt': prompt_writer, 'text': text_writer}
                    dataset(
                        writers=writers,
                        tokenizer=tokenizer,
                        tokenize=pre_tokenize)
                    prompt_writer.close()
                    text_writer.close()
                else:
                    while not os.path.exists(
                            LazyWriter.get_len_path(path, data_type='prompt')):
                        time.sleep(1)
            map_fn = (lambda x: x.tolist()) if pre_tokenize else None
            prompts = LazyLoader(
                path,
                data_type='prompt',
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize)
            texts = LazyLoader(
                path,
                data_type='text',
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize)
            text = corpora.PromptDataset(
                prompt_loader=prompts,
                text_loader=texts,
                tokenizer=tokenizer,
                to_tokenize=not pre_tokenize)
            return text
        elif issubclass(dataset, corpora.KeyReader):
            if not (exists_lazy(path, data_type='text')
                    and exists_lazy(path, data_type='mask')):
                # create cached version of dataset for lazy loading if it doesn't exist
                if local_rank == 0:
                    text_writer = LazyWriter(
                        path, data_type='text', is_array=pre_tokenize)
                    mask_writer = LazyWriter(
                        path, data_type='mask', is_array=True)
                    writers = {'mask': mask_writer, 'text': text_writer}
                    dataset(
                        writers=writers,
                        tokenizer=tokenizer,
                        tokenize=pre_tokenize)
                    mask_writer.close()
                    text_writer.close()
                else:
                    while not os.path.exists(
                            LazyWriter.get_len_path(path, data_type='mask')):
                        time.sleep(1)
            map_fn = (lambda x: x.tolist()) if pre_tokenize else None
            masks = LazyLoader(
                path,
                data_type='mask',
                map_fn=map_fn,
                mem_map=True,
                is_array=True)
            texts = LazyLoader(
                path,
                data_type='text',
                map_fn=map_fn,
                mem_map=True,
                is_array=pre_tokenize)
            text = corpora.KeyDataset(
                mask_loader=masks,
                text_loader=texts,
                tokenizer=tokenizer,
                to_tokenize=not pre_tokenize)
            return text
    else:
        raise NotImplementedError('dataset %s is not supported' % name)


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(path,
                 seq_length,
                 mem_length,
                 local_rank,
                 lazy=False,
                 xl_style=False,
                 shuffle=True,
                 split=None,
                 tokenizer=None,
                 tokenizer_type='CharacterLevelTokenizer',
                 tokenizer_model_path=None,
                 vocab_size=None,
                 model_type='bpe',
                 pad_token=0,
                 character_converage=1.0,
                 non_binary_cols=None,
                 sample_one_document=False,
                 pre_tokenize=False,
                 **kwargs):
    """function to create datasets+tokenizers for common options"""
    if split is None:
        split = [1.]
    if non_binary_cols is not None:
        # multilabel dataset support (only for csvs)
        label_key = non_binary_cols  # noqa

        # make tokenizer for dataset
    if tokenizer is None:
        tokenizer = make_tokenizer(tokenizer_type, None, tokenizer_model_path,
                                   vocab_size, model_type, pad_token,
                                   character_converage, **kwargs)

    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        ds = get_dataset(
            path,
            tokenizer=tokenizer,
            pre_tokenize=pre_tokenize,
            local_rank=local_rank)
    else:
        ds = [
            get_dataset(
                p,
                tokenizer=tokenizer,
                pre_tokenize=pre_tokenize,
                local_rank=local_rank) for p in path
        ]
        ds = ConcatDataset(ds)

    ds_type = ''
    if 'ds_type' in kwargs:
        ds_type = kwargs['ds_type']
    # Split dataset into train/val/test (and wrap bert dataset)
    if should_split(split):
        ds = split_ds(ds, split, shuffle=shuffle)
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs[
                'presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = [
                bert_sentencepair_dataset(
                    d,
                    max_seq_len=seq_length,
                    presplit_sentences=presplit_sentences)
                if d is not None else None for d in ds
            ]
        elif ds_type.lower() == 'gpt2':
            if xl_style:
                ds = [
                    XLDataset(
                        d,
                        tokenizer,
                        max_seq_len=seq_length,
                        mem_len=mem_length,
                        sample_across_doc=not sample_one_document)
                    if d is not None else None for d in ds
                ]
            else:
                ds = [
                    GPT2Dataset(
                        d,
                        tokenizer,
                        max_seq_len=seq_length,
                        sample_across_doc=not sample_one_document)
                    if d is not None else None for d in ds
                ]
    else:
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs[
                'presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = bert_sentencepair_dataset(
                ds,
                max_seq_len=seq_length,
                presplit_sentences=presplit_sentences)
        elif ds_type.lower() == 'gpt2':
            if xl_style:
                ds = XLDataset(
                    ds,
                    tokenizer,
                    max_seq_len=seq_length,
                    mem_len=mem_length,
                    sample_across_doc=not sample_one_document)
            else:
                ds = GPT2Dataset(
                    ds,
                    tokenizer,
                    max_seq_len=seq_length,
                    sample_across_doc=not sample_one_document)
    return ds, tokenizer
