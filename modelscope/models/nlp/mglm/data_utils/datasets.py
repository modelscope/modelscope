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
"""dataset objects for jsons, csvs, and BERT datasets"""

import csv
import math
import os
import random
import time
from bisect import bisect_right
from itertools import accumulate
from operator import itemgetter

import json
import nltk
import numpy as np
import pandas as pd
import torch
import tqdm
from nltk import tokenize
from torch.utils import data

from modelscope.models.nlp.mglm.utils import print_rank_0
from .lazy_loader import LazyLoader, exists_lazy


class ShuffleDataset(data.Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.shuffle_ids = list(range(len(self.ds)))
        random.shuffle(self.shuffle_ids)
        self.is_lazy = hasattr(ds, 'is_lazy') and ds.is_lazy
        if self.is_lazy:
            self.prompt_lens = [
                self.ds.prompt_lens[idx] for idx in self.shuffle_ids
            ]
            self.text_lens = [
                self.ds.text_lens[idx] for idx in self.shuffle_ids
            ]

    def __getitem__(self, idx):
        return self.ds[self.shuffle_ids[idx]]

    def __len__(self):
        return len(self.ds)


class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)  # noqa
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.is_lazy = sum([
            isinstance(ds, LazyLoader)
            or (hasattr(ds, 'is_lazy') and ds.is_lazy) for ds in self.datasets
        ]) == len(self.datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None
        self._lens = None

    def get_text_len(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_text_len(sample_idx)

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.datasets[0].GetTokenizer()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def lens(self):
        if self._lens is None:
            self._lens = []
            if self.is_lazy:
                for data in self.datasets:  # noqa
                    self._lens.extend(data.lens)
            else:
                for data in self.datasets:  # noqa
                    self._lens.extend([
                        len(d['text']) if isinstance(d, dict) else len(d)
                        for d in data
                    ])
        return self._lens

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:  # noqa
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:  # noqa
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y


class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """

    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy')
                                                      and ds.is_lazy)
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def get_text_len(self, idx):
        return self.wrapped_data.get_text_len(self.split_inds[idx])

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_data.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.wrapped_data.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(
                itemgetter(*self.split_inds)(self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]


def split_ds(ds, split=None, shuffle=True, save_splits=None, load_splits=None):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
        save_splits: save split indices to file
        load_splits: load split indices from file
    """
    if split is None:
        split = [.8, .2, .0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    if load_splits is not None:
        inds = np.load(load_splits)
        assert len(inds) == ds_len
        print_rank_0(f'Load split indices from {load_splits}')
    elif save_splits is not None:
        if torch.distributed.get_rank() == 0:
            np.save(save_splits, inds)
            print(f'Save split indices to {save_splits}')
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


class csv_dataset(data.Dataset):
    """
    Class for loading datasets from csv files.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): Path to csv file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): Callable that process a string into desired format.
        delim (str): delimiter for csv. Default: ','
        binarize_sent (bool): binarize label values to 0 or 1 if they\'re on a different scale. Default: False
        drop_unlabeled (bool): drop rows with unlabelled values. Always fills remaining empty
            columns with -1 (regardless if rows are dropped based on value) Default: False
        text_key (str): key to get text from csv. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        X (list): all strings from the csv file
        Y (np.ndarray): labels to train with
    """

    def __init__(self,
                 path,
                 tokenizer=None,
                 preprocess_fn=None,
                 delim=',',
                 binarize_sent=False,
                 drop_unlabeled=False,
                 text_key='sentence',
                 label_key='label',
                 **kwargs):
        self.is_lazy = False
        self.preprocess_fn = preprocess_fn
        self.SetTokenizer(tokenizer)
        self.path = path
        self.delim = delim
        self.text_key = text_key
        self.label_key = label_key
        self.drop_unlabeled = drop_unlabeled

        if '.tsv' in self.path:
            self.delim = '\t'

        self.X = []
        self.Y = []
        try:
            cols = [text_key]
            if isinstance(label_key, list):
                cols += label_key
            else:
                cols += [label_key]
            data = pd.read_csv(
                self.path, sep=self.delim, usecols=cols, encoding='latin-1')
        except:  # noqa
            data = pd.read_csv(
                self.path,
                sep=self.delim,
                usecols=[text_key],
                encoding='latin-1')

        data = data.dropna(axis=0)

        self.X = data[text_key].values.tolist()
        try:
            self.Y = data[label_key].values
        except Exception as e:  # noqa
            self.Y = np.ones(len(self.X)) * -1

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        if tokenizer is None:
            self.using_tokenizer = False
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self.using_tokenizer = True
            self._tokenizer = tokenizer

    def GetTokenizer(self):
        return self._tokenizer

    @property
    def tokenizer(self):
        if self.using_tokenizer:
            return self._tokenizer
        return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """process+tokenize string and return string,label,and stringlen"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        if isinstance(y, str):
            if self.tokenizer is not None:
                y = self.tokenizer.EncodeAsIds(y, self.preprocess_fn)
            elif self.preprocess_fn is not None:
                y = self.preprocess_fn(y)
        return {'text': x, 'length': len(x), 'label': y}

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a csv file
        """
        if path is None:
            path = self.path + '.results'
        print('generating csv at ' + path)
        with open(path, 'w') as csvfile:
            c = csv.writer(csvfile, delimiter=self.delim)
            if writer_gen is not None:
                # if first item of generator is a header of what the metrics mean then write header to csv file
                if not skip_header:
                    header = (self.label_key, ) + tuple(
                        next(writer_gen)) + (self.text_key, )
                    c.writerow(header)
                for i, row in enumerate(writer_gen):
                    row = (self.Y[i], ) + tuple(row) + (self.X[i], )
                    c.writerow(row)
            else:
                c.writerow([self.label_key, self.text_key])
                for row in zip(self.Y, self.X):
                    c.writerow(row)


class json_dataset(data.Dataset):
    """
    Class for loading datasets from a json dump.
    Purpose: Useful for loading data for unsupervised modeling or transfer tasks
    Arguments:
        path (str): path to json file with dataset.
        tokenizer (data_utils.Tokenizer): Tokenizer to use when processing text. Default: None
        preprocess_fn (callable): callable function that process a string into desired format.
            Takes string, maxlen=None, encode=None as arguments. Default: process_str
        text_key (str): key to get text from json dictionary. Default: 'sentence'
        label_key (str): key to get label from json dictionary. Default: 'label'
    Attributes:
        all_strs (list): list of all strings from the dataset
        all_labels (list): list of all labels from the dataset (if they have it)
    """

    def __init__(self,
                 path,
                 tokenizer=None,
                 preprocess_fn=None,
                 binarize_sent=False,
                 text_key='sentence',
                 label_key='label',
                 loose_json=False,
                 **kwargs):
        self.is_lazy = False
        self.preprocess_fn = preprocess_fn
        self.path = path
        self.SetTokenizer(tokenizer)
        self.X = []
        self.Y = []
        self.text_key = text_key
        self.label_key = label_key
        self.loose_json = loose_json

        for j in self.load_json_stream(self.path):
            s = j[text_key]
            self.X.append(s)
            self.Y.append(j[label_key])

        if binarize_sent:
            self.Y = binarize_labels(self.Y, hard=binarize_sent)

    def SetTokenizer(self, tokenizer):
        if tokenizer is None:
            self.using_tokenizer = False
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self.using_tokenizer = True
            self._tokenizer = tokenizer

    def GetTokenizer(self):
        return self._tokenizer

    @property
    def tokenizer(self):
        if self.using_tokenizer:
            return self._tokenizer
        return None

    def __getitem__(self, index):
        """gets the index'th string from the dataset"""
        x = self.X[index]
        if self.tokenizer is not None:
            x = self.tokenizer.EncodeAsIds(x, self.preprocess_fn)
        elif self.preprocess_fn is not None:
            x = self.preprocess_fn(x)
        y = self.Y[index]
        if isinstance(y, str):
            if self.tokenizer is not None:
                y = self.tokenizer.EncodeAsIds(y, self.preprocess_fn)
            elif self.preprocess_fn is not None:
                y = self.preprocess_fn(y)
        return {'text': x, 'length': len(x), 'label': y}

    def __len__(self):
        return len(self.X)

    def write(self, writer_gen=None, path=None, skip_header=False):
        """
        given a generator of metrics for each of the data points X_i,
            write the metrics, text, and labels to a json file
        """
        if path is None:
            path = self.path + '.results'

        if writer_gen is not None:
            # if first item of generator is a header of what the metrics mean then write header to csv file
            def gen_helper():
                keys = {}
                keys[0] = self.label_key
                if not skip_header:
                    for idx, k in enumerate(tuple(next(writer_gen))):
                        keys[idx + 1] = k
                for i, row in enumerate(writer_gen):
                    if i == 0 and skip_header:
                        for idx, _ in enumerate(row):
                            keys[idx + 1] = 'metric_%d' % (idx, )
                    j = {}
                    for idx, v in enumerate((self.Y[i], ) + tuple(row)):
                        k = keys[idx]
                        j[k] = v
                    yield j
        else:

            def gen_helper():
                for y in self.Y:
                    j = {}
                    j[self.label_key] = y
                    yield j

        def out_stream():
            for i, j in enumerate(gen_helper()):
                j[self.text_key] = self.X[i]
                yield j

        self.save_json_stream(path, out_stream())

    def save_json_stream(self, save_path, json_stream):
        if self.loose_json:
            with open(save_path, 'w') as f:
                for i, j in enumerate(json_stream):
                    write_string = ''
                    if i != 0:
                        write_string = '\n'
                    write_string += json.dumps(j)
                    f.write(write_string)
        else:
            jsons = [j for j in json_stream]
            json.dump(jsons, open(save_path, 'w'), separators=(',', ':'))

    def load_json_stream(self, load_path):
        if not self.loose_json:
            jsons = json.load(open(load_path, 'r', encoding='utf-8'))
            generator = iter(jsons)
        else:

            def gen_helper():
                with open(load_path, 'r', encoding='utf-8') as f:
                    for row in f:
                        yield json.loads(row)

            generator = gen_helper()

        for j in generator:
            if self.label_key not in j:
                j[self.label_key] = -1
            yield j


class XLDataset(data.Dataset):

    def __init__(self,
                 ds,
                 tokenizer,
                 max_seq_len=1024,
                 mem_len=None,
                 sample_across_doc=True,
                 **kwargs):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if mem_len is None:
            mem_len = max_seq_len
        self.mem_len = mem_len
        self.sample_across_doc = sample_across_doc
        self.indices, self.num_samples = None, None
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_indices()

    def init_indices(self):
        if self.is_lazy:
            lens = np.array(
                [self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([
                len(d['prompt'])
                + len(d['text']) if isinstance(d, dict) else len(d)
                for d in self.ds
            ])
        self.indices = list(accumulate(lens))
        print_rank_0(
            f'Dataset document count {len(lens)}, token count {self.indices[-1]}'
        )
        self.num_samples = self.indices[-1] // self.max_seq_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens, targets, loss_mask, attention_mask = self.getidx(idx)
        tokens = self.pad_seq(tokens)
        targets = self.pad_seq(targets)
        loss_mask = self.pad_seq(loss_mask, pad_id=0)
        return {
            'text': np.array(tokens),
            'target': np.array(targets),
            'loss_mask': np.array(loss_mask),
            'attention_mask': np.array(attention_mask)
        }

    def getidx(self, idx):
        tokens, targets, loss_masks = [], [], []
        attention_mask = np.concatenate(
            (np.zeros((self.max_seq_len, self.mem_len), dtype=int),
             np.ones((self.max_seq_len, self.max_seq_len), dtype=int)),
            axis=1)
        sample_idx = bisect_right(self.indices, idx * self.max_seq_len)
        last_end = 0 if sample_idx == 0 else self.indices[sample_idx - 1]
        token_offset = idx * self.max_seq_len - last_end
        if token_offset != 0:
            history = min(self.mem_len, token_offset)
            attention_mask[:,
                           -self.max_seq_len - history:-self.max_seq_len] = 1
        count = 0
        while len(tokens) < self.max_seq_len and sample_idx < len(self.ds):
            item = self.ds[sample_idx]
            text, masks = item['tokens'], item['loss_masks']
            text = text + [self.tokenizer.get_command('eos').Id]
            end = min(
                len(text) - 1, token_offset + self.max_seq_len - len(tokens))
            masks = masks + [1]
            if count > 0:
                current = len(tokens)
                attention_mask[current:, :current + self.mem_len] = 0
            tokens += text[token_offset:end]
            targets += text[token_offset + 1:end + 1]
            loss_masks += masks[token_offset + 1:end + 1]
            count += 1
            sample_idx += 1
            token_offset = 0
        return tokens, targets, loss_masks, attention_mask

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [
            self.tokenizer.get_command('pad').Id if pad_id is None else pad_id
        ] * (
            num_pad_tokens)
        return seq


class BlockDataset(data.Dataset):

    def __init__(self,
                 ds,
                 tokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start=0.0,
                 filter_english=False,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start = non_sentence_start
        self.filter_english = filter_english
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if self.filter_english:
            import fasttext
            self.model = fasttext.load_model('/mnt/lid.176.bin')
            print_rank_0('Load language detection model')
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.is_lazy:
            lens = np.array(
                [self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([
                len(d['text']) if isinstance(d, dict) else len(d)
                for d in self.ds
            ])
        self.total_len = np.sum(lens)
        print_rank_0(
            f'Dataset document count {len(lens)}, token count {self.total_len}, non sentence start{self.non_sentence_start}'  # noqa
        )
        self.weighting = list(accumulate(lens))

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.total_len)
            data_idx = bisect_right(self.weighting, idx)
            tokens, loss_mask = self.getidx(data_idx)
            if self.filter_english:
                text = self.tokenizer.DecodeIds(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        tokens, loss_mask = self.get_weighted_samples(rng)
        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(  # noqa
                            tokens[strip_left_tokens - 1]):  # noqa
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(
                                tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.get_command('ENC').Id
                      ] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.get_command(
                    'eos').Id:
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask,
                                                     self.max_seq_len)
        else:
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens
            loss_mask = [0] + loss_mask
            # Sample multiple documents
            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.get_command('ENC').Id
                                  ] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(
                        new_tokens, new_loss_mask,
                        self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {'text': np.array(tokens), 'loss_mask': np.array(loss_mask)}

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(
                    tokens) - 1 and not self.contains_sentence_end(
                        tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [
            self.tokenizer.get_command('pad').Id if pad_id is None else pad_id
        ] * (
            num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '\n' in tok:
            return True
        return False


class GPT2Dataset(data.Dataset):

    def __init__(self,
                 ds,
                 tokenizer,
                 max_seq_len=1024,
                 num_samples=None,
                 weighted=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 sentence_start=False,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.sentence_start = sentence_start
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array(
                    [self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([
                    len(d['text']) if isinstance(d, dict) else len(d)
                    for d in self.ds
                ])
            self.total_len = np.sum(lens)
            print_rank_0(
                f'Dataset document count {len(lens)}, token count {self.total_len}'
            )
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        tokens, loss_mask = self.getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len - 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]
            loss_mask = loss_mask[strip_left_tokens:]
            # if self.sentence_start:
            #     token_copy = list(tokens)
            #     not_done = True
            #     while (len(token_copy) > 0) and not_done:
            #         tok = token_copy.pop(0)
            #         if self.contains_sentence_end(tok):
            #             tokens = token_copy
            #             not_done = False
            strip_right_rokens = len(tokens) - self.max_seq_len - 1
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]
                loss_mask = loss_mask[:-strip_right_rokens]
        # Sample multiple documents
        if self.sample_across_doc:
            while (len(tokens) < (self.max_seq_len + 1)):
                if self.random_across_doc_sampling:
                    data_idx = self.get_weighted_samples(rng)
                else:
                    data_idx = (data_idx + 1) % self.ds_len
                new_tokens, new_loss_mask = self.getidx(data_idx)
                tokens += new_tokens
                loss_mask += new_loss_mask
            tokens = tokens[:(self.max_seq_len + 1)]
            loss_mask = loss_mask[:(self.max_seq_len + 1)]

        tokens = self.pad_seq(tokens)
        loss_mask = self.pad_seq(loss_mask, pad_id=0)
        return {'text': np.array(tokens), 'loss_mask': np.array(loss_mask)}

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len + 1
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [
            self.tokenizer.get_command('pad').Id if pad_id is None else pad_id
        ] * (
            num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        return False


class BertSentencepairDataset(data.Dataset):
    """
    Dataset containing sentencepairs for BERT training. Each index corresponds to a randomly generated sentence pair.
    Arguments:
        ds (Dataset or array-like): data corpus to use for training
        max_seq_len (int): maximum sequence length to use for a sentence pair
        mask_lm_prob (float): proportion of tokens to mask for masked LM
        max_preds_per_seq (int): Maximum number of masked tokens per sentence pair. Default: math.ceil(max_seq_len*mask_lm_prob/10)*10
        short_seq_prob (float): Proportion of sentence pairs purposefully shorter than max_seq_len
        dataset_size (int): number of random sentencepairs in the dataset. Default: len(ds)*(len(ds)-1)

    """ # noqa

    def __init__(self,
                 ds,
                 max_seq_len=512,
                 mask_lm_prob=.15,
                 max_preds_per_seq=None,
                 short_seq_prob=.01,
                 dataset_size=None,
                 presplit_sentences=False,
                 weighted=True,
                 **kwargs):
        self.ds = ds
        self.ds_len = len(self.ds)
        self.tokenizer = self.ds.GetTokenizer()
        self.vocab_words = list(self.tokenizer.text_token_vocab.values())
        self.ds.SetTokenizer(None)
        self.max_seq_len = max_seq_len
        self.mask_lm_prob = mask_lm_prob
        if max_preds_per_seq is None:
            max_preds_per_seq = math.ceil(max_seq_len * mask_lm_prob / 10) * 10
        self.max_preds_per_seq = max_preds_per_seq
        self.short_seq_prob = short_seq_prob
        self.dataset_size = dataset_size
        if self.dataset_size is None:
            self.dataset_size = self.ds_len * (self.ds_len - 1)
        self.presplit_sentences = presplit_sentences
        if not self.presplit_sentences:
            nltk.download('punkt', download_dir='./nltk')
        self.weighted = weighted
        self.get_weighting()

    def get_weighting(self):
        if self.weighted:
            if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
                lens = np.array(self.ds.lens)
            else:
                lens = np.array([
                    len(d['text']) if isinstance(d, dict) else len(d)
                    for d in self.ds
                ])
            self.total_len = np.sum(lens)
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # get rng state corresponding to index (allows deterministic random pair)
        rng = random.Random(idx)
        np_rng = np.random.RandomState(
            seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])
        # get seq length
        target_seq_length = self.max_seq_len
        short_seq = False  # noqa
        if rng.random() < self.short_seq_prob:
            target_seq_length = rng.randint(2, target_seq_length)
            short_seq = True  # noqa

        # get sentence pair and label
        is_random_next = None
        lena = 0
        lenb = 0
        while (is_random_next is None) or (lena < 1) or (lenb < 1):
            tokensa, tokensb, is_random_next = self.create_random_sentencepair(
                target_seq_length, rng, np_rng)
            lena = len(tokensa[0])
            lenb = len(tokensb[0])

        # truncate sentence pair to max_seq_len
        tokensa, tokensb = self.truncate_seq_pair(tokensa, tokensb,
                                                  self.max_seq_len, rng)
        # join sentence pair, mask, and pad
        tokens, mask, mask_labels, pad_mask = self.create_masked_lm_predictions(
            tokensa, tokensb, self.mask_lm_prob, self.max_preds_per_seq,
            self.vocab_words, rng)
        sample = {
            'text': np.array(tokens[0]),
            'types': np.array(tokens[1]),
            'is_random': int(is_random_next),
            'mask': np.array(mask),
            'mask_labels': np.array(mask_labels),
            'pad_mask': np.array(pad_mask)
        }
        return sample

    def sentence_split(self, document):
        """split document into sentences"""
        lines = document.split('\n')
        if self.presplit_sentences:
            return [line for line in lines if line]
        rtn = []
        for line in lines:
            if line != '':
                rtn.extend(tokenize.sent_tokenize(line))
        return rtn

    def sentence_tokenize(self,
                          sent,
                          sentence_num=0,
                          beginning=False,
                          ending=False):
        """tokenize sentence and get token types"""
        tokens = self.tokenizer.EncodeAsIds(sent).tokenization
        str_type = 'str' + str(sentence_num)
        token_types = [self.tokenizer.get_type(str_type).Id] * len(tokens)
        return tokens, token_types

    def get_doc(self, idx):
        """gets text of document corresponding to idx"""
        rtn = self.ds[idx]
        if isinstance(rtn, dict):
            rtn = rtn['text']
        return rtn

    def create_random_sentencepair(self, target_seq_length, rng, np_rng):
        """
        fetches a random sentencepair corresponding to rng state similar to
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L248-L294
        """
        is_random_next = None

        curr_strs = []
        curr_str_types = []
        curr_len = 0

        while curr_len < 1:
            curr_len = 0
            doc_a = None
            while doc_a is None:
                if self.weighted:
                    # doc_a_idx = np_rng.choice(self.ds_len, p=self.weighting)
                    doc_a_idx = self.get_weighted_samples(np_rng)
                else:
                    doc_a_idx = rng.randint(0, self.ds_len - 1)
                doc_a = self.sentence_split(self.get_doc(doc_a_idx))
                if not doc_a:
                    doc_a = None

            random_start_a = rng.randint(0, len(doc_a) - 1)
            while random_start_a < len(doc_a):
                sentence = doc_a[random_start_a]
                sentence, sentence_types = self.sentence_tokenize(
                    sentence, 0, random_start_a == 0,
                    random_start_a == len(doc_a))
                curr_strs.append(sentence)
                curr_str_types.append(sentence_types)
                curr_len += len(sentence)
                if random_start_a == len(
                        doc_a) - 1 or curr_len >= target_seq_length:
                    break
                random_start_a = (random_start_a + 1)

        if curr_strs:
            num_a = 1
            if len(curr_strs) >= 2:
                num_a = rng.randint(0, len(curr_strs))

            tokens_a = []
            token_types_a = []
            for j in range(num_a):
                tokens_a.extend(curr_strs[j])
                token_types_a.extend(curr_str_types[j])

            tokens_b = []
            token_types_b = []
            is_random_next = False
            if len(curr_strs) == 1 or rng.random() < 0.5:
                is_random_next = True
                target_b_length = target_seq_length - len(tokens_a)
                b_len = 0
                while b_len < 1:
                    doc_b = None
                    while doc_b is None:
                        doc_b_idx = rng.randint(0, self.ds_len - 2)
                        doc_b_idx += int(doc_b_idx >= doc_a_idx)

                        doc_b = self.sentence_split(self.get_doc(doc_b_idx))
                        if not doc_b:
                            doc_b = None

                    random_start_b = rng.randint(0, len(doc_b) - 1)
                    while random_start_b < len(doc_b):
                        sentence_b = doc_b[random_start_b]
                        new_b_tokens, new_b_types = self.sentence_tokenize(
                            sentence_b, 1, random_start_b == 0,
                            random_start_b == len(doc_b))
                        b_len += len(new_b_tokens)
                        tokens_b.extend(new_b_tokens)
                        token_types_b.extend(new_b_types)
                        if len(tokens_b) >= target_b_length:
                            break
                        random_start_b = (random_start_b + 1)
            else:
                is_random_next = False
                for j in range(num_a, len(curr_strs)):
                    tokens_b.extend(curr_strs[j])
                    token_types_b.extend(curr_str_types[j])

        return (tokens_a, token_types_a), (tokens_b,
                                           token_types_b), is_random_next

    def truncate_seq_pair(self, a, b, max_seq_len, rng):
        """
        Truncate sequence pair according to original BERT implementation:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L391
        """
        tokens_a, token_types_a = a
        tokens_b, token_types_b = b
        max_num_tokens = max_seq_len - 3
        while True:
            len_a = len(tokens_a)
            len_b = len(tokens_b)
            total_length = len_a + len_b
            if total_length <= max_num_tokens:
                break
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                trunc_types = token_types_a
            else:
                trunc_tokens = tokens_b
                trunc_types = token_types_b

            assert len(trunc_tokens) >= 1

            if rng.random() < 0.5:
                trunc_tokens.pop(0)
                trunc_types.pop(0)
            else:
                trunc_tokens.pop()
                trunc_types.pop()
        return (tokens_a, token_types_a), (tokens_b, token_types_b)

    def mask_token(self, idx, tokens, types, vocab_words, rng):
        """
        helper function to mask `idx` token from `tokens` according to
        section 3.3.1 of https://arxiv.org/pdf/1810.04805.pdf
        """
        label = tokens[idx]
        if rng.random() < 0.8:
            new_label = self.tokenizer.get_command('MASK').Id
        else:
            if rng.random() < 0.5:
                new_label = label
            else:
                new_label = rng.choice(vocab_words)

        tokens[idx] = new_label

        return label

    def pad_seq(self, seq):
        """helper function to pad sequence pair"""
        num_pad = max(0, self.max_seq_len - len(seq))
        pad_mask = [0] * len(seq) + [1] * num_pad
        seq += [self.tokenizer.get_command('pad').Id] * num_pad
        return seq, pad_mask

    def create_masked_lm_predictions(self, a, b, mask_lm_prob,
                                     max_preds_per_seq, vocab_words, rng):
        """
        Mask sequence pair for BERT training according to:
        https://github.com/google-research/bert/blob/master/create_pretraining_data.py#L338
        """
        tokens_a, token_types_a = a
        tokens_b, token_types_b = b
        tokens = [self.tokenizer.get_command('ENC').Id] + tokens_a + [
            self.tokenizer.get_command('sep').Id
        ] + tokens_b + [self.tokenizer.get_command('sep').Id]
        token_types = [token_types_a[0]] + token_types_a + [
            token_types_a[0]
        ] + token_types_b + [token_types_b[0]]

        len_a = len(tokens_a)
        len_b = len(tokens_b)

        cand_indices = [idx + 1 for idx in range(len_a)
                        ] + [idx + 2 + len_a for idx in range(len_b)]

        rng.shuffle(cand_indices)

        output_tokens, pad_mask = self.pad_seq(list(tokens))
        output_types, _ = self.pad_seq(list(token_types))

        num_to_predict = min(max_preds_per_seq,
                             max(1, int(round(len(tokens) * mask_lm_prob))))

        mask = [0] * len(output_tokens)
        mask_labels = [-1] * len(output_tokens)

        for idx in sorted(cand_indices[:num_to_predict]):
            mask[idx] = 1
            label = self.mask_token(idx, output_tokens, output_types,
                                    vocab_words, rng)
            mask_labels[idx] = label

        return (output_tokens, output_types), mask, mask_labels, pad_mask
