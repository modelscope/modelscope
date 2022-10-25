# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os

import numpy as np

from modelscope.preprocessors.nlp.space.args import str2bool
from modelscope.preprocessors.nlp.space.batch import batch
from modelscope.preprocessors.nlp.space.lazy_dataset import LazyDataset
from modelscope.preprocessors.nlp.space.sampler import (RandomSampler,
                                                        SequentialSampler,
                                                        SortedSampler)


def get_data_loader(batch_size, reader, hparams, file, collate_fn, is_test):
    assert os.path.exists(file), f"{file} doesn't exist"
    dataset = LazyDataset(file, reader=reader)
    data_loader = DataLoader(
        dataset,
        batch_size,
        hparams.Trainer,
        collate_fn=collate_fn,
        is_test=is_test)
    return data_loader


def get_sequential_data_loader(batch_size, reader, hparams, data_paths,
                               collate_fn, data_type):
    data_loaders = []
    for data_path in data_paths:
        file = os.path.join(
            data_path,
            f'{data_type}.{hparams.BPETextField.tokenizer_type}.jsonl')
        data_loaders.append(
            get_data_loader(
                batch_size=batch_size,
                reader=reader,
                hparams=hparams,
                file=file,
                collate_fn=collate_fn,
                is_test=(data_type != 'train')))
    data_loader = SequentialDataLoaderWrapper(data_loaders)
    return data_loader


class DataLoader(object):
    """ Implement of DataLoader. """

    @classmethod
    def add_cmdline_argument(cls, group):
        group.add_argument('--shuffle', type=str2bool, default=True)
        group.add_argument('--sort_pool_size', type=int, default=0)
        return group

    def __init__(self,
                 dataset,
                 batch_size,
                 hparams,
                 collate_fn=None,
                 sampler=None,
                 is_test=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.gpu = hparams.gpu
        self.sort_pool_size = hparams.sort_pool_size

        if sampler is None:
            if hparams.shuffle and not is_test:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if self.sort_pool_size > 0 and not is_test:
            sampler = SortedSampler(sampler, self.sort_pool_size)

        def reader():
            for idx in sampler:
                yield idx

        drop_last = False if self.gpu <= 1 or is_test else True
        self.reader = batch(reader, batch_size=batch_size, drop_last=drop_last)
        self.num_batches = math.floor(len(dataset) / batch_size) if drop_last \
            else math.ceil(len(dataset) / batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for batch_indices in self.reader():
            samples = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(samples)


class SequentialDataLoaderWrapper:

    def __init__(self, data_loaders):
        self.data_loaders = data_loaders
        self.data_file_to_dataset = {
            data_loader.dataset.data_file: data_loader.dataset
            for data_loader in self.data_loaders
        }

    def __iter__(self):
        for data_loader in self.data_loaders:
            for tmp_batch in data_loader:
                yield data_loader.dataset.data_file, tmp_batch

    def __len__(self):
        return np.sum([len(data_loader) for data_loader in self.data_loaders])
