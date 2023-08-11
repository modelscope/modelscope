# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp

import json
import numpy as np
from torch.utils.data.sampler import Sampler

from modelscope.models.multi_modal.videocomposer.ops.distributed import (
    get_rank, get_world_size, shared_random_seed)
from modelscope.models.multi_modal.videocomposer.ops.utils import (ceil_divide,
                                                                   read)

__all__ = ['BatchSampler', 'GroupSampler', 'ImgGroupSampler']


class BatchSampler(Sampler):
    r"""An infinite batch sampler.
    """

    def __init__(self,
                 dataset_size,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 seed=None):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_replicas = num_replicas or get_world_size()
        self.rank = rank or get_rank()
        self.shuffle = shuffle
        self.seed = seed or shared_random_seed()
        self.rng = np.random.default_rng(self.seed + self.rank)
        self.batches_per_rank = ceil_divide(
            dataset_size, self.num_replicas * self.batch_size)
        self.samples_per_rank = self.batches_per_rank * self.batch_size

        # rank indices
        indices = self.rng.permutation(
            self.samples_per_rank) if shuffle else np.arange(
                self.samples_per_rank)
        indices = indices * self.num_replicas + self.rank
        indices = indices[indices < dataset_size]
        self.indices = indices

    def __iter__(self):
        start = 0
        while True:
            batch = [
                self.indices[i % len(self.indices)]
                for i in range(start, start + self.batch_size)
            ]
            if self.shuffle and (start + self.batch_size) > len(self.indices):
                self.rng.shuffle(self.indices)
            start = (start + self.batch_size) % len(self.indices)
            yield batch


class GroupSampler(Sampler):

    def __init__(self,
                 group_file,
                 batch_size,
                 alpha=0.7,
                 update_interval=5000,
                 seed=8888):
        self.group_file = group_file
        self.group_folder = osp.join(osp.dirname(group_file), 'groups')
        self.batch_size = batch_size
        self.alpha = alpha
        self.update_interval = update_interval
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            # keep groups up-to-date
            self.update_groups()

            # collect items
            items = self.sample()
            while len(items) < self.batch_size:
                items += self.sample()

            # sample a batch
            batch = self.rng.choice(
                items,
                self.batch_size,
                replace=False if len(items) >= self.batch_size else True)
            yield [u.strip().split(',') for u in batch]

    def update_groups(self):
        if not hasattr(self, '_step'):
            self._step = 0
        if self._step % self.update_interval == 0:
            self.groups = json.loads(read(self.group_file))
        self._step += 1

    def sample(self):
        scales = np.array(
            [float(next(iter(u)).split(':')[-1]) for u in self.groups])
        p = scales**self.alpha / (scales**self.alpha).sum()
        group = self.rng.choice(self.groups, p=p)
        list_file = osp.join(self.group_folder,
                             self.rng.choice(next(iter(group.values()))))
        return read(list_file).strip().split('\n')


class ImgGroupSampler(Sampler):

    def __init__(self,
                 group_file,
                 batch_size,
                 alpha=0.7,
                 update_interval=5000,
                 seed=8888):
        self.group_file = group_file
        self.group_folder = osp.join(osp.dirname(group_file), 'groups')
        self.batch_size = batch_size
        self.alpha = alpha
        self.update_interval = update_interval
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            # keep groups up-to-date
            self.update_groups()

            # collect items
            items = self.sample()
            while len(items) < self.batch_size:
                items += self.sample()

            # sample a batch
            batch = self.rng.choice(
                items,
                self.batch_size,
                replace=False if len(items) >= self.batch_size else True)
            yield [u.strip().split(',', 1) for u in batch]

    def update_groups(self):
        if not hasattr(self, '_step'):
            self._step = 0
        if self._step % self.update_interval == 0:
            self.groups = json.loads(read(self.group_file))

        self._step += 1

    def sample(self):
        scales = np.array(
            [float(next(iter(u)).split(':')[-1]) for u in self.groups])
        p = scales**self.alpha / (scales**self.alpha).sum()
        group = self.rng.choice(self.groups, p=p)
        list_file = osp.join(self.group_folder,
                             self.rng.choice(next(iter(group.values()))))
        return read(list_file).strip().split('\n')
