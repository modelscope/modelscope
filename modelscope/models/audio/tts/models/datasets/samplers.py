# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import random

import torch
from torch import distributed as dist
from torch.utils.data import Sampler


class LenSortGroupPoolSampler(Sampler):

    def __init__(self, data_source, length_lst, group_size):
        super(LenSortGroupPoolSampler, self).__init__(data_source)

        self.data_source = data_source
        self.length_lst = length_lst
        self.group_size = group_size

        self.num = len(self.length_lst)
        self.buckets = self.num // group_size

    def __iter__(self):

        def getkey(item):
            return item[1]

        random_lst = torch.randperm(self.num).tolist()
        random_len_lst = [(i, self.length_lst[i]) for i in random_lst]

        # Bucket examples based on similar output sequence length for efficiency:
        groups = [
            random_len_lst[i:i + self.group_size]
            for i in range(0, self.num, self.group_size)
        ]
        if (self.num % self.group_size):
            groups.append(random_len_lst[self.buckets * self.group_size:-1])

        indices = []

        for group in groups:
            group.sort(key=getkey, reverse=True)
            for item in group:
                indices.append(item[0])

        return iter(indices)

    def __len__(self):
        return len(self.data_source)


class DistributedLenSortGroupPoolSampler(Sampler):

    def __init__(self,
                 dataset,
                 length_lst,
                 group_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=True):
        super(DistributedLenSortGroupPoolSampler, self).__init__(dataset)

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'modelscope error: Requires distributed package to be available'
                )
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'modelscope error: Requires distributed package to be available'
                )
            rank = dist.get_rank()
        self.dataset = dataset
        self.length_lst = length_lst
        self.group_size = group_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.buckets = self.num_samples // group_size
        self.shuffle = shuffle

    def __iter__(self):

        def getkey(item):
            return item[1]

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        random_len_lst = [(i, self.length_lst[i]) for i in indices]

        # Bucket examples based on similar output sequence length for efficiency:
        groups = [
            random_len_lst[i:i + self.group_size]
            for i in range(0, self.num_samples, self.group_size)
        ]
        if (self.num_samples % self.group_size):
            groups.append(random_len_lst[self.buckets * self.group_size:-1])

        new_indices = []

        for group in groups:
            group.sort(key=getkey, reverse=True)
            for item in group:
                new_indices.append(item[0])

        return iter(new_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
