# ------------------------------------------------------------------------------
# Part of implementation is adopted from DBNet,
# made publicly available under the Apache License 2.0 at https://github.com/MhLiao/DB.
# ------------------------------------------------------------------------------
import bisect
import math

import imgaug
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, ConcatDataset, Sampler

from .processes import ICDARCollectFN


def default_worker_init_fn(worker_id):
    np.random.seed(worker_id)
    imgaug.seed(worker_id)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self,
                 dataset,
                 cfg_dataloader,
                 is_train,
                 distributed,
                 drop_last=False,
                 shuffle=None):
        self.dataset = dataset
        self.batch_size = cfg_dataloader.batch_size
        self.num_workers = cfg_dataloader.num_workers
        self.num_gpus = cfg_dataloader.num_gpus
        self.is_train = is_train
        self.drop_last = drop_last
        self.shuffle = shuffle

        if hasattr(cfg_dataloader, 'collect_fn'
                   ) and cfg_dataloader.collect_fn == 'ICDARCollectFN':
            self.collect_fn = ICDARCollectFN()
        else:
            self.collect_fn = torch.utils.data.dataloader.default_collate
        if self.shuffle is None:
            self.shuffle = self.is_train

        if distributed:
            sampler = DistributedSampler(
                self.dataset, shuffle=self.shuffle, num_replicas=self.num_gpus)
            batch_sampler = BatchSampler(sampler,
                                         self.batch_size // self.num_gpus,
                                         False)
            torch.utils.data.DataLoader.__init__(
                self,
                self.dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=self.drop_last,
                collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        else:
            torch.utils.data.DataLoader.__init__(
                self,
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                shuffle=self.shuffle,
                pin_memory=True,
                collate_fn=self.collect_fn,
                worker_init_fn=default_worker_init_fn)
        self.collect_fn = str(self.collect_fn)


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
