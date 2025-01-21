# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright © Alibaba, Inc. and its affiliates.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self,
                 batch_sampler,
                 num_iterations,
                 start_iter=0,
                 enable_mosaic=False):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        self.enable_mosaic = enable_mosaic

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, 'set_epoch'):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield [(self.enable_mosaic, idx) for idx in batch]

    def __len__(self):
        return self.num_iterations

    def set_mosaic(self, enable_mosaic):
        self.enable_mosaic = enable_mosaic
