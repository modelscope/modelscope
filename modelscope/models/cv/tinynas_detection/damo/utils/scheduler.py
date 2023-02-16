# Copyright (c) Alibaba, Inc. and its affiliates.
import math


class cosine_scheduler:

    def __init__(self,
                 base_lr_per_img,
                 batch_size,
                 min_lr_ratio,
                 total_iters,
                 no_aug_iters,
                 warmup_iters,
                 warmup_start_lr=0):

        self.base_lr = base_lr_per_img * batch_size
        self.final_lr = self.base_lr * min_lr_ratio
        self.warmup_iters = warmup_iters
        self.warmup_start_lr = warmup_start_lr
        self.total_iters = total_iters
        self.no_aug_iters = no_aug_iters

    def get_lr(self, iters):

        if iters < self.warmup_iters:
            lr = (self.base_lr - self.warmup_start_lr) * pow(
                iters / float(self.warmup_iters), 2) + self.warmup_start_lr
        elif iters >= self.total_iters - self.no_aug_iters:
            lr = self.final_lr
        else:
            lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) \
                * (1.0 + math.cos(math.pi * (iters - self.warmup_iters)
                   / (self.total_iters - self.warmup_iters - self.no_aug_iters)))
        return lr
