# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.metainfo import LR_Schedulers
from modelscope.trainers.lrscheduler.builder import LR_SCHEDULER
from .base import BaseWarmup


@LR_SCHEDULER.register_module(module_name=LR_Schedulers.ConstantWarmup)
class ConstantWarmup(BaseWarmup):
    """Linear warmup scheduler.

    Args:
        base_scheduler (torch.optim._LRScheduler): an instance of torch.optim._LRScheduler type
        warmup_ratio (float): Lr used at warmup stage equals to warmup_ratio * initial_lr
        warmup_iters (int | list): Warmup iterations
        last_epoch (int): The index of last epoch.
    """

    def __init__(self,
                 base_scheduler,
                 warmup_iters,
                 warmup_ratio=0.1,
                 last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        super(ConstantWarmup, self).__init__(
            base_scheduler, warmup_iters=warmup_iters, last_epoch=last_epoch)

    def get_warmup_scale(self, cur_iter):
        if cur_iter >= self.warmup_iters:
            return 1.0
        return self.warmup_ratio


@LR_SCHEDULER.register_module(module_name=LR_Schedulers.LinearWarmup)
class LinearWarmup(BaseWarmup):
    """Linear warmup scheduler.

    Args:
        base_scheduler (torch.optim._LRScheduler): an instance of torch.optim._LRScheduler type
        warmup_iters (int | list): Warmup iterations
        warmup_ratio (float): Lr used at the beginning of warmup equals to warmup_ratio * initial_lr
        last_epoch (int): The index of last epoch.
    """

    def __init__(self,
                 base_scheduler,
                 warmup_iters,
                 warmup_ratio=0.1,
                 last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        super(LinearWarmup, self).__init__(
            base_scheduler, warmup_iters=warmup_iters, last_epoch=last_epoch)

    def get_warmup_scale(self, cur_iter):
        k = (1 - cur_iter / self.warmup_iters) * (1 - self.warmup_ratio)
        return 1 - k


@LR_SCHEDULER.register_module(module_name=LR_Schedulers.ExponentialWarmup)
class ExponentialWarmup(BaseWarmup):
    """Exponential warmup scheduler.

    Args:
        base_scheduler (torch.optim._LRScheduler): an instance of torch.optim._LRScheduler type
        warmup_iters (int | list): Warmup iterations
        warmup_ratio (float): Lr used at the beginning of warmup equals to warmup_ratio * initial_lr
        last_epoch (int): The index of last epoch.
    """

    def __init__(self,
                 base_scheduler,
                 warmup_iters,
                 warmup_ratio=0.1,
                 last_epoch=-1):
        self.warmup_ratio = warmup_ratio
        super(ExponentialWarmup, self).__init__(
            base_scheduler, warmup_iters=warmup_iters, last_epoch=last_epoch)

    def get_warmup_scale(self, cur_iter):
        k = self.warmup_ratio**(1 - cur_iter / self.warmup_iters)
        return k
