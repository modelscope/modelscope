# Copyright (c) Alibaba, Inc. and its affiliates.
from torch.optim.lr_scheduler import _LRScheduler


class BaseWarmup(_LRScheduler):
    """Base warmup scheduler

    Args:
        base_scheduler (torch.optim._LRScheduler): an instance of torch.optim._LRScheduler type
        warmup_iters (int | list): Warmup iterations
        last_epoch (int): The index of last epoch.
    """

    def __init__(self,
                 base_scheduler,
                 warmup_iters,
                 last_epoch=-1,
                 verbose=False):
        self.base_scheduler = base_scheduler
        self.warmup_iters = warmup_iters
        optimizer = self.base_scheduler.optimizer
        self._is_init_step = True

        super(BaseWarmup, self).__init__(
            optimizer, last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):
        return self.base_scheduler.get_lr()

    def state_dict(self):
        return self.base_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        return self.base_scheduler.load_state_dict(state_dict)

    def scale(self):
        """Scale the learning rates.
        """
        scale_value = self.get_warmup_scale(self.base_scheduler._step_count
                                            - 1)
        if isinstance(scale_value, (int, float)):
            scale_value = [
                scale_value for _ in range(len(self.optimizer.param_groups))
            ]
        else:
            assert isinstance(
                scale_value, (list, tuple)), 'Only support list or tuple type!'
            assert len(scale_value) == len(
                self.optimizer.param_groups), ('Size mismatch {} != {}'.format(
                    len(scale_value), len(self.optimizer.param_groups)))

        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] *= scale_value[i]

    def step(self, *args, **kwargs):
        """
        When ``self.base_scheduler._step_count`` is less than ``self.warmup_iters``, multiply lr by scale
        """
        if self.base_scheduler._step_count > self.warmup_iters:
            return self.base_scheduler.step(*args, **kwargs)

        for group, lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = lr

        # `base_scheduler` has done step() at init when build
        if self._is_init_step:
            self._is_init_step = False
        else:
            self.base_scheduler.step(*args, **kwargs)

        self.scale()

    @classmethod
    def get_warmup_scale(self, cur_iter):
        pass
