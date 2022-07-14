# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
from abc import ABCMeta, abstractmethod

import numpy as np
import torch

from modelscope.trainers.hooks.hook import Hook
from ..priority import Priority


class LoggerHook(Hook):
    """Base class for logger hooks.

    Args:
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        by_epoch (bool): Whether EpochBasedtrainer is used.
    """

    __metaclass__ = ABCMeta
    PRIORITY = Priority.VERY_LOW

    def __init__(self,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True):
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag
        self.by_epoch = by_epoch

    @abstractmethod
    def log(self, trainer):
        pass

    @staticmethod
    def is_scalar(val, include_np=True, include_torch=True):
        """Tell the input variable is a scalar or not.

        Args:
            val: Input variable.
            include_np (bool): Whether to treat 0-d np.ndarray as a scalar.
            include_torch (bool): Whether to treat 0-d torch.Tensor as a scalar.

        Returns:
            bool: True or False.
        """
        if isinstance(val, numbers.Number):
            return True
        elif include_np and isinstance(val, np.ndarray) and val.ndim == 0:
            return True
        elif include_torch and isinstance(val, torch.Tensor) and len(val) == 1:
            return True
        else:
            return False

    def get_epoch(self, trainer):
        if trainer.mode == 'train':
            epoch = trainer.epoch + 1
        elif trainer.mode == 'val':
            # normal val mode
            # trainer.epoch += 1 has been done before val workflow
            epoch = trainer.epoch
        else:
            raise ValueError(f"trainer mode should be 'train' or 'val', "
                             f'but got {trainer.mode}')
        return epoch

    def get_iter(self, trainer, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = trainer.inner_iter + 1
        else:
            current_iter = trainer.iter + 1
        return current_iter

    def before_run(self, trainer):
        for hook in trainer.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def before_epoch(self, trainer):
        trainer.log_buffer.clear()  # clear logs of last epoch

    def after_train_iter(self, trainer):
        if self.by_epoch and self.every_n_epochs(trainer, self.interval):
            trainer.log_buffer.average(self.interval)
        elif not self.by_epoch and self.every_n_iters(trainer, self.interval):
            trainer.log_buffer.average(self.interval)
        elif self.end_of_epoch(trainer) and not self.ignore_last:
            # not precise but more stable
            trainer.log_buffer.average(self.interval)

        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_train_epoch(self, trainer):
        if trainer.log_buffer.ready:
            self.log(trainer)
            if self.reset_flag:
                trainer.log_buffer.clear_output()

    def after_val_epoch(self, trainer):
        trainer.log_buffer.average()
        self.log(trainer)
        if self.reset_flag:
            trainer.log_buffer.clear_output()
