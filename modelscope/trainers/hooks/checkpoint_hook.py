# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope import __version__
from modelscope.metainfo import Hooks
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import LogKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module(module_name=Hooks.CheckpointHook)
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The frequency to save model. If `by_epoch=True`,
            it means the number of epochs, else means the number of iterations
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
        save_optimizer (bool): Whether to save optimizer state dict.  Default: True.
        save_dir (str): The directory to save checkpoints. If is None, use `trainer.work_dir`
        save_last (bool): Whether to save the last checkpoint. Default: True.
    """

    PRIORITY = Priority.NORMAL

    def __init__(self,
                 interval=0,
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None,
                 save_last=True):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.save_dir = save_dir
        self.save_last = save_last

    def before_run(self, trainer):
        if not self.save_dir:
            self.save_dir = trainer.work_dir

        if not os.path.exists(self.save_dir) and is_master():
            os.makedirs(self.save_dir)

        if not hasattr(trainer, 'logger'):
            self.logger = get_logger(__name__)
        else:
            self.logger = trainer.logger

        if is_master():
            self.logger.info(f'Checkpoints will be saved to {self.save_dir}')

    def after_train_epoch(self, trainer):
        if not self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master():
                self.logger.info(
                    f'Saving checkpoint at {trainer.epoch + 1} epoch')
                self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.EPOCH}_{trainer.epoch + 1}.pth')
        else:
            cur_save_name = os.path.join(
                self.save_dir, f'{LogKeys.ITER}_{trainer.iter + 1}.pth')

        save_checkpoint(trainer.model, cur_save_name, trainer.optimizer)

    def after_train_iter(self, trainer):
        if self.by_epoch:
            return

        if self._should_save(trainer):
            if is_master():
                self.logger.info(
                    f'Saving checkpoint at {trainer.iter + 1} iterations')
                self._save_checkpoint(trainer)

    def _should_save(self, trainer):
        if self.by_epoch:
            check_last = self.is_last_epoch
            check_frequency = self.every_n_epochs
        else:
            check_last = self.is_last_iter
            check_frequency = self.every_n_iters

        if check_frequency(trainer,
                           self.interval) or (self.save_last
                                              and check_last(trainer)):
            return True
        return False


@HOOKS.register_module(module_name=Hooks.BestCkptSaverHook)
class BestCkptSaverHook(CheckpointHook):
    """Save best checkpoints hook.
    Args:
        metric_key (str): Metric key to compare rule for best score.
        rule (str): Comparison rule for best score.
            Support "max" and "min". If rule is "max", the checkpoint at the maximum `metric_key`
            will be saved, If rule is "min", the checkpoint at the minimum `metric_key` will be saved.
        by_epoch (bool): Save best checkpoints by epoch or by iteration.
        save_optimizer (bool): Whether to save optimizer state dict.  Default: True.
        save_dir (str): Output directory to save best checkpoint.
    """

    PRIORITY = Priority.NORMAL
    rule_map = {'max': lambda x, y: x > y, 'min': lambda x, y: x < y}

    def __init__(self,
                 metric_key,
                 rule='max',
                 by_epoch=True,
                 save_optimizer=True,
                 save_dir=None):
        assert rule in ['max', 'min'], 'Only support "max" or "min" rule now.'
        super().__init__(
            by_epoch=by_epoch,
            save_optimizer=save_optimizer,
            save_dir=save_dir,
        )
        self.metric_key = metric_key
        self.rule = rule
        self._best_metric = None
        self._best_ckpt_file = None

    def _should_save(self, trainer):
        return self._is_best_metric(trainer.metric_values)

    def _is_best_metric(self, metric_values):
        if metric_values is None:
            return False

        if self.metric_key not in metric_values:
            raise ValueError(
                f'Not find metric_key: {self.metric_key} in {metric_values}')

        if self._best_metric is None:
            self._best_metric = metric_values[self.metric_key]
            return True
        else:
            compare_fn = self.rule_map[self.rule]
            if compare_fn(metric_values[self.metric_key], self._best_metric):
                self._best_metric = metric_values[self.metric_key]
                return True
        return False

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.save_dir,
                f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.metric_key}{self._best_metric}.pth'
            )
        else:
            cur_save_name = os.path.join(
                self.save_dir,
                f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.metric_key}{self._best_metric}.pth'
            )
        save_checkpoint(trainer.model, cur_save_name, trainer.optimizer)
        self._best_ckpt_file = cur_save_name
