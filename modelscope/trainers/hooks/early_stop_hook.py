# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np

from modelscope.metainfo import Hooks
from modelscope.utils.logger import get_logger
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


class EarlyStopStrategy:
    by_epoch = 'by_epoch'
    by_step = 'by_step'
    no = 'no'


@HOOKS.register_module(module_name=Hooks.EarlyStopHook)
class EarlyStopHook(Hook):
    """Early stop when a specific metric stops improving.

    Args:
        metric_key (str):  Metric key to be monitored.
        rule (str): Comparison rule for best score. Support "max" and "min".
            If rule is "max", the training will stop when `metric_key` has stopped increasing.
            If rule is "min", the training will stop when `metric_key` has stopped decreasing.
        patience (int): Trainer will stop if the monitored metric did not improve for the last `patience` times.
        min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
        check_finite (bool): If true, stops training when the metric becomes NaN or infinite.
        early_stop_strategy (str): The strategy to early stop, can be by_epoch/by_step/none
        interval (int): The frequency to trigger early stop check, by epoch or step.
    """

    PRIORITY = Priority.VERY_LOW
    rule_map = {'max': lambda x, y: x > y, 'min': lambda x, y: x < y}

    def __init__(self,
                 metric_key: str,
                 rule: str = 'max',
                 patience: int = 3,
                 min_delta: float = 0.0,
                 check_finite: bool = True,
                 early_stop_strategy: str = EarlyStopStrategy.by_epoch,
                 interval: int = 1,
                 **kwargs):
        self.metric_key = metric_key
        self.rule = rule
        self.patience = patience
        self.min_delta = min_delta
        self.check_finite = check_finite
        if 'by_epoch' in kwargs:
            self.early_stop_strategy = EarlyStopStrategy.by_epoch if kwargs[
                'by_epoch'] else EarlyStopStrategy.by_step
        else:
            self.early_stop_strategy = early_stop_strategy
        self.interval = interval

        self.wait_count = 0
        self.best_score = float('inf') if rule == 'min' else -float('inf')

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger

    def _should_stop(self, trainer):
        metric_values = trainer.metric_values

        if metric_values is None:
            return False

        if self.metric_key not in metric_values:
            raise ValueError(
                f'Metric not found: {self.metric_key} not in {metric_values}')

        should_stop = False
        current_score = metric_values[self.metric_key]
        if self.check_finite and not np.isfinite(current_score):
            should_stop = True
            self.logger.warning(
                f'Metric {self.metric_key} = {current_score} is not finite. '
                f'Previous best metric: {self.best_score:.4f}.')
        elif self.rule_map[self.rule](current_score - self.min_delta,
                                      self.best_score):
            self.best_score = current_score
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                self.logger.info(
                    f'Metric {self.metric_key} did not improve in the last {self.wait_count} epochs or iterations. '
                    f'Best score: {self.best_score:.4f}.')
        return should_stop

    def _stop_training(self, trainer):
        self.logger.info('Early Stopping!')
        trainer._stop_training = True

    def after_train_epoch(self, trainer):
        if self.early_stop_strategy != EarlyStopStrategy.by_epoch:
            return

        if not self.every_n_epochs(trainer, self.interval):
            return

        if self._should_stop(trainer):
            self._stop_training(trainer)

    def after_train_iter(self, trainer):
        if self.early_stop_strategy != EarlyStopStrategy.by_step:
            return

        if not self.every_n_iters(trainer, self.interval):
            return

        if self._should_stop(trainer):
            self._stop_training(trainer)
