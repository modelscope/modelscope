# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.constant import LogKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import get_dist_info
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
class EvaluationHook(Hook):
    """Evaluation hook.
    Args:
        interval (int): Evaluation interval.
        by_epoch (bool): Evaluate by epoch or by iteration.
        start_idx (int | None, optional): The epoch/iterations validation begins.
            Default: None, validate every interval epochs/iterations from scratch.
        save_best_ckpt (bool): Whether save the best checkpoint during evaluation.
        monitor_key (str): Monitor key to compare rule for best score, only valid when `save_best_ckpt` is true.
        rule (str): Comparison rule for best score, only valid when `save_best_ckpt` is true.
            Support "max" and "min". If rule is "max", the checkpoint at the maximum `monitor_key`
            will be saved, If rule is "min", the checkpoint at the minimum `monitor_key` will be saved.
        out_dir (str): Output directory to save best checkpoint.
    """

    PRIORITY = Priority.NORMAL
    rule_map = {'max': lambda x, y: x > y, 'min': lambda x, y: x < y}

    def __init__(self,
                 interval=1,
                 by_epoch=True,
                 start_idx=None,
                 save_best_ckpt=False,
                 monitor_key=None,
                 rule='max',
                 out_dir=None):
        assert interval > 0, 'interval must be a positive number'
        if save_best_ckpt:
            assert monitor_key is not None, 'Must provide `monitor_key` when `save_best_ckpt` is True.'
            assert rule in ['max',
                            'min'], 'Only support "max" or "min" rule now.'

        self.interval = interval
        self.start_idx = start_idx
        self.by_epoch = by_epoch
        self.save_best_ckpt = save_best_ckpt
        self.monitor_key = monitor_key
        self.rule = rule
        self.out_dir = out_dir
        self._best_metric = None
        self._best_ckpt_file = None

    def before_run(self, trainer):
        if not self.out_dir:
            self.out_dir = trainer.work_dir
        if not os.path.exists(self.out_dir):
            rank, _ = get_dist_info()
            if rank == 0:
                os.makedirs(self.out_dir)

        if self.save_best_ckpt:
            if not hasattr(trainer, 'logger'):
                self.logger = get_logger(__name__)
            else:
                self.logger = trainer.logger
            self.logger.info(
                f'Best checkpoint will be saved to {self.out_dir}')

    def after_train_iter(self, trainer):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(trainer):
            self.do_evaluate(trainer)

    def after_train_epoch(self, trainer):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch and self._should_evaluate(trainer):
            self.do_evaluate(trainer)

    def do_evaluate(self, trainer):
        """Evaluate the results."""
        eval_res = trainer.evaluate()
        for name, val in eval_res.items():
            trainer.log_buffer.output[name] = val

        trainer.log_buffer.ready = True

        if self.save_best_ckpt and self._is_best_metric(eval_res):
            # remove the previous best model and save the latest best model
            if self._best_ckpt_file is not None and os.path.exists(
                    self._best_ckpt_file):
                os.remove(self._best_ckpt_file)
            self._save_checkpoint(trainer)

    def _is_best_metric(self, eval_res):
        if self.monitor_key not in eval_res:
            raise ValueError(
                f'Not find monitor_key: {self.monitor_key} in {eval_res}')

        if self._best_metric is None:
            self._best_metric = eval_res[self.monitor_key]
            return True
        else:
            compare_fn = self.rule_map[self.rule]
            if compare_fn(eval_res[self.monitor_key], self._best_metric):
                self._best_metric = eval_res[self.monitor_key]
                return True
        return False

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(
                self.out_dir,
                f'best_{LogKeys.EPOCH}{trainer.epoch + 1}_{self.monitor_key}{self._best_metric}.pth'
            )
        else:
            cur_save_name = os.path.join(
                self.out_dir,
                f'best_{LogKeys.ITER}{trainer.iter + 1}_{self.monitor_key}{self._best_metric}.pth'
            )

        rank, _ = get_dist_info()
        if rank == 0:
            save_checkpoint(trainer.model, cur_save_name, trainer.optimizer)

        self._best_ckpt_file = cur_save_name

    def _should_evaluate(self, trainer):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the ``start_idx`` is larger than
           current epochs/iters.
        3. It will not perform evaluation when current epochs/iters is larger than
           the ``start_idx`` but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = trainer.epoch
            check_time = self.every_n_epochs
        else:
            current = trainer.iter
            check_time = self.every_n_iters

        if self.start_idx is None:
            if not check_time(trainer, self.interval):
                return False
        elif (current + 1) < self.start_idx:
            return False
        else:
            if (current + 1 - self.start_idx) % self.interval:
                return False
        return True
