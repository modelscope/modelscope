# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

from modelscope.metainfo import Hooks
from .builder import HOOKS
from .hook import Hook


@HOOKS.register_module(module_name=Hooks.EvaluationHook)
class EvaluationHook(Hook):
    """
    Evaluation hook.

    Args:
        interval (int): Evaluation interval.
        by_epoch (bool): Evaluate by epoch or by iteration.
        start_idx (int or None, optional): The epoch or iterations validation begins.
            Default: None, validate every interval epochs/iterations from scratch.
    """

    def __init__(self, interval=1, by_epoch=True, start_idx=None):
        assert interval > 0, 'interval must be a positive number'
        self.interval = interval
        self.start_idx = start_idx
        self.by_epoch = by_epoch

    def after_train_iter(self, trainer):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(trainer):
            self.do_evaluate(trainer)

    def after_train_epoch(self, trainer):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch and self._should_evaluate(trainer):
            self.do_evaluate(trainer)

    def add_visualization_info(self, trainer, results):
        if trainer.visualization_buffer.output.get('eval_results',
                                                   None) is None:
            trainer.visualization_buffer.output['eval_results'] = OrderedDict()

            trainer.visualization_buffer.output['eval_results'].update(
                trainer.visualize(results))

    def do_evaluate(self, trainer):
        """Evaluate the results."""
        eval_res = trainer.evaluate()
        for name, val in eval_res.items():
            trainer.log_buffer.output['evaluation/' + name] = val

        trainer.log_buffer.ready = True

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
