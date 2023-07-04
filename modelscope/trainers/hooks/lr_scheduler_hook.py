# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.metainfo import Hooks
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from modelscope.utils.constant import LogKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import is_master
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


class LrSchedulerProcessor:

    def __init__(self):
        self.lr_strategy = None
        self.warmup_lr_scheduler = None

    def set_lr_strategy(self, lr_strategy):
        self.lr_strategy = lr_strategy

    def set_warmup_lr_scheduler(self, warmup_lr_scheduler):
        self.warmup_lr_scheduler = warmup_lr_scheduler

    def initialize_lr_scheduler(self, trainer):
        """Initialize the lr scheduler.

        This is a strategic function which can be registered by other hook's function.
        """
        pass

    def step(self, trainer):
        """Do lr scheduler's step.

        This is a strategic function which can be registered by other hook's function.
        """
        if self.warmup_lr_scheduler is not None:
            self.warmup_lr_scheduler.step()
        else:
            trainer.lr_scheduler.step()

    def get_current_lr(self, trainer):
        import torch

        if isinstance(trainer.optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in trainer.optimizer.param_groups]
        elif isinstance(trainer.optimizer, dict):
            lr = dict()
            for name, optim in trainer.optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr


class LrStrategy:
    by_epoch = 'by_epoch'
    by_step = 'by_step'
    no = 'no'


@HOOKS.register_module(module_name=Hooks.LrSchedulerHook)
class LrSchedulerHook(Hook):
    """Lr scheduler.

    Args:
        by_epoch (bool): Whether lr changes by epoch
        warmup (dict): warm up config
    """
    PRIORITY = Priority.LOW

    def __init__(self,
                 lr_strategy=LrStrategy.by_epoch,
                 warmup=None,
                 **kwargs) -> None:
        super().__init__()
        if 'by_epoch' in kwargs:
            self.lr_strategy = LrStrategy.by_epoch if kwargs[
                'by_epoch'] else LrStrategy.by_step
        else:
            self.lr_strategy = lr_strategy
        self.warmup = warmup
        self.warmup_lr_scheduler = None
        self.processor = LrSchedulerProcessor()

    def set_processor(self, processor):
        self.processor = processor

    def before_run(self, trainer):
        self.processor.set_lr_strategy(self.lr_strategy)
        if self.warmup is not None:
            assert isinstance(self.warmup, dict) and 'type' in self.warmup
            self.warmup_lr_scheduler = build_lr_scheduler(
                cfg=self.warmup,
                default_args={'base_scheduler': trainer.lr_scheduler})
            self.processor.set_warmup_lr_scheduler(self.warmup_lr_scheduler)

        self.processor.initialize_lr_scheduler(trainer)

    def after_train_iter(self, trainer):
        if self.lr_strategy == LrStrategy.by_step and trainer.iter >= getattr(
                trainer, 'cumulative_iters', 1) - 1:
            self.processor.step(trainer)
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def before_train_epoch(self, trainer):
        trainer.log_buffer.output[LogKeys.LR] = self._get_log_lr(trainer)

    def after_train_epoch(self, trainer):
        if self.lr_strategy == LrStrategy.by_epoch:
            self.processor.step(trainer)

    def _get_log_lr(self, trainer):
        # forward compatibility with AddLrLogHook in EasyCV
        if not hasattr(self, 'processor'):
            self.processor = LrSchedulerProcessor()
        cur_lr = self.processor.get_current_lr(trainer)
        # only record lr of the first param group
        if isinstance(cur_lr, list):
            lr = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            lr = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                lr.update({k: lr_[0]})

        return lr


class PlateauLrSchedulerProcessor(LrSchedulerProcessor):

    def __init__(self, metric_key):
        super().__init__()
        self.metric_key = metric_key

    def step(self, trainer):
        # adapt to evaluation interval is greater than 1
        if trainer.metric_values is None:
            if is_master():
                print(
                    f'Current epoch {trainer.epoch} has no evaluation metric values, skip lr_scheduler.step() !'
                )
            return

        metrics = trainer.metric_values[self.metric_key]
        if self.lr_strategy == LrStrategy.by_epoch:
            if self.warmup_lr_scheduler is not None:
                self.warmup_lr_scheduler.step(metrics=metrics)
            else:
                trainer.lr_scheduler.step(metrics=metrics)


@HOOKS.register_module(module_name=Hooks.PlateauLrSchedulerHook)
class PlateauLrSchedulerHook(Hook):
    """Lr scheduler hook for `ReduceLROnPlateau`.

    Args:
        metric_key (str): Metric key returned from `trainer.metric_values`,
            get the value of metric key and pass it to `ReduceLROnPlateau.step`.
    """
    PRIORITY = Priority.LOW  # should be after EvaluationHook

    def __init__(self, metric_key, **kwargs):
        super().__init__()
        self.metric_key = metric_key

    def register_processor(self, trainer):
        lr_scheduler_hook = trainer.get_hook(LrSchedulerHook)
        if len(lr_scheduler_hook) > 0 and type(
                lr_scheduler_hook[0].processor) in (type(None),
                                                    LrSchedulerProcessor):
            lr_scheduler_hook[0].set_processor(
                PlateauLrSchedulerProcessor(self.metric_key))

    def before_run(self, trainer):
        if not hasattr(trainer, 'logger'):
            self.logger = get_logger()
        else:
            self.logger = trainer.logger


@HOOKS.register_module(module_name=Hooks.NoneLrSchedulerHook)
class NoneLrSchedulerHook(LrSchedulerHook):

    PRIORITY = Priority.LOW  # should be after EvaluationHook

    def __init__(self, by_epoch=True, warmup=None) -> None:
        super().__init__(by_epoch=by_epoch, warmup=warmup)

    def before_run(self, trainer):
        return

    def after_train_epoch(self, trainer):
        return
