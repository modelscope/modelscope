# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.trainers.lrscheduler.builder import build_lr_scheduler
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
class LrSchedulerHook(Hook):
    """Lr scheduler.

    Args:
        by_epoch (bool): Whether lr changes by epoch
        warmup (dict): warm up config
    """
    PRIORITY = Priority.VERY_HIGH

    def __init__(self, by_epoch=True, warmup=None) -> None:
        super().__init__()
        self.by_epoch = by_epoch
        if not self.by_epoch:
            raise ValueError('We only support ``by_epoch=True`` now!')

        self.warmup = warmup
        self.warmup_lr_scheduler = None

    def before_run(self, trainer):
        if self.warmup is not None:
            assert isinstance(self.warmup, dict) and 'type' in self.warmup
            self.warmup_lr_scheduler = build_lr_scheduler(
                cfg=self.warmup,
                default_args={'base_scheduler': trainer.lr_scheduler})

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

    def before_train_iter(self, trainer):
        trainer.log_buffer.output['lr'] = self._get_log_lr(trainer)

    def before_train_epoch(self, trainer):
        if self.by_epoch:
            if self.warmup_lr_scheduler is not None:
                self.warmup_lr_scheduler.step()
            else:
                trainer.lr_scheduler.step()
        trainer.log_buffer.output['lr'] = self._get_log_lr(trainer)

    def _get_log_lr(self, trainer):
        cur_lr = self.get_current_lr(trainer)
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
