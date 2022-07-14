# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope import __version__
from modelscope.utils.checkpoint import save_checkpoint
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import get_dist_info
from .builder import HOOKS
from .hook import Hook
from .priority import Priority


@HOOKS.register_module()
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

        if not hasattr(trainer, 'logger'):
            self.logger = get_logger(__name__)
        else:
            self.logger = trainer.logger

        self.logger.info(f'Checkpoints will be saved to {self.save_dir}')

    def after_train_epoch(self, trainer):
        if not self.by_epoch:
            return

        if self._should_save(trainer):
            self.logger.info(f'Saving checkpoint at {trainer.epoch + 1} epoch')
            self._save_checkpoint(trainer)

    def _save_checkpoint(self, trainer):
        if self.by_epoch:
            cur_save_name = os.path.join(self.save_dir,
                                         f'epoch_{trainer.epoch + 1}.pth')
        else:
            cur_save_name = os.path.join(self.save_dir,
                                         f'iter_{trainer.epoch + 1}.pth')

        rank, _ = get_dist_info()
        if rank == 0:
            save_checkpoint(trainer.model, cur_save_name, trainer.optimizer)

    def after_train_iter(self, trainer):
        if self.by_epoch:
            return

        if self._should_save(trainer):
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
