# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from modelscope.trainers.hooks.builder import HOOKS
from modelscope.utils.constant import LogKeys
from modelscope.utils.torch_utils import master_only
from .base import LoggerHook


@HOOKS.register_module()
class TensorboardHook(LoggerHook):
    """TensorBoard hook for visualization.
    Args:
        out_dir: output directory to save tensorboard files
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        by_epoch (bool): Whether EpochBasedtrainer is used.
        skip_keys (list): list of keys which will not add to tensorboard
    """

    def __init__(self,
                 out_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=True,
                 skip_keys=[LogKeys.ITER_TIME, LogKeys.DATA_LOAD_TIME]):
        super(TensorboardHook, self).__init__(
            interval=interval,
            ignore_last=ignore_last,
            reset_flag=reset_flag,
            by_epoch=by_epoch)
        self.out_dir = out_dir
        self.skip_keys = skip_keys

    @master_only
    def before_run(self, trainer):
        super(TensorboardHook, self).before_run(trainer)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                e.msg + ' '
                'Please pip install tensorboard by ``pip install future tensorboard`` '
                'or upgrade version by ``pip install future tensorboard --upgrade``.'
            )

        if self.out_dir is None:
            self.out_dir = os.path.join(trainer.work_dir, 'tensorboard_output')
        self.writer = SummaryWriter(self.out_dir)

    @master_only
    def log(self, trainer):
        for key, val in trainer.log_buffer.output.items():
            if key in self.skip_keys:
                continue
            if isinstance(val, str):
                self.writer.add_text(key, val, self.get_iter(trainer))
            elif self.is_scalar(val):
                self.writer.add_scalar(key, val, self.get_iter(trainer))
            else:
                pass

    @master_only
    def after_run(self, trainer):
        self.writer.close()
