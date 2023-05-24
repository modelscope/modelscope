# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from torch.nn.utils import clip_grad

from modelscope.metainfo import Hooks
from modelscope.outputs import OutputKeys
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority


class OptimizerProcessor:

    def initialize_optimizer(self, trainer):
        """Initialize the optimizer.

        This is a strategic function which can be registered by other hook's function.
        """
        trainer.optimizer.zero_grad()

    def before_forward(self, trainer):
        pass

    def backward(self, trainer, loss_keys, cumulative_iters, grad_clip):
        """Do module backward, optimizer's step and zero_grad and clip the grads.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            loss_keys(`list`): The list of loss keys.
            cumulative_iters(`int`): The cumulative iters for gradients.
            grad_clip(`dict`): The grad clipping options.
        """
        for k in loss_keys:
            trainer.train_outputs[k] /= cumulative_iters
            trainer.train_outputs[k].backward()

        if Hook.every_n_iters(trainer, cumulative_iters):
            if grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

    @staticmethod
    def clip_grads(params, **clip_args):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **clip_args)


@HOOKS.register_module(module_name=Hooks.OptimizerHook)
class OptimizerHook(Hook):
    """Optimizer hook

    Args:
        cumulative_iters (int): interval of gradients accumulation. Default: 1
        grad_clip (dict): Default None. Containing keys:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            More details please refer to `torch.nn.utils.clip_grad.clip_grad_norm_`
        loss_keys (str | list): keys list of loss
    """

    PRIORITY = Priority.ABOVE_NORMAL

    def __init__(self,
                 cumulative_iters=1,
                 grad_clip=None,
                 loss_keys=OutputKeys.LOSS,
                 **kwargs) -> None:
        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        assert isinstance(loss_keys, (tuple, list))
        self.loss_keys = loss_keys
        self.cumulative_iters = cumulative_iters
        self.grad_clip = grad_clip
        self.processor = OptimizerProcessor()

    def set_processor(self, processor):
        self.processor = processor

    def before_run(self, trainer):
        trainer.cumulative_iters = self.cumulative_iters
        self.processor.initialize_optimizer(trainer)

    def before_train_iter(self, trainer):
        self.processor.before_forward(trainer)

    def after_train_iter(self, trainer):
        self.processor.backward(trainer, self.loss_keys, self.cumulative_iters,
                                self.grad_clip)


@HOOKS.register_module(module_name=Hooks.NoneOptimizerHook)
class NoneOptimizerHook(OptimizerHook):

    def __init__(self, cumulative_iters=1, grad_clip=None, loss_keys='loss'):

        super(NoneOptimizerHook, self).__init__(
            grad_clip=grad_clip, loss_keys=loss_keys)
        self.cumulative_iters = cumulative_iters

    def before_run(self, trainer):
        return

    def after_train_iter(self, trainer):
        return
