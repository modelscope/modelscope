# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from torch.nn.utils import clip_grad

from modelscope.metainfo import Hooks
from modelscope.outputs import OutputKeys
from modelscope.trainers.hooks.builder import HOOKS
from modelscope.trainers.hooks.hook import Hook
from modelscope.trainers.hooks.priority import Priority


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
                 loss_keys=OutputKeys.LOSS) -> None:
        if isinstance(loss_keys, str):
            loss_keys = [loss_keys]
        assert isinstance(loss_keys, (tuple, list))
        self.loss_keys = loss_keys
        self.cumulative_iters = cumulative_iters
        self.grad_clip = grad_clip

    def clip_grads(self, params, **clip_args):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **clip_args)

    def before_run(self, trainer):
        trainer.optimizer.zero_grad()
        trainer.cumulative_iters = self.cumulative_iters

    def after_train_iter(self, trainer):
        for k in self.loss_keys:
            trainer.train_outputs[k] /= self.cumulative_iters
            trainer.train_outputs[k].backward()

        if self.every_n_iters(trainer, self.cumulative_iters):
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **self.grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()


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
