# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

from modelscope.metainfo import Hooks
from modelscope.trainers.hooks.builder import HOOKS
from .base import OptimizerHook


@HOOKS.register_module(module_name=Hooks.ApexAMPOptimizerHook)
class ApexAMPOptimizerHook(OptimizerHook):
    """Fp16 optimizer, if torch version is less than 1.6.0,
    you must install apex (https://www.github.com/nvidia/apex) else use torch.cuda.amp by default
    Args:
        cumulative_iters (int): interval of gradients accumulation. Default: 1
        grad_clip (dict): Default None. Containing keys:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
            More details please refer to `torch.nn.utils.clip_grad.clip_grad_norm_`
        loss_keys (str | list): keys list of loss
        opt_level (str): "O0" and "O3" are not true mixed precision,
            but they are useful for establishing accuracy and speed baselines, respectively.
            "O1" and "O2" are different implementations of mixed precision.
            Try both, and see what gives the best speedup and accuracy for your model.
    """

    def __init__(self,
                 cumulative_iters=1,
                 grad_clip=None,
                 loss_keys='loss',
                 opt_level='O1'):

        super(ApexAMPOptimizerHook, self).__init__(
            grad_clip=grad_clip, loss_keys=loss_keys)
        self.cumulative_iters = cumulative_iters
        self.opt_level = opt_level

        try:
            from apex import amp
        except ImportError:
            raise ValueError(
                'apex not installed, please install apex from https://www.github.com/nvidia/apex.'
            )

    def before_run(self, trainer):
        from apex import amp

        logging.info('open fp16')
        # TODO: fix it should initialze amp with model not wrapper by DDP or DP
        if hasattr(trainer.model, 'module'):
            trainer.model, trainer.optimizer = amp.initialize(
                trainer.model.module,
                trainer.optimizer,
                opt_level=self.opt_level)
        else:
            trainer.model, trainer.optimizer = amp.initialize(
                trainer.model, trainer.optimizer, opt_level=self.opt_level)

        trainer.optimizer.zero_grad()

    def after_train_iter(self, trainer):
        for k in self.loss_keys:
            trainer.train_outputs[k] /= self.cumulative_iters

        from apex import amp
        for k in self.loss_keys:
            with amp.scale_loss(trainer.train_outputs[k],
                                trainer.optimizer) as scaled_loss:
                scaled_loss.backward()

        if self.every_n_iters(trainer, self.cumulative_iters):
            if self.grad_clip is not None:
                self.clip_grads(trainer.model.parameters(), **self.grad_clip)

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
