# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from modelscope.metainfo import Hooks
from modelscope.trainers.multi_modal.clip.clip_trainer import CLIPTrainer
from .builder import HOOKS
from .hook import Hook


@HOOKS.register_module(module_name=Hooks.ClipClampLogitScaleHook)
class ClipClampLogitScaleHook(Hook):
    """ClipClampLogitScaleHook hook which performs clamp on CLIP logit scale parameter after update"""

    def after_train_iter(self, trainer: CLIPTrainer):
        """Called after every training iter to evaluate the results."""
        unwrapped_model = getattr(trainer.model, 'module', trainer.model)
        logit_scale = unwrapped_model.clip_model.logit_scale
        logit_scale.data = torch.clamp(logit_scale.data, 0, 4.6052)
