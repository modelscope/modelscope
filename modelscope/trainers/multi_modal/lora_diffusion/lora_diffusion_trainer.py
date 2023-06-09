# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from torch import nn
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import \
    CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor \
    import CheckpointProcessor

class LoraDiffusionCheckpointProcessor(CheckpointProcessor):

    @staticmethod
    def _bin_file(model):
        """Get bin file path for diffuser.
        """
        default_bin_file = 'diffusion_pytorch_model.bin'
        return default_bin_file
    

@TRAINERS.register_module(module_name=Trainers.lora_diffusion)
class LoraDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
        ckpt_hook.set_processor(LoraDiffusionCheckpointProcessor())

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.model.tuner, cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

    def train(self, *args, **kwargs):
        
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        eval_res = super().evaluate(*args, **kwargs)
        return eval_res
