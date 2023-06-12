# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict


@TRAINERS.register_module(module_name=Trainers.stable_diffusion)
class StableDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.model.unet,
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e