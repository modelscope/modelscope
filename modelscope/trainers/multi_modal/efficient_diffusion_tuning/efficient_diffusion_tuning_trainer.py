# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from torch import nn

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict


@TRAINERS.register_module(module_name=Trainers.efficient_diffusion_tuning)
class EfficientDiffusionTuningTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self) -> Union[nn.Module, TorchModel]:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model = Model.from_pretrained(self.model_dir, cfg_dict=self.cfg)
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            if hasattr(self, 'tuner'):
                return build_optimizer(
                    self.model.tuner, cfg=cfg, default_args=default_args)
            else:
                return build_optimizer(
                    self.model, cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

    def train(self, *args, **kwargs):
        self.print_model_params_status()
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        eval_res = super().evaluate(*args, **kwargs)
        return eval_res

    def print_model_params_status(self, model=None, logger=None):
        """Print the status and parameters of the model"""
        if model is None:
            model = self.model
        if logger is None:
            logger = self.logger
        train_param_dict = {}
        all_param_numel = 0
        for key, val in model.named_parameters():
            if val.requires_grad:
                sub_key = '.'.join(key.split('.', 1)[-1].split('.', 2)[:2])
                if sub_key in train_param_dict:
                    train_param_dict[sub_key] += val.numel()
                else:
                    train_param_dict[sub_key] = val.numel()
            all_param_numel += val.numel()
        train_param_numel = sum(train_param_dict.values())
        logger.info(
            f'Load trainable params {train_param_numel} / {all_param_numel} = '
            f'{train_param_numel/all_param_numel:.2%}, '
            f'train part: {train_param_dict}.')
