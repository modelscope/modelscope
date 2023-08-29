# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

from torch import nn

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.trainer import EpochBasedTrainer


@TRAINERS.register_module(module_name=Trainers.vision_efficient_tuning)
class VisionEfficientTuningTrainer(EpochBasedTrainer):
    """ Vision Efficient Tuning Trainer based on EpochBasedTrainer

    The trainer freezes the parameters of the pre-trained model and
    tunes the extra parameters of the different parameter-efficient
    transfer learning (PETL) method.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_model(self) -> Union[nn.Module, TorchModel]:
        """ Instantiate a pytorch model and return.

        By default, we will create a model using config from configuration file. You can
        override this method in a subclass.

        """
        model = Model.from_pretrained(self.model_dir, cfg_dict=self.cfg)
        if 'freeze_cfg' in self.cfg['model']:
            model = self.freeze(model, **self.cfg['model']['freeze_cfg'])
        if not isinstance(model, nn.Module) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, nn.Module):
            return model

    def train(self, *args, **kwargs):
        self.print_model_params_status()
        super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        metric_values = super().evaluate(*args, **kwargs)
        return metric_values

    def freeze(self, model, freeze_part=[], train_part=[]):
        """ Freeze or train the model based on the config.

        Args:
          model: the current model.
          freeze_part: the config of frozen parameters.
          train_part: the config of trainable parameters.
        """
        if hasattr(model, 'module'):
            freeze_model = model.module
        else:
            freeze_model = model

        if freeze_part and len(freeze_part) > 0:
            if 'backbone' in freeze_part:
                part = freeze_part['backbone']
                for name, param in freeze_model.model.backbone.named_parameters(
                ):
                    freeze_flag = sum([p in name for p in part]) > 0
                    if freeze_flag:
                        param.requires_grad = False
            elif 'head' in freeze_part:
                part = freeze_part['head']
                for name, param in freeze_model.model.head.named_parameters():
                    freeze_flag = sum([p in name for p in part]) > 0
                    if freeze_flag:
                        param.requires_grad = False

        if train_part and len(train_part) > 0:
            if 'backbone' in train_part:
                part = train_part['backbone']
                for name, param in freeze_model.model.backbone.named_parameters(
                ):
                    freeze_flag = sum([p in name for p in part]) > 0
                    if freeze_flag:
                        param.requires_grad = True
            elif 'head' in train_part:
                part = train_part['head']
                for name, param in freeze_model.model.head.named_parameters():
                    freeze_flag = sum([p in name for p in part]) > 0
                    if freeze_flag:
                        param.requires_grad = True
        return model

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
