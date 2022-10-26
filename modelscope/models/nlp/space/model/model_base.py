# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import torch.nn as nn

from modelscope.utils.constant import ModelFile


class SpaceModelBase(nn.Module):
    """
    Basic model wrapper for static graph and dygrpah.
    """
    _registry = dict()

    @classmethod
    def register(cls, name):
        SpaceModelBase._registry[name] = cls
        return

    @staticmethod
    def by_name(name):
        return SpaceModelBase._registry[name]

    @staticmethod
    def create(model_dir, config, *args, **kwargs):
        model_cls = SpaceModelBase.by_name(config.Model.model)
        return model_cls(model_dir, config, *args, **kwargs)

    def __init__(self, model_dir, config):
        super(SpaceModelBase, self).__init__()
        self.init_checkpoint = os.path.join(model_dir,
                                            ModelFile.TORCH_MODEL_BIN_FILE)
        self.abandon_label = config.Dataset.abandon_label
        self.use_gpu = config.use_gpu
        self.gpu = config.Trainer.gpu
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        raise NotImplementedError

    def _forward(self, inputs, is_training, with_label):
        """ NO LABEL: Real forward process of model in different mode(train/test). """
        raise NotImplementedError

    def _collect_metrics(self, inputs, outputs, with_label, data_file):
        """ NO LABEL: Calculate loss function by using inputs and outputs. """
        raise NotImplementedError

    def _optimize(self, loss, optimizer, lr_scheduler):
        """ Optimize loss function and update model. """
        raise NotImplementedError

    def _infer(self, inputs, start_id, eos_id, max_gen_len, prev_input):
        """ Real inference process of model. """
        raise NotImplementedError

    def forward(self,
                inputs,
                is_training=False,
                with_label=False,
                data_file=None):
        """
        Forward process, include real forward, collect metrices and optimize(optional)

        Args:
            inputs(`dict` of numpy.ndarray/int/float/...) : input data
        """
        if is_training:
            self.train()
        else:
            self.eval()

        with_label = False if self.abandon_label else with_label
        outputs = self._forward(inputs, is_training, with_label=with_label)
        metrics = self._collect_metrics(
            inputs, outputs, with_label=with_label, data_file=data_file)

        return metrics

    def infer(self,
              inputs,
              start_id=None,
              eos_id=None,
              max_gen_len=None,
              prev_input=None):
        """Inference process.

        Args:
            inputs(`dict` of numpy.ndarray/int/float/...) : input data
        """
        self.eval()
        results = self._infer(
            inputs,
            start_id=start_id,
            eos_id=eos_id,
            max_gen_len=max_gen_len,
            prev_input=prev_input)
        return results
