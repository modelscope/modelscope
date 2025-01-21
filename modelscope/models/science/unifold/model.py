# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import argparse
import os
from typing import Any

import torch

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .config import model_config
from .modules.alphafold import AlphaFold

__all__ = ['UnifoldForProteinStructrue']


@MODELS.register_module(Tasks.protein_structure, module_name=Models.unifold)
class UnifoldForProteinStructrue(TorchModel):

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            '--model-name',
            help='choose the model config',
        )

    def __init__(self, **kwargs):
        super().__init__()
        parser = argparse.ArgumentParser()
        parse_comm = []
        for key in kwargs:
            parser.add_argument(f'--{key}')
            parse_comm.append(f'--{key}')
            parse_comm.append(kwargs[key])
        args = parser.parse_args(parse_comm)
        base_architecture(args)
        self.args = args
        config = model_config(
            self.args.model_name,
            train=True,
        )
        self.model = AlphaFold(config)
        self.config = config

        # load model state dict
        param_path = os.path.join(kwargs['model_dir'],
                                  ModelFile.TORCH_MODEL_BIN_FILE)
        state_dict = torch.load(param_path)['ema']['params']
        state_dict = {
            '.'.join(k.split('.')[1:]): v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(state_dict)

    def half(self):
        self.model = self.model.half()
        return self

    def bfloat16(self):
        self.model = self.model.bfloat16()
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        return cls(args)

    def forward(self, batch, **kwargs):
        outputs = self.model.forward(batch)
        return outputs, self.config.loss


def base_architecture(args):
    args.model_name = getattr(args, 'model_name', 'model_2')
