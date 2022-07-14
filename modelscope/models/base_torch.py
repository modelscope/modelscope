# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Dict

import torch

from .base import Model


class TorchModel(Model, torch.nn.Module):
    """ Base model interface for pytorch

    """

    def __init__(self, model_dir=None, *args, **kwargs):
        # init reference: https://stackoverflow.com/questions\
        # /9575409/calling-parent-class-init-with-multiple-inheritance-whats-the-right-way
        super().__init__(model_dir)
        super(Model, self).__init__()

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
