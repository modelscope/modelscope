# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
import re
from typing import Dict, Optional, Union

import torch
from torch import nn

from ...utils.logger import get_logger
from .base_head import Head

logger = get_logger(__name__)


class TorchHead(Head, torch.nn.Module):
    """ Base head interface for pytorch

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        torch.nn.Module.__init__(self)

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     labels) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
