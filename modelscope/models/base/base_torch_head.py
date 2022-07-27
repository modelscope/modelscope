# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch

from modelscope.models.base.base_head import Head
from modelscope.utils.logger import get_logger

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
