# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from typing import Any, Dict

from modelscope.models.base.base_head import Head
from modelscope.utils.logger import get_logger

logger = get_logger()


class TorchHead(Head, torch.nn.Module):
    """ Base head interface for pytorch

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        torch.nn.Module.__init__(self)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
