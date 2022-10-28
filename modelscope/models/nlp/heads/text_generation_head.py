# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.utils.constant import Tasks


@HEADS.register_module(
    Tasks.text_generation, module_name=Heads.text_generation)
class TextGenerationHead(TorchHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        self.linear = nn.Linear(
            config['hidden_size'], config['vocab_size'], bias=False)

    def get_output_embeddings(self):
        return self.linear

    def forward(self, inputs=None):
        logits = self.linear(inputs)
        return logits

    def compute_loss(self, logits: torch.Tensor,
                     labels) -> Dict[str, torch.Tensor]:
        return F.cross_entropy(logits, labels)
