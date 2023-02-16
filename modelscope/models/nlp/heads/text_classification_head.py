# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import (AttentionTextClassificationModelOutput,
                                ModelOutputBase, OutputKeys)
from modelscope.utils.constant import Tasks


@HEADS.register_module(
    Tasks.text_classification, module_name=Heads.text_classification)
@HEADS.register_module(
    Tasks.sentence_similarity, module_name=Heads.text_classification)
@HEADS.register_module(Tasks.nli, module_name=Heads.text_classification)
@HEADS.register_module(
    Tasks.sentiment_classification, module_name=Heads.text_classification)
class TextClassificationHead(TorchHead):

    def __init__(self,
                 hidden_size=768,
                 classifier_dropout=0.1,
                 num_labels=None,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            classifier_dropout=classifier_dropout,
            num_labels=num_labels,
        )
        assert num_labels is not None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self,
                inputs: ModelOutputBase,
                attention_mask=None,
                labels=None,
                **kwargs):
        pooler_output = inputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, labels)

        return AttentionTextClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=inputs.hidden_states,
            attentions=inputs.attentions,
        )

    def compute_loss(self, logits: torch.Tensor, labels) -> torch.Tensor:
        return F.cross_entropy(logits, labels)
