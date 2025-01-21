# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
from torch import nn

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import (AttentionTextClassificationModelOutput,
                                ModelOutputBase, OutputKeys)
from modelscope.utils.constant import Tasks


@HEADS.register_module(Tasks.text_ranking, module_name=Heads.text_ranking)
class TextRankingHead(TorchHead):

    def __init__(self,
                 hidden_size=768,
                 classifier_dropout=0.1,
                 num_labels=1,
                 neg_sample=4,
                 **kwargs):
        super().__init__(
            hidden_size=hidden_size,
            classifier_dropout=classifier_dropout,
            num_labels=num_labels,
            neg_sample=neg_sample,
        )
        self.neg_sample = neg_sample
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
        if self.training:
            loss = self.compute_loss(logits)

        return AttentionTextClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=inputs.hidden_states,
            attentions=inputs.attentions,
        )

    def compute_loss(self, logits: torch.Tensor) -> torch.Tensor:
        scores = logits.view(-1, self.neg_sample + 1)
        batch_size = scores.size(0)
        loss_fct = torch.nn.CrossEntropyLoss()
        target_label = torch.zeros(
            batch_size, dtype=torch.long, device=scores.device)
        loss = loss_fct(scores, target_label)
        return loss
