# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import (AttentionTokenClassificationModelOutput,
                                ModelOutputBase, OutputKeys)
from modelscope.utils.constant import Tasks


@HEADS.register_module(
    Tasks.token_classification, module_name=Heads.token_classification)
@HEADS.register_module(
    Tasks.named_entity_recognition, module_name=Heads.token_classification)
@HEADS.register_module(
    Tasks.part_of_speech, module_name=Heads.token_classification)
class TokenClassificationHead(TorchHead):

    def __init__(self,
                 hidden_size=768,
                 classifier_dropout=0.1,
                 num_labels=None,
                 **kwargs):
        super().__init__(
            num_labels=num_labels,
            classifier_dropout=classifier_dropout,
            hidden_size=hidden_size,
        )
        assert num_labels is not None
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self,
                inputs: ModelOutputBase,
                attention_mask=None,
                labels=None,
                **kwargs):
        sequence_output = inputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, attention_mask, labels)

        return AttentionTokenClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=inputs.hidden_states,
            attentions=inputs.attentions)

    def compute_loss(self, logits: torch.Tensor, attention_mask,
                     labels) -> torch.Tensor:
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
