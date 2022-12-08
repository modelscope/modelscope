# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from modelscope.metainfo import Heads
from modelscope.models.base import TorchHead
from modelscope.models.builder import HEADS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks


@HEADS.register_module(
    Tasks.token_classification, module_name=Heads.token_classification)
@HEADS.register_module(
    Tasks.named_entity_recognition, module_name=Heads.token_classification)
@HEADS.register_module(
    Tasks.part_of_speech, module_name=Heads.token_classification)
class TokenClassificationHead(TorchHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        self.num_labels = config.num_labels
        classifier_dropout = (
            config['classifier_dropout'] if config.get('classifier_dropout')
            is not None else config['hidden_dropout_prob'])
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config['hidden_size'],
                                    config['num_labels'])

    def forward(self, inputs=None):
        if isinstance(inputs, dict):
            assert inputs.get('sequence_output') is not None
            sequence_output = inputs.get('sequence_output')
        else:
            sequence_output = inputs
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits

    def compute_loss(self, outputs: Dict[str, torch.Tensor],
                     labels) -> Dict[str, torch.Tensor]:
        logits = outputs[OutputKeys.LOGITS]
        return F.cross_entropy(logits, labels)
