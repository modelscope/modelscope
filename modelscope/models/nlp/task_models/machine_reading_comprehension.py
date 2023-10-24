# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig
from transformers.modeling_outputs import ModelOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaModel, RobertaPreTrainedModel)

from modelscope.metainfo import Heads, Models, TaskModels
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.outputs import MachineReadingComprehensionOutput, OutputKeys
from modelscope.utils.compatible_with_transformers import \
    compatible_position_ids
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.hub import parse_label_mapping

__all__ = ['ModelForMachineReadingComprehension']


@MODELS.register_module(
    Tasks.machine_reading_comprehension,
    module_name=TaskModels.machine_reading_comprehension)
class ModelForMachineReadingComprehension(TorchModel):
    '''
    Pretrained Machine Reader (PMR) model (https://arxiv.org/pdf/2212.04755.pdf)

    '''

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.num_labels = self.config.num_labels
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        self.span_transfer = MultiNonLinearProjection(
            self.config.hidden_size,
            self.config.hidden_size,
            self.config.hidden_dropout_prob,
            intermediate_hidden_size=self.config.
            projection_intermediate_hidden_size)
        state_dict = torch.load(
            os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE))
        compatible_position_ids(state_dict, 'roberta.embeddings.position_ids')
        self.load_state_dict(state_dict)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        label_mask=None,
        match_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # adapted from https://github.com/ShannonAI/mrc-for-flat-nested-ner
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, hidden]
        span_intermediate = self.span_transfer(sequence_output)
        # [batch, seq_len, seq_len]
        span_logits = torch.matmul(span_intermediate,
                                   sequence_output.transpose(-1, -2))

        total_loss = None
        if match_labels is not None:
            match_loss = self.compute_loss(span_logits, match_labels,
                                           label_mask)
            total_loss = match_loss
        if not return_dict:
            output = (span_logits) + outputs[2:]
            return ((total_loss, )
                    + output) if total_loss is not None else output

        return MachineReadingComprehensionOutput(
            loss=total_loss,
            span_logits=span_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MultiNonLinearProjection(nn.Module):

    def __init__(self,
                 hidden_size,
                 num_label,
                 dropout_rate,
                 act_func='gelu',
                 intermediate_hidden_size=None):
        super(MultiNonLinearProjection, self).__init__()
        self.num_label = num_label
        self.intermediate_hidden_size = hidden_size if intermediate_hidden_size is None else intermediate_hidden_size
        self.classifier1 = nn.Linear(hidden_size,
                                     self.intermediate_hidden_size)
        self.classifier2 = nn.Linear(self.intermediate_hidden_size,
                                     self.num_label)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == 'gelu':
            features_output1 = F.gelu(features_output1)
        elif self.act_func == 'relu':
            features_output1 = F.relu(features_output1)
        elif self.act_func == 'tanh':
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2
