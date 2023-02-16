# Copyright (c) Alibaba, Inc. and its affiliates.
from copy import deepcopy

import torch
from torch import nn

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import BertEncoder, BertModel, BertPreTrainedModel

__all__ = ['SiameseUieModel']


@MODELS.register_module(Tasks.siamese_uie, module_name=Models.bert)
class SiameseUieModel(BertPreTrainedModel):
    r"""SiameseUIE general information extraction model,
        based on the construction idea of prompt (Prompt) + text (Text),
        uses pointer network (Pointer Network) to
        realize segment extraction (Span Extraction), so as to
        realize named entity recognition (NER), relation extraction (RE),
        Extraction of various tasks such as event extraction (EE),
        attribute sentiment extraction (ABSA), etc. Different from
        the existing general information extraction tasks on the market:
    """

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.plm = BertModel(self.config, add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.head_clsf = nn.Linear(config.hidden_size, 1)
        self.tail_clsf = nn.Linear(config.hidden_size, 1)
        self.set_crossattention_layer()

    def set_crossattention_layer(self, num_hidden_layers=6):
        crossattention_config = deepcopy(self.config)
        crossattention_config.num_hidden_layers = num_hidden_layers
        self.config.num_hidden_layers -= num_hidden_layers
        self.crossattention = BertEncoder(crossattention_config)
        self.crossattention.layer = self.plm.encoder.layer[self.config.
                                                           num_hidden_layers:]
        self.plm.encoder.layer = self.plm.encoder.layer[:self.config.
                                                        num_hidden_layers]

    def get_cross_attention_output(self, hidden_states, attention_mask,
                                   encoder_hidden_states,
                                   encoder_attention_mask):
        cat_hidden_states = torch.cat([hidden_states, encoder_hidden_states],
                                      dim=1)
        cat_attention_mask = torch.cat(
            [attention_mask, encoder_attention_mask], dim=1)
        cat_attention_mask = self.plm.get_extended_attention_mask(
            cat_attention_mask,
            cat_hidden_states.size()[:2])
        hidden_states = self.crossattention(
            hidden_states=cat_hidden_states, attention_mask=cat_attention_mask
        )[0][:, :hidden_states.size()[1], :]
        return hidden_states

    def get_plm_sequence_output(self,
                                input_ids,
                                attention_mask,
                                position_ids=None,
                                is_hint=False):
        token_type_ids = torch.ones_like(attention_mask) if is_hint else None
        sequence_output = self.plm(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids)[0]
        return sequence_output

    def forward(self, sequence_output, attention_masks, hint_ids,
                cross_attention_masks):
        """

        Args:
            sequence_output(tensor): 3-dimension tensor (batch size, sequence length, hidden size)
            attention_masks(tensor): attention mask, 2-dimension tensor (batch size, sequence length)
            hint_ids(tensor): token ids of prompt 2-dimension tensor (batch size, sequence length)
            cross_attention_masks(tensor): cross attention mask, 2-dimension tensor (batch size, sequence length)
        Default Returns:
            head_probs(tensor): 2-dimension tensor(batch size, sequence length)
            tail_probs(tensor): 2-dimension tensor(batch size, sequence length)
        """
        position_ids = torch.arange(hint_ids.size(1)).expand(
            (1, -1)) + sequence_output.size(1)
        position_ids = position_ids.to(sequence_output.device)
        hint_sequence_output = self.get_plm_sequence_output(
            hint_ids, cross_attention_masks, position_ids, is_hint=True)
        sequence_output = self.get_cross_attention_output(
            sequence_output, attention_masks, hint_sequence_output,
            cross_attention_masks)
        # (b, l, n)
        head_logits = self.head_clsf(sequence_output).squeeze(-1)
        tail_logits = self.tail_clsf(sequence_output).squeeze(-1)
        head_probs = head_logits + (1 - attention_masks) * -10000
        tail_probs = tail_logits + (1 - attention_masks) * -10000
        return head_probs, tail_probs
