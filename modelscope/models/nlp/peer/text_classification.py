# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from torch.nn import CrossEntropyLoss, MSELoss

from modelscope.metainfo import Models
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTextClassificationModelOutput
from modelscope.utils import logger as logging
from modelscope.utils.constant import Tasks
from .backbone import (PeerClassificationHead, PeerModel, PeerPreTrainedModel,
                       PeerTopModel)

logger = logging.get_logger()


@MODELS.register_module(Tasks.text_classification, module_name=Models.peer)
@MODELS.register_module(Tasks.nli, module_name=Models.peer)
@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.peer)
@MODELS.register_module(Tasks.sentence_similarity, module_name=Models.peer)
@MODELS.register_module(
    Tasks.zero_shot_classification, module_name=Models.peer)
class PeerForSequenceClassification(PeerPreTrainedModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        config_discr_top = copy.deepcopy(config)
        config_shared_bottom = copy.deepcopy(config)

        assert config.num_hidden_layers_shared > 0, 'config.num_hidden_layers_shared should be greater than 0!'

        config_shared_bottom.num_hidden_layers = config.num_hidden_layers_shared
        config_discr_top.num_hidden_layers = config_discr_top.num_hidden_layers \
            - config_discr_top.num_hidden_layers_shared

        self.teams1_shared_bottom = PeerModel(config_shared_bottom)
        self.teams1_discr_top = PeerTopModel(config_discr_top)

        self.classifier = PeerClassificationHead(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            side_info_sets=dict(),
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states_discr_bottom = self.teams1_shared_bottom(
            input_ids, attention_mask, token_type_ids, position_ids, head_mask,
            inputs_embeds, output_attentions, output_hidden_states,
            side_info_sets, return_dict)

        hidden_states_discr_top = self.teams1_discr_top(
            hidden_states_discr_bottom[0], input_ids, attention_mask,
            token_type_ids, position_ids, head_mask, inputs_embeds,
            output_attentions, output_hidden_states, side_info_sets,
            return_dict)

        discriminator_hidden_states = hidden_states_discr_top

        sequence_output = discriminator_hidden_states[0]

        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return AttentionTextClassificationModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
