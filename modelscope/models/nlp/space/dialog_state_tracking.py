# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
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
"""PyTorch Space model. mainly copied from :module:`~transformers.modeling_xlm_roberta`"""

from typing import Dict

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.file_utils import add_start_docstrings
from transformers.modeling_utils import PreTrainedModel

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.nlp.structbert import (SbertForMaskedLM, SbertModel,
                                              SbertPreTrainedModel)
from modelscope.utils.constant import Tasks
from .configuration import SpaceConfig

SPACE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config ([`SpaceConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.
"""


@add_start_docstrings(
    'The bare Space Model transformer outputting raw hidden-states without any specific head on top. '
    'It is identical with the Bert Model from Transformers',
    SPACE_START_DOCSTRING,
)
class SpaceModel(SbertModel):
    """
    This class overrides [`SbertModel`]. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """

    config_class = SpaceConfig


class SpacePreTrainedModel(TorchModel, PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpaceConfig
    base_model_prefix = 'bert'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def __init__(self, config, **kwargs):
        super().__init__(config.name_or_path, **kwargs)
        super(Model, self).__init__(config)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        @param kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels is not input.
                    label2id: An optional label2id mapping, which will cover the label2id in configuration (if exists).

        @return: The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """

        model_dir = kwargs.pop('model_dir', None)
        if model_dir is None:
            config = SpaceConfig(**kwargs)
            model = cls(config)
        else:
            model_kwargs = {}
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_kwargs)
        return model


@add_start_docstrings(
    """
    Space Model transformer with Dialog state tracking heads on top (a inform projection
    layer with a dialog state layer and a set of slots including history infromation from
    previous dialog) e.g. for multiwoz2.2 tasks.
    """,
    SPACE_START_DOCSTRING,
)
@MODELS.register_module(
    Tasks.task_oriented_conversation, module_name=Models.space_dst)
class SpaceForDST(SpacePreTrainedModel):

    def __init__(self, config):
        super(SpaceForDST, self).__init__(config)
        self.slot_list = config.dst_slot_list
        self.class_types = config.dst_class_types
        self.class_labels = config.dst_class_labels
        self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
        self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
        self.class_aux_feats_inform = config.dst_class_aux_feats_inform
        self.class_aux_feats_ds = config.dst_class_aux_feats_ds
        self.class_loss_ratio = config.dst_class_loss_ratio

        # Only use refer loss if refer class is present in dataset.
        if 'refer' in self.class_types:
            self.refer_index = self.class_types.index('refer')
        else:
            self.refer_index = -1

        self.bert = SpaceModel(config)
        self.dropout = nn.Dropout(config.dst_dropout_rate)
        self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

        if self.class_aux_feats_inform:
            self.add_module(
                'inform_projection',
                nn.Linear(len(self.slot_list), len(self.slot_list)))
        if self.class_aux_feats_ds:
            self.add_module(
                'ds_projection',
                nn.Linear(len(self.slot_list), len(self.slot_list)))

        aux_dims = len(self.slot_list) * (
            self.class_aux_feats_inform + self.class_aux_feats_ds
        )  # second term is 0, 1 or 2

        for slot in self.slot_list:
            self.add_module(
                'class_' + slot,
                nn.Linear(config.hidden_size + aux_dims, self.class_labels))
            self.add_module('token_' + slot, nn.Linear(config.hidden_size, 2))
            self.add_module(
                'refer_' + slot,
                nn.Linear(config.hidden_size + aux_dims,
                          len(self.slot_list) + 1))

        self.init_weights()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'inputs': dict(input_ids, input_masks,start_pos), # tracking states
                        'outputs': dict(slots_logits),
                        'unique_ids': str(test-example.json-0), # default value
                        'input_ids_unmasked': array([101, 7632, 1010,0,0,0])
                        'values': array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}]),
                        'inform':  array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}]),
                        'prefix': str('final'), #default value
                        'ds':  array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}])
                    }

        Example:
            >>> from modelscope.hub.snapshot_download import snapshot_download
            >>> from modelscope.models.nlp import SpaceForDST
            >>> from modelscope.preprocessors import DialogStateTrackingPreprocessor
            >>> cache_path = snapshot_download('damo/nlp_space_dialog-state-tracking')
            >>> model = SpaceForDST.from_pretrained(cache_path)
            >>> preprocessor = DialogStateTrackingPreprocessor(model_dir=cache_path)
            >>> print(model(preprocessor({
                    'utter': {
                        'User-1': "Hi, I'm looking for a train that is going"
                            "to cambridge and arriving there by 20:45, is there anything like that?"
                    },
                    'history_states': [{}]
                })))
        """
        import numpy as np
        import torch

        # self.model.eval() ????
        batch = input['batch']

        features = input['features']
        diag_state = input['diag_state']
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in self.config.dst_slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'input_mask': batch[1],
                'segment_ids': batch[2],
                'start_pos': batch[3],
                'end_pos': batch[4],
                'inform_slot_id': batch[5],
                'refer_id': batch[6],
                'diag_state': diag_state,
                'class_label_id': batch[8]
            }
            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [
                features[i.item()].input_ids_unmasked for i in batch[9]
            ]
            inform = [features[i.item()].inform for i in batch[9]]
            outputs = self._forward(**inputs)

            # Update dialog state for next turn.
            for slot in self.config.dst_slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

        return {
            'inputs': inputs,
            'outputs': outputs,
            'unique_ids': unique_ids,
            'input_ids_unmasked': input_ids_unmasked,
            'values': values,
            'inform': inform,
            'prefix': 'final',
            'ds': input['ds']
        }

    def _forward(self,
                 input_ids,
                 input_mask=None,
                 segment_ids=None,
                 position_ids=None,
                 head_mask=None,
                 start_pos=None,
                 end_pos=None,
                 inform_slot_id=None,
                 refer_id=None,
                 class_label_id=None,
                 diag_state=None):
        outputs = self.bert(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            position_ids=position_ids,
            head_mask=head_mask)

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # TODO: establish proper format in labels already?
        if inform_slot_id is not None:
            inform_labels = torch.stack(list(inform_slot_id.values()),
                                        1).float()
        if diag_state is not None:
            diag_state_labels = torch.clamp(
                torch.stack(list(diag_state.values()), 1).float(), 0.0, 1.0)

        total_loss = 0
        per_slot_per_example_loss = {}
        per_slot_class_logits = {}
        per_slot_start_logits = {}
        per_slot_end_logits = {}
        per_slot_refer_logits = {}
        for slot in self.slot_list:
            if self.class_aux_feats_inform and self.class_aux_feats_ds:
                pooled_output_aux = torch.cat(
                    (pooled_output, self.inform_projection(inform_labels),
                     self.ds_projection(diag_state_labels)), 1)
            elif self.class_aux_feats_inform:
                pooled_output_aux = torch.cat(
                    (pooled_output, self.inform_projection(inform_labels)), 1)
            elif self.class_aux_feats_ds:
                pooled_output_aux = torch.cat(
                    (pooled_output, self.ds_projection(diag_state_labels)), 1)
            else:
                pooled_output_aux = pooled_output
            class_logits = self.dropout_heads(
                getattr(self, 'class_' + slot)(pooled_output_aux))

            token_logits = self.dropout_heads(
                getattr(self, 'token_' + slot)(sequence_output))
            start_logits, end_logits = token_logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            refer_logits = self.dropout_heads(
                getattr(self, 'refer_' + slot)(pooled_output_aux))

            per_slot_class_logits[slot] = class_logits
            per_slot_start_logits[slot] = start_logits
            per_slot_end_logits[slot] = end_logits
            per_slot_refer_logits[slot] = refer_logits

            # If there are no labels, don't compute loss
            if class_label_id is not None and start_pos is not None and end_pos is not None and refer_id is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_pos[slot].size()) > 1:
                    start_pos[slot] = start_pos[slot].squeeze(-1)
                if len(end_pos[slot].size()) > 1:
                    end_pos[slot] = end_pos[slot].squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)  # This is a single index
                start_pos[slot].clamp_(0, ignored_index)
                end_pos[slot].clamp_(0, ignored_index)

                class_loss_fct = CrossEntropyLoss(reduction='none')
                token_loss_fct = CrossEntropyLoss(
                    reduction='none', ignore_index=ignored_index)
                refer_loss_fct = CrossEntropyLoss(reduction='none')

                start_loss = token_loss_fct(start_logits, start_pos[slot])
                end_loss = token_loss_fct(end_logits, end_pos[slot])
                token_loss = (start_loss + end_loss) / 2.0

                token_is_pointable = (start_pos[slot] > 0).float()
                if not self.token_loss_for_nonpointable:
                    token_loss *= token_is_pointable

                refer_loss = refer_loss_fct(refer_logits, refer_id[slot])
                token_is_referrable = torch.eq(class_label_id[slot],
                                               self.refer_index).float()
                if not self.refer_loss_for_nonpointable:
                    refer_loss *= token_is_referrable

                class_loss = class_loss_fct(class_logits, class_label_id[slot])

                if self.refer_index > -1:
                    per_example_loss = (self.class_loss_ratio) * class_loss + (
                        (1 - self.class_loss_ratio) / 2) * token_loss + (
                            (1 - self.class_loss_ratio) / 2) * refer_loss
                else:
                    per_example_loss = self.class_loss_ratio * class_loss + (
                        1 - self.class_loss_ratio) * token_loss

                total_loss += per_example_loss.sum()
                per_slot_per_example_loss[slot] = per_example_loss

        # add hidden states and attention if they are here
        outputs = (total_loss, ) + (
            per_slot_per_example_loss,
            per_slot_class_logits,
            per_slot_start_logits,
            per_slot_end_logits,
            per_slot_refer_logits,
        ) + (outputs.embedding_output, )

        return outputs
