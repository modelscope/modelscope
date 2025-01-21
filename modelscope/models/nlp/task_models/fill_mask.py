# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Heads, TaskModels
from modelscope.models.builder import MODELS
from modelscope.models.nlp.task_models.task_model import EncoderModel
from modelscope.utils.constant import Tasks

__all__ = ['ModelForFillMask']


@MODELS.register_module(Tasks.fill_mask, module_name=TaskModels.fill_mask)
class ModelForFillMask(EncoderModel):
    task = Tasks.fill_mask

    # The default base head type is fill-mask for this head
    head_type = Heads.fill_mask

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        outputs = super().forward(input_ids, attention_mask, token_type_ids,
                                  position_ids, head_mask, inputs_embeds,
                                  labels, output_attentions,
                                  output_hidden_states, **kwargs)

        outputs.input_ids = input_ids
        return outputs

    def get_output_embeddings(self):
        return self.head.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.head.cls.predictions.decoder = new_embeddings

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError('The PAD token should be defined for generation')
        attention_shape0 = attention_mask.shape[0]
        attention_mask = torch.cat(
            [attention_mask,
             attention_mask.new_zeros((attention_shape0, 1))],
            dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}
