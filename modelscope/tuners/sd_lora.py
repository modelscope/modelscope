# Copyright 2023-2024 The Alibaba Fundamental Vision Team Authors. All rights reserved.
# The implementation is adopted from HighCWu,
# made pubicly available under the Apache License 2.0 License at https://github.com/HighCWu/ControlLoRA
import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.cross_attention import CrossAttention, LoRALinearLayer
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.outputs import BaseOutput


@dataclass
class TunerOutput(BaseOutput):
    lora_states: Tuple[torch.FloatTensor]


class LoRACrossAttnProcessor(nn.Module):
    """ The implementation of lora attention module.
    """

    def __init__(self,
                 hidden_size,
                 cross_attention_dim=None,
                 rank=4,
                 post_add=False,
                 key_states_skipped=False,
                 value_states_skipped=False,
                 output_states_skipped=False):
        """ Initialize a lora attn instance.
        Args:
            hidden_size (`int`): The number of channels in embedding.
            cross_attention_dim (`int`, *optional*):
                The number of channels in the hidden_states. If not given, defaults to `hidden_size`.
            rank (`int`,  *optional*, defaults to 4): The number of rank of lora.
            post_add (`bool`,  *optional*, defaults to False): Set to `True`, conduct weighted
            adding operation after lora.
            key_states_skipped (`bool`, *optional*, defaults to False):
                Set to `True` for skip to perform lora on key value.
            value_states_skipped (`bool`, *optional*, defaults to False):
                Set to `True` for skip to perform lora on value.
            output_states_skipped (`bool`, *optional*, defaults to False):
                Set to `True` for skip to perform lora on output value.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.post_add = post_add

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        if not key_states_skipped:
            self.to_k_lora = LoRALinearLayer(
                hidden_size if post_add else
                (cross_attention_dim or hidden_size), hidden_size, rank)
        if not value_states_skipped:
            self.to_v_lora = LoRALinearLayer(
                hidden_size if post_add else
                (cross_attention_dim or hidden_size), hidden_size, rank)
        if not output_states_skipped:
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

        self.key_states_skipped: bool = key_states_skipped
        self.value_states_skipped: bool = value_states_skipped
        self.output_states_skipped: bool = output_states_skipped

    def skip_key_states(self, is_skipped: bool = True):
        if not is_skipped:
            assert hasattr(self, 'to_k_lora')
        self.key_states_skipped = is_skipped

    def skip_value_states(self, is_skipped: bool = True):
        if not is_skipped:
            assert hasattr(self, 'to_q_lora')
        self.value_states_skipped = is_skipped

    def skip_output_states(self, is_skipped: bool = True):
        if not is_skipped:
            assert hasattr(self, 'to_out_lora')
        self.output_states_skipped = is_skipped

    def __call__(self,
                 attn: CrossAttention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length,
                                                     batch_size)

        query = attn.to_q(hidden_states)
        query = query + scale * self.to_q_lora(
            query if self.post_add else hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        if not self.key_states_skipped:
            key = key + scale * self.to_k_lora(
                key if self.post_add else encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        if not self.value_states_skipped:
            value = value + scale * self.to_v_lora(
                value if self.post_add else encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        out = attn.to_out[0](hidden_states)
        if not self.output_states_skipped:
            out = out + scale * self.to_out_lora(
                out if self.post_add else hidden_states)
        hidden_states = out
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class LoRATuner(ModelMixin, ConfigMixin):

    @staticmethod
    def tune(
        model: nn.Module,
        tuner_config=None,
        pretrained_tuner=None,
    ):
        tuner = LoRATuner.from_config(tuner_config)
        if pretrained_tuner is not None and os.path.exists(pretrained_tuner):
            tuner.load_state_dict(
                torch.load(pretrained_tuner, map_location='cpu'), strict=True)
        tune_layers_list = list(
            [list(layer_list) for layer_list in tuner.lora_layers])
        assert hasattr(model, 'unet')
        unet = model.unet
        tuner.to(unet.device)
        tune_attn_procs = tuner.set_tune_layers(unet, tune_layers_list)
        unet.set_attn_processor(tune_attn_procs)
        return tuner

    def set_tune_layers(self, unet, tune_layers_list):
        n_ch = len(unet.config.block_out_channels)
        control_ids = [i for i in range(n_ch)]
        tune_attn_procs = {}

        for name in unet.attn_processors.keys():
            if name.startswith('mid_block'):
                control_id = control_ids[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                control_id = list(reversed(control_ids))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                control_id = control_ids[block_id]

            tune_layers = tune_layers_list[control_id]
            if len(tune_layers) != 0:
                tune_layer = tune_layers.pop(0)
                tune_attn_procs[name] = tune_layer
        return tune_attn_procs

    @register_to_config
    def __init__(
        self,
        lora_block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        lora_cross_attention_dims: Tuple[List[int]] = ([
            None, 768, None, 768, None, 768, None, 768, None, 768
        ], [None, 768, None, 768, None, 768, None, 768, None,
            768], [None, 768, None, 768, None, 768, None, 768, None,
                   768], [None, 768]),
        lora_rank: int = 4,
        lora_post_add: bool = False,
        lora_key_states_skipped: bool = False,
        lora_value_states_skipped: bool = False,
        lora_output_states_skipped: bool = False,
    ):
        super().__init__()

        lora_cls = LoRACrossAttnProcessor

        self.lora_layers = nn.ModuleList([])

        for i, lora_cross_attention_dim in enumerate(
                lora_cross_attention_dims):
            self.lora_layers.append(
                nn.ModuleList([
                    lora_cls(
                        lora_block_out_channels[i],
                        cross_attention_dim=cross_attention_dim,
                        rank=lora_rank,
                        post_add=lora_post_add,
                        key_states_skipped=lora_key_states_skipped,
                        value_states_skipped=lora_value_states_skipped,
                        output_states_skipped=lora_output_states_skipped)
                    for cross_attention_dim in lora_cross_attention_dim
                ]))

    def forward(self) -> Union[TunerOutput, Tuple]:
        lora_states_list = []
        tune_layers_list = list(
            [list(layer_list) for layer_list in self.lora_layers])
        for tune_list in tune_layers_list:
            for tune_layer in tune_list:
                lora_states_list.append(tune_layer.to_q_lora.down.weight)
        return TunerOutput(lora_states=tuple(lora_states_list))
