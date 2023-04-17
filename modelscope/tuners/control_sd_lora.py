# Copyright 2023-2024 The Alibaba Fundamental Vision Team Authors. All rights reserved.
# The implementation is adopted from HighCWu,
# made pubicly available under the Apache License 2.0 License at https://github.com/HighCWu/ControlLoRA

import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.cross_attention import CrossAttention, LoRALinearLayer
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import (Downsample2D, Mish, Upsample2D,
                                     downsample_2d, partial, upsample_2d)
from diffusers.models.unet_2d_blocks import \
    get_down_block as get_down_block_default
from diffusers.utils.outputs import BaseOutput

from .sd_lora import LoRACrossAttnProcessor


@dataclass
class ControlLoRAOutput(BaseOutput):
    control_states: Tuple[torch.FloatTensor]


class ControlLoRATuner(ModelMixin, ConfigMixin):
    """ The implementation of control lora module.
    This module conduct encoding operation for control-condition and use lora to perform efficient tuning.
    """

    @staticmethod
    def tune(
        model: nn.Module,
        tuner_config=None,
        pretrained_tuner=None,
    ):
        tuner = ControlLoRATuner.from_config(tuner_config)
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
    def __init__(self,
                 in_channels: int = 3,
                 down_block_types: Tuple[str] = (
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                 ),
                 block_out_channels: Tuple[int] = (32, 64, 128, 256),
                 layers_per_block: int = 1,
                 act_fn: str = 'silu',
                 norm_num_groups: int = 32,
                 lora_pre_down_block_types: Tuple[str] = (
                     None,
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                 ),
                 lora_pre_down_layers_per_block: int = 1,
                 lora_pre_conv_skipped: bool = False,
                 lora_pre_conv_types: Tuple[str] = (
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                     'SimpleDownEncoderBlock2D',
                 ),
                 lora_pre_conv_layers_per_block: int = 1,
                 lora_pre_conv_layers_kernel_size: int = 1,
                 lora_block_in_channels: Tuple[int] = (256, 256, 256, 256),
                 lora_block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
                 lora_cross_attention_dims: Tuple[List[int]] = ([
                     None, 768, None, 768, None, 768, None, 768, None, 768
                 ], [None, 768, None, 768, None, 768, None, 768, None, 768], [
                     None, 768, None, 768, None, 768, None, 768, None, 768
                 ], [None, 768]),
                 lora_rank: int = 4,
                 lora_control_rank: int = None,
                 lora_post_add: bool = False,
                 lora_concat_hidden: bool = False,
                 lora_control_channels: Tuple[int] = (None, None, None, None),
                 lora_control_self_add: bool = True,
                 lora_key_states_skipped: bool = False,
                 lora_value_states_skipped: bool = False,
                 lora_output_states_skipped: bool = False,
                 lora_control_version: int = 1):
        """ Initialize a control lora module instance.
               Args:
                   in_channels (`int`): The number of channels for input conditional data.
                   down_block_types (Tuple[str], *optional*):
                       The down block types for conditional data's downsample operation.
                   block_out_channels (Tuple[int],  *optional*, defaults to (32, 64, 128, 256)):
                       The number of channels for every down-block.
                   layers_per_block (`int`,  *optional*, defaults to 1):
                       The number of layers of every block.
                   act_fn (`str`, *optional*, defaults to silu):
                       The activation function.
                   norm_num_groups (`int`, *optional*, defaults to 32):
                       The number of groups for norm operation.
                   lora_pre_down_block_types (Tuple[str], *optional*):
                       The block'types for pre down-block.
                   lora_pre_down_layers_per_block (`int`, *optional*, defaults to 1)
                       The number of layers of every pre down-block block.
                   lora_pre_conv_skipped ('bool', *optional*, defaults to False )
                       Set to True to skip conv in pre downsample.
                   lora_pre_conv_types (Tuple[str], *optional*):
                       The block'types for pre conv.
                   lora_pre_conv_layers_per_block (`int`, *optional*, defaults to 1)
                       The number of layers of every pre conv block.
                   lora_pre_conv_layers_kernel_size (`int`, *optional*, defaults to 1)
                       The conv kernel size of pre conv block.
                   lora_block_in_channels (Tuple[int],  *optional*, defaults to (256, 256, 256, 256)):
                       The number of input channels for lora block.
                   lora_block_out_channels (Tuple[int],  *optional*, defaults to (256, 256, 256, 256)):
                       The number of output channels for lora block.
                   lora_rank (int,  *optional*, defaults to 4):
                       The rank of lora block.
                   lora_control_rank (int,  *optional*, defaults to 4):
                       The rank of lora block.
                   lora_post_add (`bool`,  *optional*, defaults to False):
                        Set to `True`, conduct weighted adding operation after lora.
                   lora_concat_hidden (`bool`,  *optional*, defaults to False):
                        Set to `True`, conduct concat operation for hidden embedding.
                   lora_control_channels  (Tuple[int],  *optional*, defaults to (None, None, None, None)):
                        The number of control channels.
                   lora_control_self_add (`bool`,  *optional*, defaults to True):
                        Set to `True` to perform self attn add.
                   lora_key_states_skipped (`bool`, *optional*, defaults to False):
                        Set to `True` for skip to perform lora on key value.
                    value_states_skipped (`bool`, *optional*, defaults to False):
                        Set to `True` for skip to perform lora on value.
                    output_states_skipped (`bool`, *optional*, defaults to False):
                        Set to `True` for skip to perform lora on output value.
                    lora_control_version (int,  *optional*, defaults to 1):
                        Use lora attn version: ControlLoRACrossAttnProcessor vs ControlLoRACrossAttnProcessorV2.
               """

        super().__init__()
        lora_control_cls = ControlLoRACrossAttnProcessor
        if lora_control_version == 2:
            lora_control_cls = ControlLoRACrossAttnProcessorV2

        assert lora_block_in_channels[0] == block_out_channels[-1]

        if lora_pre_conv_skipped:
            lora_control_channels = lora_block_in_channels
            lora_control_self_add = False

        self.layers_per_block = layers_per_block
        self.lora_pre_down_layers_per_block = lora_pre_down_layers_per_block
        self.lora_pre_conv_layers_per_block = lora_pre_conv_layers_per_block

        self.conv_in = torch.nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1)

        self.down_blocks = nn.ModuleList([])
        self.pre_lora_layers = nn.ModuleList([])
        self.lora_layers = nn.ModuleList([])

        # pre_down
        pre_down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            pre_down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            pre_down_blocks.append(pre_down_block)
        self.down_blocks.append(nn.Sequential(*pre_down_blocks))
        self.pre_lora_layers.append(
            get_down_block(
                lora_pre_conv_types[0],
                num_layers=self.lora_pre_conv_layers_per_block,
                in_channels=lora_block_in_channels[0],
                out_channels=(
                    lora_block_out_channels[0] if lora_control_channels[0] is
                    None else lora_control_channels[0]),
                add_downsample=False,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                resnet_kernel_size=lora_pre_conv_layers_kernel_size,
            ) if not lora_pre_conv_skipped else nn.Identity())
        self.lora_layers.append(
            nn.ModuleList([
                lora_control_cls(
                    lora_block_out_channels[0],
                    cross_attention_dim=cross_attention_dim,
                    rank=lora_rank,
                    control_rank=lora_control_rank,
                    post_add=lora_post_add,
                    concat_hidden=lora_concat_hidden,
                    control_channels=lora_control_channels[0],
                    control_self_add=lora_control_self_add,
                    key_states_skipped=lora_key_states_skipped,
                    value_states_skipped=lora_value_states_skipped,
                    output_states_skipped=lora_output_states_skipped)
                for cross_attention_dim in lora_cross_attention_dims[0]
            ]))

        # down
        output_channel = lora_block_in_channels[0]
        for i, down_block_type in enumerate(lora_pre_down_block_types):
            if i == 0:
                continue
            input_channel = output_channel
            output_channel = lora_block_in_channels[i]

            down_block = get_down_block(
                down_block_type,
                num_layers=self.lora_pre_down_layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=True,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

            self.pre_lora_layers.append(
                get_down_block(
                    lora_pre_conv_types[i],
                    num_layers=self.lora_pre_conv_layers_per_block,
                    in_channels=output_channel,
                    out_channels=(
                        lora_block_out_channels[i] if lora_control_channels[i]
                        is None else lora_control_channels[i]),
                    add_downsample=False,
                    resnet_eps=1e-6,
                    downsample_padding=0,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attn_num_head_channels=None,
                    temb_channels=None,
                    resnet_kernel_size=lora_pre_conv_layers_kernel_size,
                ) if not lora_pre_conv_skipped else nn.Identity())
            self.lora_layers.append(
                nn.ModuleList([
                    lora_control_cls(
                        lora_block_out_channels[i],
                        cross_attention_dim=cross_attention_dim,
                        rank=lora_rank,
                        control_rank=lora_control_rank,
                        post_add=lora_post_add,
                        concat_hidden=lora_concat_hidden,
                        control_channels=lora_control_channels[i],
                        control_self_add=lora_control_self_add,
                        key_states_skipped=lora_key_states_skipped,
                        value_states_skipped=lora_value_states_skipped,
                        output_states_skipped=lora_output_states_skipped)
                    for cross_attention_dim in lora_cross_attention_dims[i]
                ]))

    def forward(self,
                x: torch.FloatTensor,
                return_dict: bool = True) -> Union[ControlLoRAOutput, Tuple]:
        lora_layer: ControlLoRACrossAttnProcessor

        orig_dtype = x.dtype
        dtype = self.conv_in.weight.dtype

        h = x.to(dtype)
        h = self.conv_in(h)
        control_states_list = []

        # down
        for down_block, pre_lora_layer, lora_layer_list in zip(
                self.down_blocks, self.pre_lora_layers, self.lora_layers):
            h = down_block(h)
            control_states = pre_lora_layer(h)
            if isinstance(control_states, tuple):
                control_states = control_states[0]
            control_states = control_states.to(orig_dtype)
            for lora_layer in lora_layer_list:
                lora_layer.inject_control_states(control_states)
            control_states_list.append(control_states)

        if not return_dict:
            return tuple(control_states_list)

        return ControlLoRAOutput(control_states=tuple(control_states_list))


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift='default',
    resnet_kernel_size=3,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith(
        'UNetRes') else down_block_type
    if down_block_type == 'SimpleDownEncoderBlock2D':
        return SimpleDownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            convnet_eps=resnet_eps,
            convnet_act_fn=resnet_act_fn,
            convnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            convnet_time_scale_shift=resnet_time_scale_shift,
            convnet_kernel_size=resnet_kernel_size)
    else:
        return get_down_block_default(
            down_block_type,
            num_layers,
            in_channels,
            out_channels,
            temb_channels,
            add_downsample,
            resnet_eps,
            resnet_act_fn,
            attn_num_head_channels,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            downsample_padding=downsample_padding,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            # resnet_kernel_size=resnet_kernel_size
        )


class ControlLoRACrossAttnProcessor(LoRACrossAttnProcessor):

    def __init__(self,
                 hidden_size,
                 cross_attention_dim=None,
                 rank=4,
                 control_rank=None,
                 post_add=False,
                 concat_hidden=False,
                 control_channels=None,
                 control_self_add=True,
                 key_states_skipped=False,
                 value_states_skipped=False,
                 output_states_skipped=False,
                 **kwargs):
        super().__init__(
            hidden_size,
            cross_attention_dim,
            rank,
            post_add=post_add,
            key_states_skipped=key_states_skipped,
            value_states_skipped=value_states_skipped,
            output_states_skipped=output_states_skipped)

        control_rank = rank if control_rank is None else control_rank
        control_channels = hidden_size if control_channels is None else control_channels
        self.concat_hidden = concat_hidden
        self.control_self_add = control_self_add if control_channels is None else False
        self.control_states: torch.Tensor = None

        self.to_control = LoRALinearLayer(
            control_channels + (hidden_size if concat_hidden else 0),
            hidden_size, control_rank)
        self.pre_loras: List[LoRACrossAttnProcessor] = []
        self.post_loras: List[LoRACrossAttnProcessor] = []

    def inject_pre_lora(self, lora_layer):
        self.pre_loras.append(lora_layer)

    def inject_post_lora(self, lora_layer):
        self.post_loras.append(lora_layer)

    def inject_control_states(self, control_states):
        self.control_states = control_states

    def process_control_states(self, hidden_states, scale=1.0):
        control_states = self.control_states.to(hidden_states.dtype)
        if hidden_states.ndim == 3 and control_states.ndim == 4:
            batch, _, height, width = control_states.shape
            control_states = control_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, -1)
            self.control_states = control_states
        _control_states = control_states
        if self.concat_hidden:
            b1, b2 = control_states.shape[0], hidden_states.shape[0]
            if b1 != b2:
                control_states = control_states[:, None].repeat(
                    1, b2 // b1, *([1] * (len(control_states.shape) - 1)))
                control_states = control_states.view(-1,
                                                     *control_states.shape[2:])
            _control_states = torch.cat([hidden_states, control_states], -1)
        _control_states = scale * self.to_control(_control_states)
        if self.control_self_add:
            control_states = control_states + _control_states
        else:
            control_states = _control_states

        return control_states

    def __call__(self,
                 attn: CrossAttention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 scale=1.0):
        pre_lora: LoRACrossAttnProcessor
        post_lora: LoRACrossAttnProcessor
        assert self.control_states is not None

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length)
        query = attn.to_q(hidden_states)
        for pre_lora in self.pre_loras:
            lora_in = query if pre_lora.post_add else hidden_states
            if isinstance(pre_lora, ControlLoRACrossAttnProcessor):
                lora_in = lora_in + pre_lora.process_control_states(
                    hidden_states, scale)
            query = query + scale * pre_lora.to_q_lora(lora_in)
        query = query + scale * self.to_q_lora(
            (query if self.post_add else hidden_states)
            + self.process_control_states(hidden_states, scale))
        for post_lora in self.post_loras:
            lora_in = query if post_lora.post_add else hidden_states
            if isinstance(post_lora, ControlLoRACrossAttnProcessor):
                lora_in = lora_in + post_lora.process_control_states(
                    hidden_states, scale)
            query = query + scale * post_lora.to_q_lora(lora_in)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.key_states_skipped:
                key = key + scale * pre_lora.to_k_lora(
                    key if pre_lora.post_add else encoder_hidden_states)
        if not self.key_states_skipped:
            key = key + scale * self.to_k_lora(
                key if self.post_add else encoder_hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.key_states_skipped:
                key = key + scale * post_lora.to_k_lora(
                    key if post_lora.post_add else encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.value_states_skipped:
                value = value + pre_lora.to_v_lora(
                    value if pre_lora.post_add else encoder_hidden_states)
        if not self.value_states_skipped:
            value = value + scale * self.to_v_lora(
                value if self.post_add else encoder_hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.value_states_skipped:
                value = value + post_lora.to_v_lora(
                    value if post_lora.post_add else encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        out = attn.to_out[0](hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.output_states_skipped:
                out = out + scale * pre_lora.to_out_lora(
                    out if pre_lora.post_add else hidden_states)
        out = out + scale * self.to_out_lora(
            out if self.post_add else hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.output_states_skipped:
                out = out + scale * post_lora.to_out_lora(
                    out if post_lora.post_add else hidden_states)
        hidden_states = out
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class ControlLoRACrossAttnProcessorV2(LoRACrossAttnProcessor):

    def __init__(self,
                 hidden_size,
                 cross_attention_dim=None,
                 rank=4,
                 control_rank=None,
                 control_channels=None,
                 **kwargs):
        super().__init__(
            hidden_size,
            cross_attention_dim,
            rank,
            post_add=False,
            key_states_skipped=True,
            value_states_skipped=True,
            output_states_skipped=False)

        control_rank = rank if control_rank is None else control_rank
        control_channels = hidden_size if control_channels is None else control_channels
        self.concat_hidden = True
        self.control_self_add = False
        self.control_states: torch.Tensor = None

        self.to_control = LoRALinearLayer(hidden_size + control_channels,
                                          hidden_size, control_rank)
        self.to_control_out = LoRALinearLayer(hidden_size + control_channels,
                                              hidden_size, control_rank)
        self.pre_loras: List[LoRACrossAttnProcessor] = []
        self.post_loras: List[LoRACrossAttnProcessor] = []

    def inject_pre_lora(self, lora_layer):
        self.pre_loras.append(lora_layer)

    def inject_post_lora(self, lora_layer):
        self.post_loras.append(lora_layer)

    def inject_control_states(self, control_states):
        self.control_states = control_states

    def process_control_states(self, hidden_states, scale=1.0, is_out=False):
        control_states = self.control_states.to(hidden_states.dtype)
        if hidden_states.ndim == 3 and control_states.ndim == 4:
            batch, _, height, width = control_states.shape
            control_states = control_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, -1)
            self.control_states = control_states
        _control_states = control_states
        if self.concat_hidden:
            b1, b2 = control_states.shape[0], hidden_states.shape[0]
            if b1 != b2:
                control_states = control_states[:, None].repeat(
                    1, b2 // b1, *([1] * (len(control_states.shape) - 1)))
                control_states = control_states.view(-1,
                                                     *control_states.shape[2:])
            _control_states = torch.cat([hidden_states, control_states], -1)
        _control_states = scale * (self.to_control_out
                                   if is_out else self.to_control)(
                                       _control_states)
        if self.control_self_add:
            control_states = control_states + _control_states
        else:
            control_states = _control_states

        return control_states

    def __call__(self,
                 attn: CrossAttention,
                 hidden_states,
                 encoder_hidden_states=None,
                 attention_mask=None,
                 scale=1.0):
        pre_lora: LoRACrossAttnProcessor
        post_lora: LoRACrossAttnProcessor
        assert self.control_states is not None

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length)
        for pre_lora in self.pre_loras:
            if isinstance(pre_lora, ControlLoRACrossAttnProcessorV2):
                hidden_states = hidden_states + pre_lora.process_control_states(
                    hidden_states, scale)
        hidden_states = hidden_states + self.process_control_states(
            hidden_states, scale)
        for post_lora in self.post_loras:
            if isinstance(post_lora, ControlLoRACrossAttnProcessorV2):
                hidden_states = hidden_states + post_lora.process_control_states(
                    hidden_states, scale)
        query = attn.to_q(hidden_states)
        for pre_lora in self.pre_loras:
            lora_in = query if pre_lora.post_add else hidden_states
            query = query + scale * pre_lora.to_q_lora(lora_in)
        query = query + scale * self.to_q_lora(
            query if self.post_add else hidden_states)
        for post_lora in self.post_loras:
            lora_in = query if post_lora.post_add else hidden_states
            query = query + scale * post_lora.to_q_lora(lora_in)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.key_states_skipped:
                key = key + scale * pre_lora.to_k_lora(
                    key if pre_lora.post_add else encoder_hidden_states)
        if not self.key_states_skipped:
            key = key + scale * self.to_k_lora(
                key if self.post_add else encoder_hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.key_states_skipped:
                key = key + scale * post_lora.to_k_lora(
                    key if post_lora.post_add else encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.value_states_skipped:
                value = value + pre_lora.to_v_lora(
                    value if pre_lora.post_add else encoder_hidden_states)
        if not self.value_states_skipped:
            value = value + scale * self.to_v_lora(
                value if self.post_add else encoder_hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.value_states_skipped:
                value = value + post_lora.to_v_lora(
                    value if post_lora.post_add else encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        for pre_lora in self.pre_loras:
            if isinstance(pre_lora, ControlLoRACrossAttnProcessorV2):
                hidden_states = hidden_states + pre_lora.process_control_states(
                    hidden_states, scale, is_out=True)
        hidden_states = hidden_states + self.process_control_states(
            hidden_states, scale, is_out=True)
        for post_lora in self.post_loras:
            if isinstance(post_lora, ControlLoRACrossAttnProcessorV2):
                hidden_states = hidden_states + post_lora.process_control_states(
                    hidden_states, scale, is_out=True)
        out = attn.to_out[0](hidden_states)
        for pre_lora in self.pre_loras:
            if not pre_lora.output_states_skipped:
                out = out + scale * pre_lora.to_out_lora(
                    out if pre_lora.post_add else hidden_states)
        out = out + scale * self.to_out_lora(
            out if self.post_add else hidden_states)
        for post_lora in self.post_loras:
            if not post_lora.output_states_skipped:
                out = out + scale * post_lora.to_out_lora(
                    out if post_lora.post_add else hidden_states)
        hidden_states = out
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class ConvBlock2D(nn.Module):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_kernel_size=3,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity='swish',
        time_embedding_norm='default',
        kernel=None,
        output_scale_factor=1.0,
        up=False,
        down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=conv_kernel_size,
            stride=1,
            padding=conv_kernel_size // 2)

        if temb_channels is not None:
            if self.time_embedding_norm == 'default':
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == 'scale_shift':
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f'unknown time_embedding_norm : {self.time_embedding_norm} '
                )

            self.time_emb_proj = torch.nn.Linear(temb_channels,
                                                 time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out,
            num_channels=out_channels,
            eps=eps,
            affine=True)
        self.dropout = torch.nn.Dropout(dropout)

        if non_linearity == 'swish':
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == 'mish':
            self.nonlinearity = Mish()
        elif non_linearity == 'silu':
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == 'fir':
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == 'sde_vp':
                self.upsample = partial(
                    F.interpolate, scale_factor=2.0, mode='nearest')
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == 'fir':
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == 'sde_vp':
                self.downsample = partial(
                    F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name='op')

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes.
            # see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            _ = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            _ = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None,
                                                               None]

        if temb is not None and self.time_embedding_norm == 'default':
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == 'scale_shift':
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        output_tensor = self.dropout(hidden_states)

        return output_tensor


class SimpleDownEncoderBlock2D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        convnet_eps: float = 1e-6,
        convnet_time_scale_shift: str = 'default',
        convnet_act_fn: str = 'swish',
        convnet_groups: int = 32,
        convnet_pre_norm: bool = True,
        convnet_kernel_size: int = 3,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        convnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            convnets.append(
                ConvBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=convnet_eps,
                    groups=convnet_groups,
                    dropout=dropout,
                    time_embedding_norm=convnet_time_scale_shift,
                    non_linearity=convnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=convnet_pre_norm,
                    conv_kernel_size=convnet_kernel_size,
                ))
        in_channels = in_channels if num_layers == 0 else out_channels

        self.convnets = nn.ModuleList(convnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    in_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name='op')
            ])
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for convnet in self.convnets:
            hidden_states = convnet(hidden_states, temb=None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states
