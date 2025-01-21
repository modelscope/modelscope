# Copyright 2021-2023 The Alibaba DAMO mPLUG Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch MPLUG OWL model. """

import copy
import logging
import math
import os
import os.path as osp
import random
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions)
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.auto import AutoModelForCausalLM
from transformers.utils import ModelOutput

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.mplug_owl.configuration_mplug_owl import (
    MplugOwlConfig, MplugOwlVisionConfig, MplugOwlVisualAbstractorConfig)
from modelscope.outputs import OutputKeys
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['MplugOwlForConditionalGeneration']


@dataclass
class MplugOwlForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`MPlugOwlForConditionalGeneration`].

    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.

        language_model_outputs (`CausalLMOutputWithPast`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    vision_outputs: Optional[torch.FloatTensor] = None
    language_model_outputs: Optional[Tuple[torch.FloatTensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ['vision_outputs', 'language_model_outputs'
                                 ] else getattr(self, k).to_tuple()
            for k in self.keys())


# Hack for bloomz
def bloom_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                    ...]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **deprecated_arguments,
) -> Union[Tuple[torch.Tensor, ...],
           BaseModelOutputWithPastAndCrossAttentions]:
    if len(deprecated_arguments) > 0:
        raise ValueError(f'Got unexpected arguments: {deprecated_arguments}')

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else
        self.config.output_hidden_states)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            'You cannot specify both input_ids and inputs_embeds at the same time'
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError(
            'You have to specify either input_ids or inputs_embeds')

    if past_key_values is None:
        past_key_values = tuple([None] * len(self.h))

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape batch_size x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = self.word_embeddings_layernorm(inputs_embeds)

    hidden_states = inputs_embeds

    presents = () if use_cache else None
    all_self_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    # Compute alibi tensor: check build_alibi_tensor documentation
    seq_length_with_past = seq_length
    past_key_values_length = 0
    if past_key_values[0] is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length
    if attention_mask is None:
        attention_mask = torch.ones((batch_size, seq_length_with_past),
                                    device=hidden_states.device)
    else:
        attention_mask = attention_mask.to(hidden_states.device)

    alibi = self.build_alibi_tensor(
        attention_mask, self.num_heads, dtype=hidden_states.dtype)

    causal_mask = self._prepare_attn_mask(
        attention_mask,
        input_shape=(batch_size, seq_length),
        past_key_values_length=past_key_values_length,
    )

    for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):

                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(
                        *inputs,
                        use_cache=use_cache,
                        output_attentions=output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                alibi,
                causal_mask,
                layer_past,
                head_mask[i],
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1], )

        if output_attentions:
            all_self_attentions = all_self_attentions + (
                outputs[2 if use_cache else 1], )

    # Add last hidden state
    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states, )

    if not return_dict:
        return tuple(
            v for v in
            [hidden_states, presents, all_hidden_states, all_self_attentions]
            if v is not None)

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def get_ltor_masks_and_position_ids_from_embeddings(data):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()[:2]

    # Attention mask (lower triangular).
    att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length),
                   device=data.device)).view(att_mask_batch, 1, seq_length,
                                             seq_length)

    # Loss mask.
    loss_mask = torch.ones(
        data.size()[:2], dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data[..., 0])

    # Convert attention mask to binary:
    attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids


class MplugOwlVisionEmbeddings(nn.Module):

    def __init__(self, config: MplugOwlVisionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_size))

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False)

        self.num_patches = (self.image_size // self.patch_size)**2

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.hidden_size))

        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.pre_layernorm = layernorm_func(
            self.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.size(0)
        image_embeds = self.patch_embed(pixel_values)
        image_embeds = image_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.cls_token.expand(batch_size, 1,
                                             -1).to(image_embeds.dtype)
        embeddings = torch.cat([class_embeds, image_embeds], dim=1)
        embeddings = embeddings + \
            self.position_embedding[:, : embeddings.size(1)].to(
                image_embeds.dtype)
        embeddings = self.pre_layernorm(embeddings)
        return embeddings


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        output = torch.nn.functional.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(x)


class MplugOwlVisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:'
                f' {self.num_heads}).')
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        self.query_key_value = nn.Linear(self.hidden_size,
                                         3 * self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, seq_len, embed_dim = hidden_states.size()

        mixed_qkv = self.query_key_value(hidden_states)

        mixed_qkv = mixed_qkv.reshape(bsz, seq_len, self.num_heads, 3,
                                      embed_dim // self.num_heads).permute(
                                          3, 0, 2, 1, 4)  # [3, b, np, sq, hn]
        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_states,
                                        key_states.transpose(-1, -2))

        attention_scores = attention_scores * self.scale

        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs,
                                     value_states).permute(0, 2, 1, 3)

        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size, )
        context_layer = context_layer.reshape(new_context_layer_shape)

        output = self.dense(context_layer)

        outputs = (output, attention_probs) if output_attentions else (output,
                                                                       None)

        return outputs


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MplugOwlMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MplugOwlVisionEncoderLayer(nn.Module):

    def __init__(self, config: MplugOwlVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MplugOwlVisionAttention(config)
        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.input_layernorm = layernorm_func(
            self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MplugOwlMLP(config)
        self.post_attention_layernorm = layernorm_func(
            self.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + residual

        outputs = (hidden_states, )

        if output_attentions:
            outputs += (attn_weights, )

        return outputs


class MplugOwlPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MplugOwlConfig
    base_model_prefix = 'mplug_owl'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r'position_ids',
        r'language_model.encoder.embed_tokens.weight',
        r'language_model.decoder.embed_tokens.weight',
        r'language_model.lm_head.weight',
    ]
    _no_split_modules = ['MplugOwlAttention']
    _keep_in_fp32_modules = ['wo']

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_range
        if isinstance(module, nn.Conv2d) or isinstance(
                module, nn.Embedding) or isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=factor)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, MplugOwlVisionEmbeddings):
            if hasattr(self.config, 'vision_config'):
                factor = self.config.vision_config.initializer_range
            nn.init.trunc_normal_(
                module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=factor)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.Parameter):
            nn.init.trunc_normal_(module.data, mean=0.0, std=factor)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MplugOwlVisionEncoder):
            module.gradient_checkpointing = value


MPLUG_OWL_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MplugOwlConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MPLUG_OWL_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`MplugOwlPreprocessor`].
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MPLUG_OWL_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it. Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

MPLUG_OWL_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`MplugOwlPreprocessor`].
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary of the language model. Input tokens can optionally be
            provided to serve as text prompt, which the language model can continue.

            Indices can be obtained using [`MplugOwlPreprocessor`]. See [`MplugOwlPreprocessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary of the language model. Only relevant in case an
            encoder-decoder language model (like T5) is used.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details. [What are decoder input IDs?](../glossary#decoder-input-ids)

        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.

            Only relevant in case an encoder-decoder language model (like T5) is used.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class MplugOwlVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`MplugOwlVisionEncoderLayer`].

    Args:
        config (`MplugOwlVisionConfig`):
            The corresponding vision configuration for the `MplugOwlEncoder`.
    """

    def __init__(self, config: MplugOwlVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MplugOwlVisionEncoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states, )
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states, )

        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_states, all_attentions]
                if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions)


class MplugOwlVisionModel(MplugOwlPreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = MplugOwlVisionConfig

    def __init__(self, config: MplugOwlVisionConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size

        self.embeddings = MplugOwlVisionEmbeddings(config)
        self.encoder = MplugOwlVisionEncoder(config)
        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.post_layernorm = layernorm_func(
            self.hidden_size, eps=config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.embeddings


class MplugOwlVisualAbstractorMLP(nn.Module):

    def __init__(self, config: MplugOwlVisualAbstractorConfig):
        super().__init__()
        self.config = config
        in_features = config.hidden_size
        hidden_features = config.intermediate_size
        hidden_features = int(2 * hidden_features / 3)
        multiple_of = 256
        hidden_features = multiple_of * \
            ((hidden_features + multiple_of - 1) // multiple_of)
        self.act = nn.SiLU()

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)
        self.w3 = nn.Linear(in_features, hidden_features)
        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.ffn_ln = layernorm_func(
            hidden_features, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act(
            self.w1(hidden_states)) * self.w3(hidden_states)
        hidden_states = self.ffn_ln(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


class MplugOwlVisualAbstractorMultiHeadAttention(nn.Module):

    def __init__(self, config: MplugOwlVisualAbstractorConfig):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                % (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size
                                       / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(
            self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )

        outputs = outputs + (past_key_value, )
        return outputs


class MplugOwlVisualAbstractorCrossOutput(nn.Module):

    def __init__(self, config: MplugOwlVisualAbstractorConfig):
        super().__init__()
        dim = config.hidden_size
        self.out_proj = nn.Linear(dim, dim, bias=True)
        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.norm2 = layernorm_func(dim)
        self.mlp = MplugOwlVisualAbstractorMLP(config)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor + self.out_proj(hidden_states)
        input_tensor = input_tensor + self.mlp(self.norm2(input_tensor))
        return input_tensor


class MplugOwlVisualAbstractorAttention(nn.Module):

    def __init__(self, config: MplugOwlVisualAbstractorConfig):
        super().__init__()
        self.attention = MplugOwlVisualAbstractorMultiHeadAttention(config)
        self.output = MplugOwlVisualAbstractorCrossOutput(config)
        self.pruned_heads = set()
        layernorm_func = LayerNormFp32 if config.use_fp32_layernorm else nn.LayerNorm
        self.norm1 = layernorm_func(config.hidden_size)
        self.normk = layernorm_func(config.hidden_size)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads,
            self.attention.attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(
            self.output.out_proj, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - \
            len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * \
            self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # HACK we apply norm on q and k
        hidden_states = self.norm1(hidden_states)
        encoder_hidden_states = self.normk(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [hidden_states, encoder_hidden_states], dim=1)
        encoder_attention_mask = torch.cat(
            [attention_mask, encoder_attention_mask], dim=-1)
        self_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        # add attentions if we output them
        outputs = (attention_output, ) + self_outputs[1:]
        return outputs


class MplugOwlVisualAbstractorLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        self.layer_idx = layer_idx

        self.crossattention = MplugOwlVisualAbstractorAttention(config)
        self.has_cross_attention = True

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        if encoder_hidden_states is None:
            raise ValueError(
                'encoder_hidden_states must be given for cross-attention layers'
            )
        cross_attention_outputs = self.crossattention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]

        outputs = (query_attention_output, )
        return outputs


class MplugOwlVisualAbstractorEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MplugOwlVisualAbstractorLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None

        for i in range(self.config.num_hidden_layers):
            layer_module = self.layers[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None

            if getattr(self.config, 'gradient_checkpointing',
                       False) and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value,
                                      output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]

        return BaseModelOutput(last_hidden_state=hidden_states, )


class MplugOwlVisualAbstractorModel(MplugOwlPreTrainedModel):

    def __init__(self, config: MplugOwlVisualAbstractorConfig,
                 language_hidden_size):
        super().__init__(config)
        self.config = config

        self.encoder = MplugOwlVisualAbstractorEncoder(config)
        self.visual_fc = torch.nn.Linear(config.hidden_size,
                                         language_hidden_size)
        self.vit_eos = torch.nn.Parameter(
            torch.randn(1, 1, language_hidden_size))
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.
            device: (`torch.device`):
                The device of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                'Wrong shape for input_ids (shape {}) or attention_mask (shape {})'
                .format(input_shape, attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(
        self,
        query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors:
            shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`): Contains precomputed key and
            value hidden states of the attention blocks. Can be used to speed up decoding. If `past_key_values` are
            used, the user can optionally input only the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead of all `decoder_input_ids` of shape
            `(batch_size, sequence_length)`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = query_embeds
        input_shape = embedding_output.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding_output.device

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask is None:
            attention_mask = torch.ones(
                (query_embeds.shape[0], query_embeds.shape[1]),
                dtype=torch.long,
                device=query_embeds.device)
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0].size()
            else:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)

            if type(encoder_attention_mask) == list:
                encoder_extended_attention_mask = [
                    self.invert_attention_mask(mask)
                    for mask in encoder_attention_mask
                ]
            elif encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]

        sequence_output = self.visual_fc(sequence_output)
        eos_repeat = self.vit_eos.repeat(sequence_output.shape[0], 1, 1)
        sequence_output = torch.cat([sequence_output, eos_repeat], dim=1)

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


class MplugOwlModel(MplugOwlPreTrainedModel):
    r"""The mPLUG-Owl model is a multi-modal conversation model that support various modalities as input.
    mPLUG-Owl consists a visual encoder, a visual abstrator module and a language decoder model, which enables
    both image and text input.
    This model is implemented base on mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality.
    `Paper <https://arxiv.org/abs/2304.14178>`.
    """
    config_class = MplugOwlConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: MplugOwlConfig):
        super().__init__(config)

        self.vision_model = MplugOwlVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens,
                        config.visual_abstractor_config.hidden_size))
        self.abstractor = MplugOwlVisualAbstractorModel(
            config.visual_abstractor_config, config.text_config.hidden_size)

        # if config.use_decoder_only_language_model:
        language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.language_model = language_model

        if config.text_config.model_type == 'bloom':
            bound_method = bloom_forward.__get__(
                self.language_model.transformer,
                self.language_model.transformer.__class__)
            setattr(self.language_model.transformer, 'forward', bound_method)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(
                input_ids)

            text_outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )

        return text_outputs

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return vision_outputs


def get_media_indices(my_list):
    if isinstance(my_list, torch.Tensor):
        my_list = my_list.cpu().tolist()
    result = []
    for i in range(len(my_list)):
        if i == 0 and my_list[i] < 0:
            result.append(i)
        elif my_list[i] != my_list[i - 1] and my_list[i] < 0:
            result.append(i)
    return result


class MplugOwlForConditionalGenerationHF(MplugOwlPreTrainedModel):
    config_class = MplugOwlConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: MplugOwlConfig, **kwargs):
        super().__init__(config)

        self.vision_model = MplugOwlVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens,
                        config.visual_abstractor_config.hidden_size))
        self.abstractor = MplugOwlVisualAbstractorModel(
            config.visual_abstractor_config, config.text_config.hidden_size)

        # if config.use_decoder_only_language_model:
        language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()
        self.main_input_name = 'input_ids'

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_encoder(self):
        return self.language_model.get_encoder()

    def get_decoder(self):
        return self.language_model.get_decoder()

    def _tie_weights(self):
        if not self.config.use_decoder_only_language_model:
            self.language_model.encoder.embed_tokens = self.language_model.shared
            self.language_model.decoder.embed_tokens = self.language_model.shared

    def _preprocess_accelerate(self):
        r"""
        Some pre-processing hacks to make the model `accelerate` compatible. Check
        https://github.com/huggingface/transformers/pull/21707 for more details.
        """
        hf_device_map = self.hf_device_map

        if len(
                hf_device_map
        ) > 1 and 'language_model' not in hf_device_map and torch.cuda.device_count(
        ) > 1:
            # warn users about unexpected behavior when using multi-GPU + mPLUG-Owl + `accelerate`.
            logger.warning(
                'The `language_model` is not in the `hf_device_map` dictionary and you are running your script'
                ' in a multi-GPU environment. this may lead to unexpected behavior when using `accelerate`.'
                ' Please pass a `device_map` that contains `language_model` to remove this warning.'
                ' Please refer to https://github.com/huggingface/blog/blob/main/accelerate-large-models.md for'
                ' more details on creating a `device_map` for large models.', )

        if hasattr(self.language_model, '_hf_hook'):
            self.language_model._hf_hook.io_same_device = True  # For `generate` compatibility

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        num_images,
        non_padding_mask: Optional[torch.LongTensor] = None,
        non_media_mask: Optional[torch.LongTensor] = None,
        prompt_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MplugOwlForConditionalGenerationModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get text embedding
        text_tokens_ = input_ids
        batch_size = input_ids.shape[0]

        media_token_indices = [
            # [:-1] since we would not use the last token for embedding
            get_media_indices(text_tokens_[i][:-1]) for i in range(batch_size)
        ]
        text_tokens_[text_tokens_ < 0] = 1  # Not used
        text_embeds = self.get_input_embeddings()(
            text_tokens_)  # Temporally Embedding

        if pixel_values is not None:
            pixel_values = pixel_values.half()
            image_embeds = self.vision_model(
                pixel_values, return_dict=True).last_hidden_state

            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1,
                                                    -1)

            query_features = self.abstractor(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
            )['last_hidden_state']
            img_seq_length = query_features.shape[1]

        num_images_per_sample = num_images.long().cpu().tolist()

        text_chunk_embeds = []
        img_idx = 0
        for b in range(batch_size):
            start = 0
            result = []
            if len(media_token_indices[b]) > 0:
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(text_embeds[b, start:pos])
                    result.append(query_features[img_idx + i])
                    start = pos + img_seq_length
            if start < text_embeds.shape[1]:
                result.append(text_embeds[b, start:])

            img_idx += num_images_per_sample[b]
            text_chunk_embeds.append(torch.cat(result, dim=0))

        # Actual Input Embeddings
        input_embeds = torch.stack(text_chunk_embeds, dim=0)

        # Create causal mask and position ids
        _, loss_mask, position_ids = \
            get_ltor_masks_and_position_ids_from_embeddings(input_embeds)

        # Calculate the loss_mask
        non_padding_mask = non_padding_mask.long()
        non_media_mask = non_media_mask.long()
        prompt_mask = prompt_mask.long()  # TODO How to deal with prompt mask
        loss_mask = loss_mask[:, :-1]

        loss_mask = loss_mask * non_padding_mask * non_media_mask * prompt_mask

        # Forward into GPT
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        outputs.loss = (outputs.loss
                        * loss_mask.view(-1)).sum() / loss_mask.sum()
        return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        if input_ids is not None:
            batch_size = input_ids.size(0)
            media_token_indices = [
                get_media_indices(input_ids[i]) for i in range(batch_size)
            ]
            num_images_per_sample = [len(x) for x in media_token_indices]
            input_ids = input_ids.clone()
            input_ids[input_ids < 0] = 0  # Not used

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).long().to(
                input_ids.device)

        if hasattr(self, 'hf_device_map'):
            # preprocess for `accelerate`
            self._preprocess_accelerate()
        batch_size = input_ids.shape[0]
        # get text embedding
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # get visual embedding
        if pixel_values is not None:
            pixel_values = pixel_values.half()
            pixel_values = pixel_values.to(input_ids.device)
            with torch.no_grad():
                image_embeds = self.vision_model(
                    pixel_values, return_dict=True).last_hidden_state
                image_attention_mask = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=image_embeds.device)
                query_tokens = self.query_tokens.expand(
                    image_embeds.shape[0], -1, -1)
                query_outputs = self.abstractor(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
                query_output = query_outputs['last_hidden_state']
                image_embeds = query_output
            img_seq_length = image_embeds.shape[1]

            # ===================
            # Get actual input embeddings
            # ===================
            text_chunk_embeds = []
            text_chunk_attns = []
            img_idx = 0

            for b in range(batch_size):
                start = 0
                result = []
                result_attn = []
                for i, pos in enumerate(media_token_indices[b]):
                    if pos > start:
                        result.append(inputs_embeds[b, start:pos])
                        result_attn.append(attention_mask[b, start:pos])
                    result.append(image_embeds[img_idx + i])
                    result_attn.append(
                        torch.ones(
                            image_embeds[img_idx + i].shape[0],
                            device=inputs_embeds.device))
                    start = pos + img_seq_length
                if start < inputs_embeds.shape[1]:
                    result.append(inputs_embeds[b, start:])
                    result_attn.append(attention_mask[b, start:])

                img_idx += num_images_per_sample[b]
                text_chunk_embeds.append(torch.cat(result, dim=0))
                text_chunk_attns.append(torch.cat(result_attn, dim=0))
            inputs_embeds = torch.stack(text_chunk_embeds, dim=0)
            attention_mask = torch.stack(text_chunk_attns, dim=0)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


@MODELS.register_module(
    Tasks.multimodal_dialogue, module_name=Models.mplug_owl)
class MplugOwlForConditionalGeneration(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the mPLUG-Owl model from the `model_dir` path.
        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        self.model = MplugOwlForConditionalGenerationHF.from_pretrained(
            model_dir,
            torch_dtype=torch.half,
        )

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        output = self.model.generate(**input)
        return output
