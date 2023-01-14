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
"""PyTorch MPLUG model. """

import math
import os
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch import Tensor, device, nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertTokenizer
from transformers.activations import ACT2FN
from transformers.file_utils import (add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward,
                                     replace_return_docstrings)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions)
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.utils import logging

from modelscope.models.multi_modal.mplug.configuration_mplug import (
    HiTeAConfig, MPlugConfig)
from modelscope.models.multi_modal.mplug.mvit import MViTv2, MViTv2_Base_config
from modelscope.models.multi_modal.mplug.predictor import TextGenerator
from modelscope.utils.constant import ModelFile

transformers.logging.set_verbosity_error()

logger = logging.get_logger()

CONFIG_NAME = 'config.yaml'

_CONFIG_FOR_DOC = 'BertConfig'
_TOKENIZER_FOR_DOC = 'BertTokenizer'


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            'Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see '
            'https://www.tensorflow.org/install/ for installation instructions.'
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info('Converting TensorFlow checkpoint from {}'.format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info('Loading TF weight {} with shape {}'.format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in [
                'adam_v', 'adam_m', 'AdamWeightDecayOptimizer',
                'AdamWeightDecayOptimizer_1', 'global_step'
        ] for n in name):
            logger.info('Skipping {}'.format('/'.join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                scope_names = re.split(r'_(\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == 'kernel' or scope_names[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info('Skipping {}'.format('/'.join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info('Initialize PyTorch weight {}'.format(name))
        pointer.data = torch.from_numpy(array)
    return model


def clamp_inf(tensor):
    if tensor.dtype == torch.float16 and torch.isinf(tensor).any():
        clamp_value = torch.finfo(tensor.dtype).max - 1000
        tensor = torch.clamp(tensor, min=-clamp_value, max=clamp_value)
    return tensor


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               'absolute')

        self.config = config

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:,
                                             past_key_values_length:seq_length
                                             + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention '
                'heads (%d)' %
                (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size
                                       / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               'absolute')
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size)
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
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = clamp_inf(attention_scores)
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    'bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
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


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads,
            self.self.attention_head_size, self.pruned_heads)

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = clamp_inf(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = clamp_inf(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FusionLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.stride_layer = getattr(self.config, 'stride_layer', 100)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)

        self.crossattention = BertAttention(config, is_cross_attention=True)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_nums=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        if layer_nums == 0 or layer_nums % self.stride_layer != 0:
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]

            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
            assert encoder_hidden_states is not None, 'encoder_hidden_states must be given for cross-attention layers'

            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[
                1:-1]  # add cross attentions if we output attention weights
        elif layer_nums != 0 and layer_nums % self.stride_layer == 0:
            self_attention_outputs = self.attention(
                torch.cat([encoder_hidden_states, hidden_states], 1),
                torch.cat([encoder_attention_mask, attention_mask], 3),
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]

            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs

        outputs = outputs + (present_key_value[0], present_key_value[1])

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)

        self.has_cross_attention = getattr(self.config, 'add_cross_attention',
                                           False)
        if self.has_cross_attention:
            self.crossattention = BertAttention(
                config, is_cross_attention=True)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if self.has_cross_attention:
            assert encoder_hidden_states is not None, 'encoder_hidden_states must be given for cross-attention layers'

            if type(encoder_hidden_states) == list:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states[(self.layer_num
                                           - self.config.fusion_layer)
                                          % len(encoder_hidden_states)],
                    encoder_attention_mask[(self.layer_num
                                            - self.config.fusion_layer)
                                           % len(encoder_hidden_states)],
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]

            else:
                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[
                    1:
                    -1]  # add cross attentions if we output attention weights
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs

        outputs = outputs + (present_key_value[0], present_key_value[1])

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class FusionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [FusionLayer(config, i) for i in range(config.num_hidden_layers)])
        self.start_layer = max(0,
                               config.num_hidden_layers - config.fusion_layers)

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        next_decoder_cache = () if use_cache else None

        self.stride_layer = getattr(self.config, 'stride_layer', 100)
        image_length = encoder_hidden_states.shape[1]
        text_length = hidden_states.shape[1]

        for i in range(self.start_layer, len(self.layer)):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None

            if getattr(self.config, 'gradient_checkpointing',
                       False) and self.training:
                if use_cache:
                    logger.warning(
                        '`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting '
                        '`use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return tuple(
                            module(*inputs, past_key_value, output_attentions))

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    i - self.start_layer,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    i - self.start_layer,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1], )
            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1], )
            if hidden_states.shape[1] == (image_length + text_length):
                encoder_hidden_states_new, hidden_states = torch.split(
                    hidden_states, (image_length, text_length), 1)
                encoder_hidden_states += encoder_hidden_states_new

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        return [encoder_hidden_states, hidden_states]


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
        ) if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        for i in range(len(self.layer)):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None

            if getattr(self.config, 'gradient_checkpointing',
                       False) and self.training:
                if use_cache:
                    logger.warning(
                        '`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting '
                        '`use_cache=False`...')
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return tuple(
                            module(*inputs, past_key_value, output_attentions))

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states,
                    attention_mask, layer_head_mask, encoder_hidden_states,
                    encoder_attention_mask)
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1], )
            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1], )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'bert'
    _keys_to_ignore_on_load_missing = [r'position_ids']

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    'The bare Bert Model transformer outputting raw hidden-states without any specific head on top.',
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint='bert-base-uncased',
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def get_extended_attention_mask(self, attention_mask: Tensor,
                                    input_shape: Tuple[int], device: device,
                                    is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[
                        1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:,
                                                      None, :, :] * attention_mask[:,
                                                                                   None,
                                                                                   None, :]
            else:
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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
    ):
        r"""
        encoder_hidden_states
        (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values
        (:obj:`tuple(tuple(torch.FloatTensor))` of length
         :obj:`config.n_layers` with each tuple having 4 tensors of shape
         :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = encoder_embeds.device
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds or encoder_embeds'
            )

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
                )
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

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class FusionModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = FusionEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(
        # tokenizer_class=_TOKENIZER_FOR_DOC,
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint='bert-base-uncased',
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def get_extended_attention_mask(self, attention_mask: Tensor,
                                    input_shape: Tuple[int], device: device,
                                    is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            # [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[
                        1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len),
                                device=device,
                                dtype=causal_mask.dtype),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:,
                                                      None, :, :] * attention_mask[:,
                                                                                   None,
                                                                                   None, :]
            else:
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

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                is_decoder=False):
        r"""
        encoder_hidden_states
        (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values
        (:obj:`tuple(tuple(torch.FloatTensor))` of length
         :obj:`config.n_layers` with each tuple having 4 tensors of shape
         :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif encoder_embeds is not None:
            input_shape = encoder_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = encoder_embeds.device
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds or encoder_embeds'
            )

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[
            2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                ((batch_size, seq_length + past_key_values_length)),
                device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if encoder_hidden_states is not None:
            if type(encoder_hidden_states) == list:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[
                    0].size()
            else:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
                )
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

        if encoder_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )
        else:
            embedding_output = encoder_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_hidden_states, sequence_output = encoder_outputs
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return [encoder_hidden_states, sequence_output]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning. """,
    BERT_START_DOCSTRING)
class BertLMHeadModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=True,
        reduction='mean',
        soft_labels=None,
        alpha=0,
        return_logits=False,
    ):
        r"""
        encoder_hidden_states
        (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with labels n ``[0, ..., config.vocab_size]``
        past_key_values
        (:obj:`tuple(tuple(torch.FloatTensor))` of length
         :obj:`config.n_layers` with each tuple having 4 tensors of shape
         :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        Returns:
        Example::
            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :
                                                          -1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction)
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))
            lm_loss = lm_loss.view(prediction_scores.size(0), -1).sum(1)

        if soft_labels is not None:
            loss_distill = -torch.sum(
                F.log_softmax(shifted_prediction_scores, dim=1) * soft_labels,
                dim=-1)
            loss_distill = (loss_distill * (labels != -100)).sum(1)
            lm_loss = (1 - alpha) * lm_loss + alpha * loss_distill

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((lm_loss, ) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past=None,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            'input_ids':
            input_ids,
            'attention_mask':
            attention_mask,
            'past_key_values':
            past,
            'encoder_hidden_states':
            model_kwargs.get('encoder_hidden_states', None),
            'encoder_attention_mask':
            model_kwargs.get('encoder_attention_mask', None),
            'is_decoder':
            True,
        }

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past), )
        return reordered_past


class BertPrefixModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint='bert-base-uncased',
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=True,
        reduction='mean',
        soft_labels=None,
        alpha=0,
        return_logits=False,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :
                                                          -1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                shifted_prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))
        if soft_labels is not None:
            loss_distill = -torch.sum(
                F.log_softmax(shifted_prediction_scores, dim=1) * soft_labels,
                dim=-1)
            loss_distill = loss_distill[labels != -100].mean()
            lm_loss = (1 - alpha) * lm_loss + alpha * loss_distill

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((lm_loss, ) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


class MPlug(PreTrainedModel):
    config_class = MPlugConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(config.model_dir, ModelFile.VOCAB_FILE))
        self.module_setting(config)
        self.visual_encoder = self._initialize_clip(config)
        self.text_encoder = BertModel(
            self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionModel(
            self.config_fusion, add_pooling_layer=False)

    @classmethod
    def from_pretrained(cls, model_dir, task=None, load_checkpoint=True):
        from modelscope.utils.constant import Tasks

        task_mapping = {
            Tasks.visual_question_answering: MPlugForVisualQuestionAnswering,
            Tasks.image_captioning: MPlugForImageCaption,
            Tasks.image_text_retrieval: MPlugForImageTextRetrieval,
        }
        config = cls.config_class.from_yaml_file(
            os.path.join(model_dir, CONFIG_NAME))
        config.model_dir = model_dir
        if task is None:
            task = config.task
        model = task_mapping[task](config)
        if load_checkpoint:
            checkpoint_path = os.path.join(model_dir,
                                           ModelFile.TORCH_MODEL_BIN_FILE)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            if 'module' in checkpoint:
                checkpoint = checkpoint['module']
            checkpoint = {
                k.replace('model.', ''): v
                for k, v in checkpoint.items()
            }

            msg = model.load_state_dict(checkpoint, strict=False)
            print('load checkpoint from %s' % checkpoint_path)
            print(msg)
        return model

    @staticmethod
    def _initialize_clip(config, num_patches=240):

        def resize_pos_embed(posemb, posemb_new):
            # Rescale the grid of position embeddings when loading from state_dict. Adapted from
            # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
            ntok_new = posemb_new.shape[1]

            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
            ntok_new -= 1

            gs_old = int(math.sqrt(len(posemb_grid)))
            gs_new = int(math.sqrt(ntok_new))
            # _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
            posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                              -1).permute(0, 3, 1, 2)
            orig = posemb_grid.dtype
            posemb_grid = F.interpolate(
                posemb_grid.float(), size=(gs_new, gs_new), mode='bilinear')
            posemb_grid = posemb_grid.to(orig)
            posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(
                1, gs_new * gs_new, -1)
            posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
            return posemb

        from .clip import clip
        clip_model = clip.load_from_config(config)
        if 'ViT-B-16' in config.clip_name:
            num_patches = int(config.image_res * config.image_res / (16 * 16))
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        else:
            num_patches = int(config.image_res * config.image_res / (14 * 14))
            pos_embed = nn.Parameter(
                torch.zeros(num_patches + 1, 1024).float())
        pos_embed.weight = resize_pos_embed(
            clip_model.visual.positional_embedding.unsqueeze(0),
            pos_embed.unsqueeze(0))
        clip_model.visual.positional_embedding = pos_embed
        return clip_model

    def init_distill(self, config):
        self.distill = config.distill
        if self.distill:
            self.visual_encoder_m = self._initialize_clip(config)
            self.text_encoder_m = BertModel(
                self.config_encoder, add_pooling_layer=False)
            self.fusion_encoder_m = FusionModel(
                self.config_fusion, add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel(self.config_decoder)
            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.text_decoder, self.text_decoder_m],
            ]
            if self.config_encoder.hidden_size != config.vision_width:
                self.visn_fc_m = nn.Linear(config.vision_width,
                                           self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(
                    self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(
                    self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend(
                    [[self.visn_fc, self.visn_fc_m],
                     [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params()
            self.momentum = 0.995

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def module_setting(self, config):
        bert_config_path = os.path.join(config.model_dir, config.bert_config)
        self.config_encoder = BertConfig.from_json_file(bert_config_path)
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(bert_config_path)
        self.config_decoder = BertConfig.from_json_file(bert_config_path)
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != config.vision_width:
            self.visn_fc = nn.Linear(config.vision_width,
                                     self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(
                self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1. - self.momentum)

    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        topk_ids, topk_scores = self.beam_generator.translate_batch(
            encoder_inputs, out_size=out_size)
        return topk_ids, topk_scores

    @staticmethod
    def _tile(x, dim, n_tile):
        import numpy as np
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate(
                [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device))


class MPlugForVisualQuestionAnswering(MPlug):

    def __init__(self, config):
        super().__init__(config)
        self.text_decoder = BertLMHeadModel(self.config_decoder)
        self.beam_generator = TextGenerator(config, self.text_decoder)
        self.init_distill(config)

    def forward(self,
                image,
                question,
                answer=None,
                alpha=0,
                k=None,
                weights=None,
                train=True):
        image = image.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(
                self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100)
            text_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(
                encoder_embeds=text_embeds,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False)

            image_output, question_output = fusion_output

            question_output = torch.cat([image_output, question_output], 1)
            merge_text_attention = torch.cat(
                [image_atts, question.attention_mask], 1)

            if k is None:
                k = [1] * question_output.shape[0]
            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output[b]] * n
                question_atts += [merge_text_attention[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m.visual(
                        image, skip_last_layer=True)
                    if self.large:
                        image_embeds_m = self.dropout_m(
                            self.visn_layer_norm_m(
                                self.visn_fc_m(image_embeds_m)))
                    text_output_m = self.text_encoder_m(
                        question.input_ids,
                        attention_mask=question.attention_mask,
                        return_dict=True)
                    text_embeds_m = text_output_m.last_hidden_state
                    fusion_output_m = self.fusion_encoder_m(
                        encoder_embeds=text_embeds_m,
                        attention_mask=question.attention_mask,
                        encoder_hidden_states=image_embeds_m,
                        encoder_attention_mask=image_atts,
                        return_dict=False)

                    image_output_m, question_output_m = fusion_output_m
                    question_output_m = torch.cat(
                        [image_output_m, question_output_m], 1)

                    question_states_m = []
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m[b]] * n
                    question_states_m = torch.stack(question_states_m, 0)

                    logits_m = self.text_decoder_m(
                        answer.input_ids,
                        attention_mask=answer.attention_mask,
                        encoder_hidden_states=question_states_m,
                        encoder_attention_mask=question_atts,
                        return_logits=True,
                    )

                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    soft_labels=F.softmax(logits_m, dim=-1),
                    reduction='none',
                )
            else:
                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    reduction='none',
                )
            if weights is None:
                weights = 1
            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        else:
            text_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(
                encoder_embeds=text_embeds,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False)
            image_output, question_output = fusion_output
            question_output = torch.cat([image_output, question_output], 1)
            merge_text_attention = torch.cat(
                [image_atts, question.attention_mask], 1)
            topk_ids, topk_probs = self.generation(question_output,
                                                   merge_text_attention)
            return topk_ids, topk_probs


class MPlugForImageCaption(MPlug):

    def __init__(self, config):
        super().__init__(config)
        self.text_decoder = BertPrefixModel(self.config_decoder)
        self.beam_generator = TextGenerator(config, self.text_decoder)

    def beam_search(self,
                    image,
                    question,
                    answer=None,
                    train=True,
                    out_size=5):
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(
                self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text_output = self.text_encoder(
            question.input_ids,
            attention_mask=question.attention_mask,
            return_dict=True)
        text_embeds = text_output.last_hidden_state
        fusion_output = self.fusion_encoder(
            encoder_embeds=text_embeds,
            attention_mask=question.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False)
        image_output, question_output = fusion_output
        question_output = torch.cat([image_output, question_output], 1)
        merge_text_attention = torch.cat([image_atts, question.attention_mask],
                                         1)
        topk_ids, topk_probs = self.generation(
            question_output, merge_text_attention, out_size=out_size)
        return topk_ids, topk_probs

    def forward(self,
                image,
                question,
                answer=None,
                train=True,
                out_size=5,
                scst=False):
        if (scst):
            return self.beam_search(
                image, question, answer, train=True, out_size=out_size)
        image = image.to(dtype=next(self.parameters()).dtype)
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(
                self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100)
            answer_output = self.text_decoder(
                answer.input_ids,
                attention_mask=answer.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                labels=answer_targets,
                return_dict=True,
                reduction='none')
            loss = answer_output.loss

            return loss
        else:
            topk_ids, topk_probs = self.generation(image_embeds, image_atts)
            return topk_ids, topk_probs


class MPlugForImageTextRetrieval(MPlug):

    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.embed_dim
        self.temp = nn.Parameter(torch.ones([]) * config.temp)
        self.queue_size = config.queue_size
        self.momentum = config.momentum
        self.alpha = config.alpha

        self.queue_size = config.queue_size
        self.text_width = self.config_encoder.hidden_size
        self.embed_dim = config.embed_dim

        self.vision_proj = nn.Linear(self.text_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        self.itm_head = nn.Linear(self.text_width, 2)

        self.register_buffer('image_queue',
                             torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer('text_queue',
                             torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer('idx_queue', torch.full((1, self.queue_size),
                                                     -100))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.image_queue = F.normalize(self.image_queue, dim=0)
        self.text_queue = F.normalize(self.text_queue, dim=0)
        self.init_distill(config)

    def init_distill(self, config):
        self.distill = config.distill
        if self.distill:
            self.visual_encoder_m = self._initialize_clip(config)
            self.text_encoder_m = BertModel(
                self.config_encoder, add_pooling_layer=False)
            self.fusion_encoder_m = FusionModel(
                self.config_fusion, add_pooling_layer=False)
            self.vision_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.text_proj_m = nn.Linear(self.text_width, self.embed_dim)
            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.text_proj, self.text_proj_m],
                [self.vision_proj, self.vision_proj_m],
            ]
            if self.config_encoder.hidden_size != config.vision_width:
                self.visn_fc_m = nn.Linear(config.vision_width,
                                           self.config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(
                    self.config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(
                    self.config_encoder.hidden_dropout_prob)
                self.model_pairs.extend(
                    [[self.visn_fc, self.visn_fc_m],
                     [self.visn_layer_norm, self.visn_layer_norm_m]])
            self.copy_params()
            self.momentum = 0.995

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):

        def concat_all_gather(tensor):
            """
            Performs all_gather operation on the provided tensors.
            *** Warning ***: torch.distributed.all_gather has no gradient.
            """
            if not torch.distributed.is_initialized():
                return tensor
            tensors_gather = [
                torch.ones_like(tensor)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(
                tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
            return output

        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        idxs = concat_all_gather(idx)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, image, text, idx=None, train=True):
        if train:
            image_embeds = self.visual_encoder.visual(
                image, skip_last_layer=True)
            if self.large:
                image_embeds = self.dropout(
                    self.visn_layer_norm(self.visn_fc(image_embeds)))
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            image_feat = F.normalize(
                self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(
                self.text_proj(text_embeds[:, 0, :]), dim=-1)

            idx = idx.view(-1, 1)
            idx_all = torch.cat(
                [idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            with torch.no_grad():
                self._momentum_update()
                image_embeds_m = self.visual_encoder_m.visual(
                    image, skip_last_layer=True)
                if self.large:
                    image_embeds_m = self.dropout_m(
                        self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
                image_feat_m = F.normalize(
                    self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
                image_feat_all = torch.cat(
                    [image_feat_m.t(),
                     self.image_queue.clone().detach()],
                    dim=1)
                text_output_m = self.text_encoder_m(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True)
                text_feat_m = F.normalize(
                    self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]),
                    dim=-1)
                text_feat_all = torch.cat(
                    [text_feat_m.t(),
                     self.text_queue.clone().detach()], dim=1)

                if self.distill:
                    sim_i2t_m = image_feat_m @ text_feat_all / self.temp
                    sim_t2i_m = text_feat_m @ image_feat_all / self.temp

                    sim_i2t_targets = self.alpha * F.softmax(
                        sim_i2t_m, dim=1) + (1 - self.alpha) * sim_targets
                    sim_t2i_targets = self.alpha * F.softmax(
                        sim_t2i_m, dim=1) + (1 - self.alpha) * sim_targets

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp

            if self.distill:
                loss_i2t = -torch.sum(
                    F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
                    dim=1).mean()
                loss_t2i = -torch.sum(
                    F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,
                    dim=1).mean()
            else:
                loss_i2t = -torch.sum(
                    F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
                loss_t2i = -torch.sum(
                    F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

            loss_ita = (loss_i2t + loss_t2i) / 2

            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

            # forward the positve image-text pair
            _, output_pos = self.fusion_encoder(
                encoder_embeds=text_embeds,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
            )
            with torch.no_grad():
                bs = image.size(0)
                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

                mask = torch.eq(idx, idx.T)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

            # select a negative image for each text
            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            # select a negative text for each image
            text_embeds_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_embeds_neg.append(text_embeds[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
            text_atts_all = torch.cat([text.attention_mask, text_atts_neg],
                                      dim=0)

            image_embeds_all = torch.cat([image_embeds_neg, image_embeds],
                                         dim=0)
            image_atts_all = torch.cat([image_atts, image_atts], dim=0)

            _, output_neg = self.fusion_encoder(
                encoder_embeds=text_embeds_all,
                attention_mask=text_atts_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=False,
            )

            vl_embeddings = torch.cat(
                [output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
            vl_output = self.itm_head(vl_embeddings)

            ones_tmp = torch.ones(bs, dtype=torch.long)
            zeros_tmp = torch.zeros(2 * bs, dtype=torch.long)
            itm_labels = torch.cat([ones_tmp, zeros_tmp],
                                   dim=0).to(image.device)
            loss_itm = F.cross_entropy(vl_output, itm_labels)

            return loss_ita + loss_itm
        else:
            text_output = self.text_encoder(
                text.input_ids, attention_mask=text.attention_mask)
            text_feat = text_output.last_hidden_state
            image_feat = self.visual_encoder.visual(
                image, skip_last_layer=True)
            image_feat = self.visn_layer_norm(self.visn_fc(image_feat))
            image_att = torch.ones(
                image_feat.size()[:-1],
                dtype=torch.long,
                device=image_feat.device)
            _, output = self.fusion_encoder(
                encoder_embeds=text_feat,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_feat,
                encoder_attention_mask=image_att,
                return_dict=False,
            )
            scores = self.itm_head(output[:, 0, :])
            scores = F.softmax(scores, dim=-1)

            return scores


class HiTeA(PreTrainedModel):
    config_class = HiTeAConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(config.model_dir, ModelFile.VOCAB_FILE))
        self.module_setting(config)
        self.visual_encoder = MViTv2(
            img_size=config.image_res,
            config=MViTv2_Base_config,
            num_frames=config.num_frames)
        self.text_encoder = BertModel(
            self.config_encoder, add_pooling_layer=False)
        self.fusion_encoder = FusionModel(
            self.config_fusion, add_pooling_layer=False)

    @classmethod
    def from_pretrained(cls, model_dir, load_checkpoint=True):
        from modelscope.utils.constant import Tasks

        task_mapping = {
            Tasks.video_question_answering: HiTeAForVideoQuestionAnswering,
            Tasks.video_captioning: HiTeAForVideoCaption,
        }
        config = cls.config_class.from_yaml_file(
            os.path.join(model_dir, CONFIG_NAME))
        config.model_dir = model_dir
        model = task_mapping[config.task](config)
        if load_checkpoint:
            checkpoint_path = os.path.join(model_dir,
                                           ModelFile.TORCH_MODEL_BIN_FILE)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                checkpoint = checkpoint['model']
            if 'module' in checkpoint:
                checkpoint = checkpoint['module']
            checkpoint = {
                k.replace('model.', ''): v
                for k, v in checkpoint.items()
            }

            model.load_state_dict(checkpoint, strict=False)
        return model

    def init_distill(self, config):
        self.distill = config.distill
        if self.distill:
            self.visual_encoder_m = MViTv2(
                img_size=config.image_res,
                config=MViTv2_Base_config,
                num_frames=config.num_frames)
            self.text_encoder_m = BertModel(
                self.config_encoder, add_pooling_layer=False)
            self.fusion_encoder_m = FusionModel(
                self.config_fusion, add_pooling_layer=False)
            self.text_decoder_m = BertLMHeadModel(self.config_decoder)
            self.model_pairs = [
                [self.visual_encoder, self.visual_encoder_m],
                [self.text_encoder, self.text_encoder_m],
                [self.text_decoder, self.text_decoder_m],
            ]
            self.copy_params()
            self.momentum = 0.995

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def module_setting(self, config):
        bert_config_path = os.path.join(config.model_dir, config.bert_config)
        self.config_encoder = BertConfig.from_json_file(bert_config_path)
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(bert_config_path)
        self.config_decoder = BertConfig.from_json_file(bert_config_path)
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(),
                                      model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1. - self.momentum)

    def generation(self, question_states, question_atts, out_size=1):
        encoder_inputs = [question_states, question_atts]
        topk_ids, topk_scores = self.beam_generator.translate_batch(
            encoder_inputs, out_size=out_size)
        return topk_ids, topk_scores

    @staticmethod
    def _tile(x, dim, n_tile):
        import numpy as np
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate(
                [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device))


class HiTeAForVideoQuestionAnswering(HiTeA):

    def __init__(self, config):
        super().__init__(config)
        self.text_decoder = BertLMHeadModel(self.config_decoder)
        self.beam_generator = TextGenerator(config, self.text_decoder)
        self.init_distill(config)

    def forward(self,
                video,
                question,
                answer=None,
                alpha=0,
                k=None,
                weights=None,
                train=True):
        video = video.to(dtype=next(self.parameters()).dtype)
        video_embeds = self.visual_encoder(video)
        video_atts = torch.ones(
            video_embeds.size()[:-1], dtype=torch.long).to(video.device)

        if train:
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100)
            text_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(
                encoder_embeds=text_embeds,
                attention_mask=question.attention_mask,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                return_dict=False)

            video_output, question_output = fusion_output

            question_output = torch.cat([video_output, question_output], 1)
            merge_text_attention = torch.cat(
                [video_atts, question.attention_mask], 1)

            if k is None:
                k = [1] * question_output.shape[0]
            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output[b]] * n
                question_atts += [merge_text_attention[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            if self.distill:
                with torch.no_grad():
                    self._momentum_update()
                    video_embeds_m = self.visual_encoder_m(video)
                    text_output_m = self.text_encoder_m(
                        question.input_ids,
                        attention_mask=question.attention_mask,
                        return_dict=True)
                    text_embeds_m = text_output_m.last_hidden_state
                    fusion_output_m = self.fusion_encoder_m(
                        encoder_embeds=text_embeds_m,
                        attention_mask=question.attention_mask,
                        encoder_hidden_states=video_embeds_m,
                        encoder_attention_mask=video_atts,
                        return_dict=False)

                    image_output_m, question_output_m = fusion_output_m
                    question_output_m = torch.cat(
                        [image_output_m, question_output_m], 1)

                    question_states_m = []
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m[b]] * n
                    question_states_m = torch.stack(question_states_m, 0)

                    logits_m = self.text_decoder_m(
                        answer.input_ids,
                        attention_mask=answer.attention_mask,
                        encoder_hidden_states=question_states_m,
                        encoder_attention_mask=question_atts,
                        return_logits=True,
                    )

                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    soft_labels=F.softmax(logits_m, dim=-1),
                    reduction='none',
                )
            else:
                answer_output = self.text_decoder(
                    answer.input_ids,
                    attention_mask=answer.attention_mask,
                    encoder_hidden_states=question_states,
                    encoder_attention_mask=question_atts,
                    labels=answer_targets,
                    return_dict=True,
                    reduction='none',
                )
            if weights is None:
                weights = 1
            loss = weights * answer_output.loss
            loss = loss.sum() / video.size(0)

            return loss

        else:
            text_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                return_dict=True)
            text_embeds = text_output.last_hidden_state
            fusion_output = self.fusion_encoder(
                encoder_embeds=text_embeds,
                attention_mask=question.attention_mask,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                return_dict=False)
            video_output, question_output = fusion_output
            question_output = torch.cat([video_output, question_output], 1)
            merge_text_attention = torch.cat(
                [video_atts, question.attention_mask], 1)
            topk_ids, topk_probs = self.generation(question_output,
                                                   merge_text_attention)
            return topk_ids, topk_probs


class HiTeAForVideoCaption(HiTeA):

    def __init__(self, config):
        super().__init__(config)
        self.text_decoder = BertPrefixModel(self.config_decoder)
        self.beam_generator = TextGenerator(config, self.text_decoder)

    def beam_search(self,
                    video,
                    question,
                    answer=None,
                    train=True,
                    out_size=5):
        video_embeds = self.visual_encoder(video)
        video_atts = torch.ones(
            video_embeds.size()[:-1], dtype=torch.long).to(video.device)
        text_output = self.text_encoder(
            question.input_ids,
            attention_mask=question.attention_mask,
            return_dict=True)
        text_embeds = text_output.last_hidden_state
        fusion_output = self.fusion_encoder(
            encoder_embeds=text_embeds,
            attention_mask=question.attention_mask,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_atts,
            return_dict=False)
        video_output, question_output = fusion_output
        question_output = torch.cat([video_output, question_output], 1)
        merge_text_attention = torch.cat([video_atts, question.attention_mask],
                                         1)
        topk_ids, topk_probs = self.generation(
            question_output, merge_text_attention, out_size=out_size)
        return topk_ids, topk_probs

    def forward(self,
                video,
                question,
                answer=None,
                train=True,
                out_size=5,
                scst=False):
        if (scst):
            return self.beam_search(
                video, question, answer, train=True, out_size=out_size)
        video = video.to(dtype=next(self.parameters()).dtype)
        video_embeds = self.visual_encoder(video)
        video_atts = torch.ones(
            video_embeds.size()[:-1], dtype=torch.long).to(video.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100)
            answer_output = self.text_decoder(
                answer.input_ids,
                attention_mask=answer.attention_mask,
                encoder_hidden_states=video_embeds,
                encoder_attention_mask=video_atts,
                labels=answer_targets,
                return_dict=True,
                reduction='none')
            loss = answer_output.loss

            return loss
        else:
            topk_ids, topk_probs = self.generation(video_embeds, video_atts)
            return topk_ids, topk_probs
