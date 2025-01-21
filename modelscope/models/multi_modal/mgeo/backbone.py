import math
import os
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.file_utils import (ModelOutput, add_code_sample_docstrings,
                                     add_start_docstrings,
                                     add_start_docstrings_to_model_forward,
                                     replace_return_docstrings)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions, MaskedLMOutput,
    MultipleChoiceModelOutput, NextSentencePredictorOutput,
    QuestionAnsweringModelOutput, SequenceClassifierOutput,
    TokenClassifierOutput)
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.nlp.utils import parse_labels_in_order

transformers.logging.set_verbosity_error()

logger = logging.get_logger()

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
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to
        # calculated m and v which are not required for using pretrained model
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


class GisEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, padding_idx=0)
        self.rel_type_embeddings = nn.Embedding(
            config.rel_type_vocab_size, config.hidden_size, padding_idx=0)
        self.absolute_x_embeddings = nn.Embedding(
            config.absolute_x_vocab_size, config.hidden_size, padding_idx=0)
        self.absolute_y_embeddings = nn.Embedding(
            config.absolute_y_vocab_size, config.hidden_size, padding_idx=0)
        self.relative_x_embeddings = nn.Embedding(
            config.relative_x_vocab_size, config.hidden_size, padding_idx=0)
        self.relative_y_embeddings = nn.Embedding(
            config.relative_y_vocab_size, config.hidden_size, padding_idx=0)
        if hasattr(config, 'prov_vocab_size'):
            self.prov_embeddings = nn.Embedding(
                config.prov_vocab_size, config.hidden_size, padding_idx=0)
            self.city_embeddings = nn.Embedding(
                config.city_vocab_size, config.hidden_size, padding_idx=0)
            self.dist_embeddings = nn.Embedding(
                config.dist_vocab_size, config.hidden_size, padding_idx=0)

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
                past_key_values_length=0,
                rel_type_ids=None,
                absolute_position_ids=None,
                relative_position_ids=None,
                prov_ids=None,
                city_ids=None,
                dist_ids=None):
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
        embeddings += self.rel_type_embeddings(rel_type_ids)
        embeddings += self.absolute_x_embeddings(absolute_position_ids[:, :,
                                                                       0])
        embeddings += self.absolute_y_embeddings(absolute_position_ids[:, :,
                                                                       1])
        embeddings += self.absolute_x_embeddings(absolute_position_ids[:, :,
                                                                       2])
        embeddings += self.absolute_y_embeddings(absolute_position_ids[:, :,
                                                                       3])
        embeddings += self.relative_x_embeddings(relative_position_ids[:, :,
                                                                       0])
        embeddings += self.relative_y_embeddings(relative_position_ids[:, :,
                                                                       1])
        embeddings += self.relative_x_embeddings(relative_position_ids[:, :,
                                                                       2])
        embeddings += self.relative_y_embeddings(relative_position_ids[:, :,
                                                                       3])
        if prov_ids is not None:
            embeddings += self.prov_embeddings(prov_ids)
            embeddings += self.city_embeddings(city_ids)
            embeddings += self.dist_embeddings(dist_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type
    embeddings."""

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
                past_key_values_length=0,
                rel_type_ids=None,
                absolute_position_ids=None,
                relative_position_ids=None):
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
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)

        self.has_cross_attention = (layer_num >= config.fusion_layer)
        if self.has_cross_attention:
            self.layer_num = layer_num
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
        # decoder uni-directional self-attention cached key/values tuple is at
        # positions 1,2
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

        outputs = outputs + (present_key_value, )

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        mode='multi_modal',
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
        ) if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        if mode == 'text':
            start_layer = 0
            output_layer = self.config.fusion_layer

        elif mode == 'query':
            start_layer = 0
            output_layer = self.config.num_hidden_layers

        elif mode == 'fusion':
            start_layer = self.config.fusion_layer
            output_layer = self.config.num_hidden_layers

        elif mode == 'multi_modal':
            start_layer = 0
            output_layer = self.config.num_hidden_layers

        for i in range(start_layer, output_layer):
            layer_module = self.layer[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None

            if getattr(self.config, 'gradient_checkpointing',
                       False) and self.training:

                if use_cache:
                    logger.warn(
                        '`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting '
                        '`use_cache=False`...')
                    use_cache = False

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
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
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


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`. Args:
        loss (`optional`, returned when ``labels`` is provided,
        ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the
            next sequence prediction (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size,
        sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each
            vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape
        :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification)
            head (scores of True/False continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned
        when ``output_hidden_states=True`` is passed or when
        ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the
            embeddings + one for the output of each layer) of shape
            :obj:`(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the initial embedding
            outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when
        ``output_attentions=True`` is passed or when
        ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertModel(BertPreTrainedModel):
    """
    Noted that the bert model here is slightly updated from original bert, so we
    maintian the code here independently. The Bert Model transformer outputting
    raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass
    documentation for the generic methods the library implements for all its
    model (such as downloading or saving, resizing the input embeddings, pruning
    heads etc.)

    This model is also a PyTorch
    [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch
    documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the
        parameters of the model.
            Initializing with a config file does not load the weights associated
            with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.

    The model can behave as an encoder (with only self-attention) as well as a
    decoder, in which case a layer of cross-attention is added between the
    self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam
    Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    `is_decoder` argument of the configuration set to `True`. To be used in a
    Seq2Seq model, the model needs to initialized with both `is_decoder`
    argument and `add_cross_attention` set to `True`; an `encoder_hidden_states`
    is then expected as an input to the forward pass.


    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if config.gis_embedding == 0:
            self.embeddings = BertEmbeddings(config)
        else:
            self.embeddings = GisEmbeddings(config)

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

    def get_extended_attention_mask(self, attention_mask: Tensor,
                                    input_shape: Tuple[int], device: device,
                                    is_decoder: bool) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked
        tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens
                to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same
            dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size,
        # from_seq_length, to_seq_length] ourselves in which case we just need
        # to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to
            #   the padding mask
            # - if the model is an encoder, make the mask broadcastable to
            #   [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(
                    batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones
                # mask to the causal mask causal and attention masks must have
                # same type with pytorch version < 1.3
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

        # Since attention_mask is 1.0 for positions we want to attend and 0.0
        # for masked positions, this operation will create a tensor which is 0.0
        # for positions we want to attend and -10000.0 for masked positions.
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
        mode='multi_modal',
        rel_type_ids=None,
        absolute_position_ids=None,
        relative_position_ids=None,
        prov_ids=None,
        city_ids=None,
        dist_ids=None,
    ):
        r"""
        Args:
        input_ids (`torch.LongTensor` of shape `((batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`BertTokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `((batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask
            values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `((batch_size, sequence_length)`, *optional*):
            Segment token indices to indicate first and second portions of the
            inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `((batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position
            embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers,
        num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask
            values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `((batch_size, sequence_length, hidden_size)`,
        *optional*):
            Optionally, instead of passing `input_ids` you can choose to
            directly pass an embedded representation. This is useful if you want
            more control over how to convert `input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention
            layers. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See
            `hidden_states` under returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a
            plain tuple.
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size,
        sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the
            encoder. Used in the cross-attention if the model is configured as a
            decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size,
        sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of
            the encoder input. This mask is used in the cross-attention if the
            model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length
        `config.n_layers` with each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention
            blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only
            the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead
            of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding (see `past_key_values`).
        Others (**kwargs)
            some additional parameters might passed in from upstream pipeline,
            which not influence the results.
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

        # We can provide a self-attention mask of dimensions [batch_size,
        # from_seq_length, to_seq_length] ourselves in which case we just need
        # to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device, is_decoder)

        # If a 2D or 3D attention mask is provided for the cross-attention we
        # need to make broadcastable to [batch_size, num_heads, seq_length,
        # seq_length]
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
                rel_type_ids=rel_type_ids,
                absolute_position_ids=absolute_position_ids,
                relative_position_ids=relative_position_ids,
                prov_ids=prov_ids,
                city_ids=city_ids,
                dist_ids=dist_ids,
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
            mode=mode,
        )
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


class BertForPreTraining(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        Returns:
        Example:
            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')
            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss, )
                    + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
        mode='multi_modal',
        soft_labels=None,
        alpha=0,
        return_logits=False,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape
        :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the
            encoder. Used in the cross-attention if the model is configured as a
            decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape
        :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of
            the encoder input. This mask is used in the cross-attention if the
            model is configured as a decoder. Mask values selected in ``[0,
            1]``: - 1 for tokens that are **not masked**, - 0 for tokens that
            are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next
            word prediction). Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with
            indices set to ``-100`` are ignored (masked), the loss is only
            computed for the tokens with labels n ``[0, ...,
            config.vocab_size]``
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length
        :obj:`config.n_layers` with each tuple having 4 tensors of shape
        :obj:`(batch_size, num_heads, sequence_length - 1,
        embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention
            blocks. Can be used to speed up decoding. If :obj:`past_key_values`
            are used, the user can optionally input only the last
            :obj:`decoder_input_ids` (those that don't have their past key value
            states given to this model) of shape :obj:`(batch_size, 1)` instead
            of all :obj:`decoder_input_ids` of shape :obj:`(batch_size,
            sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are
            returned and can be used to speed up decoding (see
            :obj:`past_key_values`).

        Returns:

        Example:
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
            mode=mode,
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
                F.log_softmax(shifted_prediction_scores, dim=-1) * soft_labels,
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


class GisBertLMPredictionHead(nn.Module):

    def __init__(self, config, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertForGisMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']
    _keys_to_ignore_on_load_missing = [
        r'position_ids', r'predictions.decoder.bias'
    ]

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls_geom_id = GisBertLMPredictionHead(config, config.vocab_size)
        self.cls_geom_type = GisBertLMPredictionHead(config,
                                                     config.type_vocab_size)
        self.cls_rel_type = GisBertLMPredictionHead(config,
                                                    config.rel_type_vocab_size)
        self.cls_absolute_position_x1 = GisBertLMPredictionHead(
            config, config.absolute_x_vocab_size)
        self.cls_absolute_position_x2 = GisBertLMPredictionHead(
            config, config.absolute_x_vocab_size)
        self.cls_absolute_position_y1 = GisBertLMPredictionHead(
            config, config.absolute_y_vocab_size)
        self.cls_absolute_position_y2 = GisBertLMPredictionHead(
            config, config.absolute_y_vocab_size)
        self.cls_relative_position_x1 = GisBertLMPredictionHead(
            config, config.relative_x_vocab_size)
        self.cls_relative_position_x2 = GisBertLMPredictionHead(
            config, config.relative_x_vocab_size)
        self.cls_relative_position_y1 = GisBertLMPredictionHead(
            config, config.relative_y_vocab_size)
        self.cls_relative_position_y2 = GisBertLMPredictionHead(
            config, config.relative_y_vocab_size)
        self.config = config

        self.init_weights()

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multi_modal',
        soft_labels=None,
        alpha=0,
        return_logits=False,
        rel_type_ids=None,
        absolute_position_ids=None,
        relative_position_ids=None,
        token_type_ids_label=None,
        rel_type_ids_label=None,
        absolute_position_ids_label=None,
        relative_position_ids_label=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices
            should be in ``[-100, 0, ..., config.vocab_size]`` (see
            ``input_ids`` docstring) Tokens with indices set to ``-100`` are
            ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_embeds=encoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
            rel_type_ids=rel_type_ids,
            absolute_position_ids=absolute_position_ids,
            relative_position_ids=relative_position_ids,
        )

        sequence_output = outputs[0]

        prediction_scores = self.cls_geom_id(sequence_output)
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            labels.view(-1))

        positions_cls = [
            self.cls_geom_type, self.cls_rel_type,
            self.cls_absolute_position_x1, self.cls_absolute_position_x2,
            self.cls_absolute_position_y1, self.cls_absolute_position_y2,
            self.cls_relative_position_x1, self.cls_relative_position_x2,
            self.cls_relative_position_y1, self.cls_relative_position_y2
        ]
        positions_label = [
            token_type_ids_label, rel_type_ids_label,
            absolute_position_ids_label[:, :,
                                        0], absolute_position_ids_label[:, :,
                                                                        2],
            absolute_position_ids_label[:, :,
                                        1], absolute_position_ids_label[:, :,
                                                                        3],
            relative_position_ids_label[:, :,
                                        0], relative_position_ids_label[:, :,
                                                                        2],
            relative_position_ids_label[:, :,
                                        1], relative_position_ids_label[:, :,
                                                                        3]
        ]
        positions_size = [
            self.config.type_vocab_size, self.config.rel_type_vocab_size,
            self.config.absolute_x_vocab_size,
            self.config.absolute_x_vocab_size,
            self.config.absolute_y_vocab_size,
            self.config.absolute_y_vocab_size,
            self.config.relative_x_vocab_size,
            self.config.relative_x_vocab_size,
            self.config.relative_y_vocab_size,
            self.config.relative_y_vocab_size
        ]
        for mycls, mylabels, mysize in zip(positions_cls, positions_label,
                                           positions_size):
            if mylabels is not None:
                myprediction_scores = mycls(sequence_output)
                masked_lm_loss += loss_fct(
                    myprediction_scores.view(-1, mysize), mylabels.view(-1))

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((masked_lm_loss, )
                    + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, 'The PAD token should be defined for generation'
        padding_mask = attention_mask.new_zeros((attention_mask.shape[0], 1))
        attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class BertForMaskedLM(BertPreTrainedModel):

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        is_decoder=False,
        mode='multi_modal',
        soft_labels=None,
        alpha=0,
        return_logits=False,
        rel_type_ids=None,
        absolute_position_ids=None,
        relative_position_ids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_embeds=encoder_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_decoder=is_decoder,
            mode=mode,
            rel_type_ids=rel_type_ids,
            absolute_position_ids=absolute_position_ids,
            relative_position_ids=relative_position_ids,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if return_logits:
            return prediction_scores

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1))

        if soft_labels is not None:
            loss_distill = -torch.sum(
                F.log_softmax(prediction_scores, dim=-1) * soft_labels, dim=-1)
            loss_distill = loss_distill[labels != -100].mean()
            masked_lm_loss = (1
                              - alpha) * masked_lm_loss + alpha * loss_distill

        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((masked_lm_loss, )
                    + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      attention_mask=None,
                                      **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, 'The PAD token should be defined for generation'

        padding_mask = attention_mask.new_zeros((attention_mask.shape[0], 1))
        attention_mask = torch.cat([attention_mask, padding_mask], dim=-1)
        dummy_token = torch.full((effective_batch_size, 1),
                                 self.config.pad_token_id,
                                 dtype=torch.long,
                                 device=input_ids.device)
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class BertForNextSentencePrediction(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

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
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequenåce prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring). Indices should be in ``[0, 1]``:
            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        Returns:

        Example:
            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')
            >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        pooled_output = outputs[1]

        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(
                seq_relationship_scores.view(-1, 2), labels.view(-1))

        if not return_dict:
            output = (seq_relationship_scores, ) + outputs[2:]
            return ((next_sentence_loss, )
                    + output) if next_sentence_loss is not None else output

        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForMultipleChoice(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where
            :obj:`num_choices` is the size of the second dimension of the input
            tensors. (See :obj:`input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(
            -1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(
            -1,
            attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(
            -1,
            token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(
            -1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2),
                               inputs_embeds.size(-1))
            if inputs_embeds is not None else None)

        outputs = self.bert(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForTokenClassification(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,
        sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should
            be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertForQuestionAnswering(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r'pooler']

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for position (index) of the start of the labelled span for
            computing the token classification loss. Positions are clamped to
            the length of the sequence (:obj:`sequence_length`). Position
            outside of the sequence are not taken into account for computing the
            loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for position (index) of the end of the labelled span for
            computing the token classification loss. Positions are clamped to
            the length of the sequence (:obj:`sequence_length`). Position
            outside of the sequence are not taken into account for computing the
            loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss, )
                    + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MGeoPreTrainedModel(TorchModel, PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = BertConfig
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

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the
                    label information. num_labels: An optional arg to tell the
                    model how many classes to initialize.
                                    Method will call utils.parse_label_mapping
                                    if num_labels not supplied. If num_labels is
                                    not found, the model will use the default
                                    setting (2 classes).

        Returns:
            The loaded model, which is initialized by
            transformers.PreTrainedModel.from_pretrained
        """

        model_dir = kwargs.pop('model_dir', None)
        cfg = kwargs.pop('cfg', None)
        model_args = parse_labels_in_order(model_dir, cfg, **kwargs)
        if model_dir is None:
            config = BertConfig(**model_args)
            model = cls(config)
        else:
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_args)
        model.model_dir = model_dir
        return model


@MODELS.register_module(Tasks.backbone, module_name=Models.mgeo)
class MGeo(MGeoPreTrainedModel):

    def __init__(self,
                 config: BertConfig,
                 finetune_mode: str = 'single-modal',
                 gis_num: int = 1,
                 add_pooling_layer=False,
                 **kwargs):
        super().__init__(config)

        self.finetune_mode = finetune_mode

        self.config = config
        self.text_encoder = BertModel(
            config, add_pooling_layer=add_pooling_layer)

        if self.finetune_mode == 'multi-modal':
            gis_config = BertConfig.from_dict(config.gis_encoder)
            self.gis_encoder = BertModel(gis_config, add_pooling_layer=False)
            for param in self.gis_encoder.parameters():
                param.requires_grad = False
            self.gis2text = nn.Linear(gis_config.hidden_size,
                                      self.config.hidden_size)
            self.gis_type = nn.Embedding(gis_num, gis_config.hidden_size)

        self.init_weights()

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
                is_decoder=False,
                mode='single-modal',
                gis_list=None,
                gis_tp=None,
                use_token_type=False):
        if self.finetune_mode == 'multi-modal' and gis_list is not None and len(
                gis_list) > 0:
            gis_embs = []
            gis_atts = []
            for gis in gis_list:
                gis_embs.append(
                    self.gis_encoder(return_dict=True, mode='text',
                                     **gis).last_hidden_state)
                gis_atts.append(gis['attention_mask'])
        if use_token_type:
            embedding_output = self.text_encoder.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            embedding_output = self.text_encoder.embeddings(
                input_ids=input_ids, )

        if self.finetune_mode == 'multi-modal' and gis_list is not None and len(
                gis_list) > 0:
            embs = [embedding_output]
            atts = [attention_mask]
            tp_emb = [self.gis_type(gtp) for gtp in gis_tp]
            for ge, ga, gt in zip(gis_embs, gis_atts, tp_emb):
                embs.append(self.gis2text(ge + gt))
                atts.append(ga)
            merge_emb = torch.cat(embs, dim=1)
            merge_attention = torch.cat(atts, dim=-1)
        else:
            merge_emb = embedding_output
            merge_attention = attention_mask
        encoder_outputs = self.text_encoder(
            attention_mask=merge_attention,
            encoder_embeds=merge_emb,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mode='text')

        if not return_dict:
            return encoder_outputs

        return AttentionBackboneModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=encoder_outputs.pooler_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
        return output

    def extract_sequence_outputs(self, outputs):
        return outputs['last_hidden_state']

    def extract_pooled_outputs(self, outputs):
        return outputs['pooler_output']
