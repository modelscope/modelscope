# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch PEER model. """

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN, get_activation
from transformers.file_utils import ModelOutput, add_start_docstrings
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)

from modelscope.models import Model, TorchModel
from modelscope.utils import logger as logging
from modelscope.utils.nlp.utils import parse_labels_in_order
from .configuration import PeerConfig
from .sas_utils import SequenceSideInfo

logger = logging.get_logger()

PEER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    'google/peer-small-generator',
    'google/peer-base-generator',
    'google/peer-large-generator',
    'google/peer-small-discriminator',
    'google/peer-base-discriminator',
    'google/peer-large-discriminator',
    # See all PEER models at https://huggingface.co/models?filter=peer
]


class PeerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.embedding_size,
            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               ['absolute'])
        if 'absolute_token_position_in_sentence' in self.position_embedding_type:
            self.side_info_size = 16
            self.position_embeddings__token_position_in_sentence = nn.Embedding(
                self.side_info_size, config.embedding_size)

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
            side_info_sets=dict(),
    ):
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
        if 'absolute' in self.position_embedding_type:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if 'absolute_token_position_in_sentence' in self.position_embedding_type:
            position_idx = torch.clamp(
                side_info_sets['ss_token_position_in_sentence'],
                min=0,
                max=self.side_info_size - 1)
            position_embeddings__token_position_in_sentence = self.position_embeddings__token_position_in_sentence(
                position_idx)
            embeddings += position_embeddings__token_position_in_sentence

        # Pass to attention layers to calcualte position-2-position attention scores
        if 'absolute_self_only' in self.position_embedding_type:
            if 'embeddings' not in side_info_sets:
                side_info_sets['embeddings'] = dict()
            side_info_sets['embeddings'][
                'ss_token_position_in_sequence'] = self.position_embeddings(
                    position_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PeerSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
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
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               ['absolute'])

        if 'relative_scalar_bias' in self.position_embedding_type:
            self.max_relative_position_embeddings = config.max_position_embeddings // 4
            self.distance_embedding = nn.Embedding(
                2 * self.max_relative_position_embeddings,
                self.num_attention_heads)

        elif 'relative_scalar_bias_with_side_info_token' in self.position_embedding_type:
            self.max_relative_position_embeddings = config.max_position_embeddings // 4
            self.side_info_size = 16  # leverage the information of token_position_in_sentence
            self.distance_embedding = nn.Embedding(
                (2 * self.max_relative_position_embeddings)
                * self.side_info_size, self.num_attention_heads)

        elif 'relative_scalar_bias_token_plus_sentence' in self.position_embedding_type:
            self.max_relative_position_embeddings = config.max_position_embeddings // 4
            self.max_sen_relative_position_embeddings = self.max_relative_position_embeddings // 4

            self.distance_embedding = nn.Embedding(
                2 * self.max_relative_position_embeddings,
                self.num_attention_heads)
            self.distance_embedding_sentence = nn.Embedding(
                2 * self.max_sen_relative_position_embeddings,
                self.num_attention_heads)

        elif 'relative_scalar_bias_with_side_info_sentence' in self.position_embedding_type:
            self.max_relative_position_embeddings = config.max_position_embeddings // 4
            self.max_sen_relative_position_embeddings = self.max_relative_position_embeddings // 4

            vocab = (2 * self.max_relative_position_embeddings) * (
                2 * self.max_sen_relative_position_embeddings)
            self.distance_embedding = nn.Embedding(vocab,
                                                   self.num_attention_heads)

        elif 'relative_key' in self.position_embedding_type or 'relative_key_query' in self.position_embedding_type:
            self.max_relative_position_embeddings = config.max_position_embeddings // 4
            self.distance_embedding = nn.Embedding(
                2 * self.max_relative_position_embeddings,
                self.attention_head_size)

        self.is_decoder = config.is_decoder

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
            side_info_sets=dict(),
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
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

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(
            -1, -2)) / math.sqrt(self.attention_head_size)
        attention_scores_terms = 1

        if 'absolute_self_only' in self.position_embedding_type:
            attention_scores += side_info_sets[
                'side_info_attention_scores']  # already normalized by sqrt(attention_head_size)
            attention_scores_terms += 1

        if 'relative_key' in self.position_embedding_type or 'relative_key_query' in self.position_embedding_type \
                or 'relative_scalar_bias' in self.position_embedding_type \
                or 'relative_scalar_bias_with_side_info_token' in self.position_embedding_type \
                or 'relative_scalar_bias_token_plus_sentence' in self.position_embedding_type \
                or 'relative_scalar_bias_with_side_info_sentence' in self.position_embedding_type:

            distance_idx = side_info_sets['distance_idx']

            positional_embedding = self.distance_embedding(distance_idx)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if 'relative_scalar_bias' in self.position_embedding_type:
                relative_scalar_bias = positional_embedding.permute(
                    [2, 0, 1]).unsqueeze(0)
                attention_scores = attention_scores / math.sqrt(
                    attention_scores_terms) + relative_scalar_bias

            elif ('relative_scalar_bias_with_side_info_token'
                  in self.position_embedding_type
                  or 'relative_scalar_bias_with_side_info_sentence'
                  in self.position_embedding_type):
                relative_scalar_bias = positional_embedding.permute(
                    [0, 3, 1, 2])
                attention_scores = attention_scores / math.sqrt(
                    attention_scores_terms) + relative_scalar_bias

            elif 'relative_scalar_bias_token_plus_sentence' in self.position_embedding_type:
                relative_scalar_bias = positional_embedding.permute(
                    [2, 0, 1]).unsqueeze(0)

                distance_idx_sentence = side_info_sets['distance_idx_sentence']
                positional_embedding_sentence = self.distance_embedding_sentence(
                    distance_idx_sentence)
                positional_embedding_sentence = positional_embedding_sentence.to(
                    dtype=query_layer.dtype)  # fp16 compatibility
                relative_scalar_bias_sentence = positional_embedding_sentence.permute(
                    [0, 3, 1, 2])

                attention_scores = attention_scores / math.sqrt(
                    attention_scores_terms
                ) + relative_scalar_bias + relative_scalar_bias_sentence

            elif 'relative_key' in self.position_embedding_type:
                relative_position_scores = torch.einsum(
                    'bhld,lrd->bhlr', query_layer,
                    positional_embedding) / math.sqrt(self.attention_head_size)
                attention_scores_terms += 1
                attention_scores = (attention_scores + relative_position_scores
                                    ) / math.sqrt(attention_scores_terms)
            elif 'relative_key_query' in self.position_embedding_type:
                relative_position_scores_query = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    'bhrd,lrd->bhlr', key_layer, positional_embedding)
                relative_position_scores = (
                    relative_position_scores_query
                    + relative_position_scores_key) / math.sqrt(
                        self.attention_head_size)
                attention_scores_terms += 2
                attention_scores = (attention_scores + relative_position_scores
                                    ) / math.sqrt(attention_scores_terms)

        else:
            attention_scores = attention_scores / math.sqrt(
                attention_scores_terms)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in PeerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer,
                   attention_probs) if output_attentions else (context_layer, )

        if self.is_decoder:
            outputs = outputs + (past_key_value, )
        return outputs


class PeerSelfOutput(nn.Module):

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


class PeerAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = PeerSelfAttention(config)
        self.output = PeerSelfOutput(config)
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
            side_info_sets=dict(),
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
            side_info_sets,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class PeerIntermediate(nn.Module):

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


class PeerOutput(nn.Module):

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


class PeerLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PeerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = PeerAttention(config)
        self.intermediate = PeerIntermediate(config)
        self.output = PeerOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            side_info_sets=dict(),
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
            side_info_sets=side_info_sets,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[
                1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, 'crossattention'
            ), f'If `encoder_hidden_states` are passed, {self} has to be instantiated \
                with cross-attention layers by setting `config.add_cross_attention=True`'

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[
                -2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[
                1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value, )

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class PeerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [PeerLayer(config) for _ in range(config.num_hidden_layers)])

        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               ['absolute'])
        if 'absolute_self_only' in self.position_embedding_type:
            # To be used/shared in all self-attention layers. Copy their dimensions here to be consistent.
            self.self_attention = self.layer[0].attention.self

            self.num_attention_heads = self.self_attention.num_attention_heads
            self.attention_head_size = self.self_attention.attention_head_size
            self.all_head_size = self.self_attention.all_head_size

            self.pos_query = nn.Linear(self.self_attention.query.in_features,
                                       self.self_attention.query.out_features)
            self.pos_key = nn.Linear(self.self_attention.key.in_features,
                                     self.self_attention.key.out_features)

    def get_position_attention_score(self, hidden_states):
        query_layer = self.self_attention.transpose_for_scores(
            self.pos_query(hidden_states))
        key_layer = self.self_attention.transpose_for_scores(
            self.pos_key(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        return attention_scores

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
            side_info_sets=dict(),
            return_dict=True,
    ):

        if 'absolute_self_only' in self.position_embedding_type:
            side_info_attention_scores = self.get_position_attention_score(
                hidden_states=side_info_sets['embeddings']
                ['ss_token_position_in_sequence'])
            side_info_sets[
                'side_info_attention_scores'] = side_info_attention_scores

        if 'relative_key' in self.position_embedding_type or 'relative_key_query' in self.position_embedding_type \
                or 'relative_scalar_bias' in self.position_embedding_type \
                or 'relative_scalar_bias_with_side_info_token' in self.position_embedding_type \
                or 'relative_scalar_bias_token_plus_sentence' in self.position_embedding_type \
                or 'relative_scalar_bias_with_side_info_sentence' in self.position_embedding_type:
            seq_length = hidden_states.shape[1]
            batch_size = hidden_states.shape[0]

            position_ids_l = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(1, -1)
            max_relative_position_embeddings = self.layer[
                0].attention.self.max_relative_position_embeddings
            distance_idx = torch.clamp(
                position_ids_l - position_ids_r
                + max_relative_position_embeddings - 2,
                min=0,
                max=2 * max_relative_position_embeddings - 4)
            distance_idx[
                0, :] = 2 * max_relative_position_embeddings - 3  # CLS-to-others
            distance_idx[:,
                         0] = 2 * max_relative_position_embeddings - 2  # others-to-CLS
            distance_idx[
                0, 0] = 2 * max_relative_position_embeddings - 1  # CLS-to-CLS
            distance_idx_max = 2 * max_relative_position_embeddings

            # token position-aware relative position
            if 'relative_scalar_bias_with_side_info_token' in self.position_embedding_type:
                idx1 = torch.clamp(
                    side_info_sets['ss_token_position_in_sentence'],
                    min=0,
                    max=self.layer[0].attention.self.side_info_size
                    - 1).unsqueeze(2).repeat(1, 1, seq_length)
                idx2 = distance_idx.unsqueeze(0).repeat(batch_size, 1, 1)
                distance_idx = idx1 * distance_idx_max + idx2
            # relative token position + relative sentence position
            elif 'relative_scalar_bias_with_side_info_sentence' in self.position_embedding_type:
                sen_position_ids_l = side_info_sets[
                    'ss_sentence_position_in_sequence'].view(
                        batch_size, -1, 1)
                sen_position_ids_r = side_info_sets[
                    'ss_sentence_position_in_sequence'].view(
                        batch_size, 1, -1)
                max_sen_relative_position_embeddings = self.layer[
                    0].attention.self.max_sen_relative_position_embeddings
                idx1 = torch.clamp(
                    sen_position_ids_l - sen_position_ids_r
                    + max_sen_relative_position_embeddings,
                    min=0,
                    max=2 * max_sen_relative_position_embeddings - 1)
                idx2 = distance_idx.unsqueeze(0).repeat(batch_size, 1, 1)
                distance_idx = idx1 * distance_idx_max + idx2
            elif 'relative_scalar_bias_token_plus_sentence' in self.position_embedding_type:
                sen_position_ids_l = side_info_sets[
                    'ss_sentence_position_in_sequence'].view(
                        batch_size, -1, 1)
                sen_position_ids_r = side_info_sets[
                    'ss_sentence_position_in_sequence'].view(
                        batch_size, 1, -1)
                max_sen_relative_position_embeddings = self.layer[
                    0].attention.self.max_sen_relative_position_embeddings
                idx1 = torch.clamp(
                    sen_position_ids_l - sen_position_ids_r
                    + max_sen_relative_position_embeddings,
                    min=0,
                    max=2 * max_sen_relative_position_embeddings - 1)
                side_info_sets['distance_idx_sentence'] = idx1

            side_info_sets['distance_idx'] = distance_idx

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
        ) if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[
                i] if past_key_values is not None else None
            if getattr(self.config, 'gradient_checkpointing', False):

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
                    side_info_sets,
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
                    side_info_sets,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1], )
            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    layer_outputs[1], )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[2], )

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


class PeerDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class PeerGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation('gelu')(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class PeerPreTrainedModel(TorchModel, PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PeerConfig
    base_model_prefix = 'teams1_shared_bottom'
    _keys_to_ignore_on_load_missing = [r'position_ids']
    _keys_to_ignore_on_load_unexpected = [
        r'peer\.embeddings_project\.weight', r'peer\.embeddings_project\.bias'
    ]

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

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels is not input.
                    label2id: An optional label2id mapping, which will cover the label2id in configuration (if exists).

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """

        model_dir = kwargs.pop('model_dir', None)
        cfg = kwargs.pop('cfg', None)
        model_args = parse_labels_in_order(model_dir, cfg, **kwargs)

        if model_dir is None:
            config = PeerConfig(**model_args)
            model = cls(config)
        else:
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_args)
        return model


@dataclass
class PeerForRTDOutput(ModelOutput):
    """
    Output type of :class:`~transformers.PeerForRTD`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the PEER objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`,
            returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`,
            returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PeerForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.PeerForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss of the PEER objective.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`,
            returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`,
            returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mlm_loss: Optional[torch.FloatTensor] = None
    rtd_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    rtd_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


PEER_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.PeerConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

PEER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.PeerTokenizer`. See
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
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

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
    'The bare Peer Model transformer outputting raw hidden-states without any specific head on top. Identical to '
    'the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the '
    'hidden size and embedding size are different.'
    ''
    'Both the generator and discriminator checkpoints may be loaded into this model.',
    PEER_START_DOCSTRING,
)
class PeerModel(PeerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.embeddings = PeerEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size,
                                                config.hidden_size)

        self.encoder = PeerEncoder(config)
        self.config = config
        self.init_weights()

        if self.config.seq_side_info_embeddings:
            self.input_sequence_side_info = dict()
            self.sequence_side_info = SequenceSideInfo()

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

    def update_seq_side_info(self, side_info_sets, input_ids):

        device = input_ids.device
        if 'input_sequence_side_info' not in side_info_sets or len(
                side_info_sets['input_sequence_side_info']) == 0:
            input_sequence_side_info = self.sequence_side_info.generate_seq_side_info(
                self.config.seq_side_info_embeddings, input_ids)

        else:
            # Save compute in PEER pre-training
            # (Save the extra side info into cpu in the first epoch; Directly retrieve it from cpu in later epochs)
            input_sequence_side_info = side_info_sets[
                'input_sequence_side_info']

        for ss in input_sequence_side_info.keys():
            input_sequence_side_info[ss] = input_sequence_side_info[ss].to(
                device=device).long()
        side_info_sets = {**side_info_sets, **input_sequence_side_info}
        return side_info_sets

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            side_info_sets=dict(),
            return_dict=None,
    ):
        if self.config.seq_side_info_embeddings:
            side_info_sets = self.update_seq_side_info(side_info_sets,
                                                       input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            side_info_sets=side_info_sets,
        )

        if hasattr(self, 'embeddings_project'):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            side_info_sets=side_info_sets,
            return_dict=return_dict,
        )

        return hidden_states


class PeerTopModel(PeerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.encoder = PeerEncoder(config)
        self.config = config
        self.init_weights()

        if self.config.seq_side_info_embeddings:
            self.input_sequence_side_info = dict()
            self.sequence_side_info = SequenceSideInfo()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def update_seq_side_info(self, side_info_sets, input_ids):

        device = input_ids.device
        if 'input_sequence_side_info' not in side_info_sets or len(
                side_info_sets['input_sequence_side_info']) == 0:
            input_sequence_side_info = self.sequence_side_info.generate_seq_side_info(
                self.config.seq_side_info_embeddings, input_ids)

        else:
            # Save compute in PEER pre-training
            # (Save the extra side info into cpu in the first epoch; Directly retrieve it from cpu in later epochs)
            input_sequence_side_info = side_info_sets[
                'input_sequence_side_info']

        for ss in input_sequence_side_info.keys():
            input_sequence_side_info[ss] = input_sequence_side_info[ss].to(
                device=device).long()
        side_info_sets = {**side_info_sets, **input_sequence_side_info}
        return side_info_sets

    def forward(
            self,
            hidden_states,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            side_info_sets=dict(),
            return_dict=None,
    ):

        if self.config.seq_side_info_embeddings:
            side_info_sets = self.update_seq_side_info(side_info_sets,
                                                       input_ids)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask,
                                       self.config.num_hidden_layers)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            side_info_sets=side_info_sets,
            return_dict=return_dict,
        )

        return hidden_states


class PeerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation('gelu')(
            x
        )  # although BERT uses tanh here, it seems Peer authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
