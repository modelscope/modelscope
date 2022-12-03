# Copyright 2021-2022 The Alibaba DAMO Team Authors. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function
import copy
import math
import os
import shutil
import tarfile
import tempfile

import numpy as np
import torch
from torch import nn

from modelscope.models.nlp.space_T_cn.configuration import SpaceTCnConfig
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()

CONFIG_NAME = ModelFile.CONFIGURATION
WEIGHTS_NAME = ModelFile.TORCH_MODEL_BIN_FILE


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.match_type_embeddings = nn.Embedding(11, config.hidden_size)
        self.type_embeddings = nn.Embedding(6, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,
                input_ids,
                header_ids,
                token_type_ids=None,
                match_type_ids=None,
                l_hs=None,
                header_len=None,
                type_idx=None,
                col_dict_list=None,
                ids=None,
                header_flatten_tokens=None,
                header_flatten_index=None,
                header_flatten_output=None,
                token_column_id=None,
                token_column_mask=None,
                column_start_index=None,
                headers_length=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        header_embeddings = self.word_embeddings(header_ids)

        if col_dict_list is not None and l_hs is not None:
            col_dict_list = np.array(col_dict_list)[ids.cpu().numpy()].tolist()
            header_len = np.array(
                header_len, dtype=object)[ids.cpu().numpy()].tolist()
            for bi, col_dict in enumerate(col_dict_list):
                for ki, vi in col_dict.items():
                    length = header_len[bi][vi]
                    if length == 0:
                        continue
                    words_embeddings[bi, ki, :] = torch.mean(
                        header_embeddings[bi, vi, :length, :], dim=0)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if match_type_ids is not None:
            match_type_embeddings = self.match_type_embeddings(match_type_ids)
            embeddings += match_type_embeddings

        if type_idx is not None:
            type_embeddings = self.type_embeddings(type_idx)
            embeddings += type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
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

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, schema_link_matrix=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfAttentionWithRelationsRAT(nn.Module):
    '''
    Adapted from https://github.com/microsoft/rat-sql/blob/master/ratsql/models/transformer.py
    '''

    def __init__(self, config):
        super(BertSelfAttentionWithRelationsRAT, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
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

        self.relation_k_emb = nn.Embedding(
            7, config.hidden_size // config.num_attention_heads)
        self.relation_v_emb = nn.Embedding(
            7, config.hidden_size // config.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, relation):
        '''
        relation is [batch, seq len, seq len]
        '''
        mixed_query_layer = self.query(
            hidden_states)  # [batch, seq len, hidden dim]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        relation_k = self.relation_k_emb(
            relation)  # [batch, seq len, seq len, head dim]
        relation_v = self.relation_v_emb(
            relation)  # [batch, seq len, seq len, head dim]

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [batch, num attn heads, seq len, head dim]
        key_layer = self.transpose_for_scores(
            mixed_key_layer)  # [batch, num attn heads, seq len, head dim]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [batch, num attn heads, seq len, head dim]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(
            -1, -2))  # [batch, num attn heads, seq len, seq len]

        # relation_k_t is [batch, seq len, head dim, seq len]
        relation_k_t = relation_k.transpose(-2, -1)
        # query_layer_t is [batch, seq len, num attn heads, head dim]
        query_layer_t = query_layer.permute(0, 2, 1, 3)
        # relation_attention_scores is [batch, seq len, num attn heads, seq len]
        relation_attention_scores = torch.matmul(query_layer_t, relation_k_t)
        # relation_attention_scores_t is [batch, num attn heads, seq len, seq len]
        relation_attention_scores_t = relation_attention_scores.permute(
            0, 2, 1, 3)

        merged_attention_scores = (attention_scores
                                   + relation_attention_scores_t) / math.sqrt(
                                       self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        merged_attention_scores = merged_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(merged_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs is [batch, num attn heads, seq len, seq len]
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        # attention_probs_t is [batch, seq len, num attn heads, seq len]
        attention_probs_t = attention_probs.permute(0, 2, 1, 3)

        #   [batch, seq len, num attn heads, seq len]
        # * [batch, seq len, seq len, head dim]
        # = [batch, seq len, num attn heads, head dim]
        context_relation = torch.matmul(attention_probs_t, relation_v)

        # context_relation_t is [batch, num attn heads, seq len, head dim]
        context_relation_t = context_relation.permute(0, 2, 1, 3)

        merged_context_layer = context_layer + context_relation_t
        merged_context_layer = merged_context_layer.permute(0, 2, 1,
                                                            3).contiguous()
        new_context_layer_shape = merged_context_layer.size()[:-2] + (
            self.all_head_size, )
        merged_context_layer = merged_context_layer.view(
            *new_context_layer_shape)
        return merged_context_layer


class BertSelfAttentionWithRelationsTableformer(nn.Module):

    def __init__(self, config):
        super(BertSelfAttentionWithRelationsTableformer, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
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
        self.schema_link_embeddings = nn.Embedding(7, self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, relation):
        '''
        relation is [batch, seq len, seq len]
        '''
        mixed_query_layer = self.query(
            hidden_states)  # [batch, seq len, hidden dim]
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        schema_link_embeddings = self.schema_link_embeddings(
            relation)  # [batch, seq len, seq len, 1]
        schema_link_embeddings = schema_link_embeddings.permute(0, 3, 1, 2)

        query_layer = self.transpose_for_scores(
            mixed_query_layer)  # [batch, num attn heads, seq len, head dim]
        key_layer = self.transpose_for_scores(
            mixed_key_layer)  # [batch, num attn heads, seq len, head dim]
        value_layer = self.transpose_for_scores(
            mixed_value_layer)  # [batch, num attn heads, seq len, head dim]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(
            -1, -2))  # [batch, num attn heads, seq len, seq len]
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        merged_attention_scores = attention_scores + schema_link_embeddings

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        merged_attention_scores = merged_attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(merged_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs is [batch, num attn heads, seq len, seq len]
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config, schema_link_module='none'):
        super(BertAttention, self).__init__()
        if schema_link_module == 'none':
            self.self = BertSelfAttention(config)
        if schema_link_module == 'rat':
            self.self = BertSelfAttentionWithRelationsRAT(config)
        if schema_link_module == 'add':
            self.self = BertSelfAttentionWithRelationsTableformer(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, schema_link_matrix=None):
        self_output = self.self(input_tensor, attention_mask,
                                schema_link_matrix)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config, schema_link_module='none'):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(
            config, schema_link_module=schema_link_module)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, schema_link_matrix=None):
        attention_output = self.attention(hidden_states, attention_mask,
                                          schema_link_matrix)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SqlBertEncoder(nn.Module):

    def __init__(self, layers, config):
        super(SqlBertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertEncoder(nn.Module):

    def __init__(self, config, schema_link_module='none'):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config, schema_link_module=schema_link_module)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                all_schema_link_matrix=None,
                all_schema_link_mask=None,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask,
                                         all_schema_link_matrix)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
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
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, SpaceTCnConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `SpaceTCnConfig`. '
                'To create a model from a Google pretrained model use '
                '`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name,
                        state_dict=None,
                        cache_dir=None,
                        *inputs,
                        **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object)
                to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        resolved_archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info('extracting archive file {} to temp dir {}'.format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = SpaceTCnConfig.from_json_file(config_file)
        logger.info('Model config {}'.format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info(
                'Weights of {} not initialized from pretrained model: {}'.
                format(model.__class__.__name__, missing_keys))
            print()
            print('*' * 10, 'WARNING missing weights', '*' * 10)
            print('Weights of {} not initialized from pretrained model: {}'.
                  format(model.__class__.__name__, missing_keys))
            print()
        if len(unexpected_keys) > 0:
            logger.info(
                'Weights from pretrained model not used in {}: {}'.format(
                    model.__class__.__name__, unexpected_keys))
            print()
            print('*' * 10, 'WARNING unexpected weights', '*' * 10)
            print('Weights from pretrained model not used in {}: {}'.format(
                model.__class__.__name__, unexpected_keys))
            print()
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class SpaceTCnModel(PreTrainedBertModel):
    """SpaceTCnModel model ("Bidirectional Embedding Representations from a Transformer pretrained on STAR-T-CN").

    Params:
        config: a SpaceTCnConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output
            as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.SpaceTCnConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.SpaceTCnModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, schema_link_module='none'):
        super(SpaceTCnModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(
            config, schema_link_module=schema_link_module)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                header_ids,
                token_order_ids=None,
                token_type_ids=None,
                attention_mask=None,
                match_type_ids=None,
                l_hs=None,
                header_len=None,
                type_ids=None,
                col_dict_list=None,
                ids=None,
                header_flatten_tokens=None,
                header_flatten_index=None,
                header_flatten_output=None,
                token_column_id=None,
                token_column_mask=None,
                column_start_index=None,
                headers_length=None,
                all_schema_link_matrix=None,
                all_schema_link_mask=None,
                output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # Bowen: comment out the following line for Pytorch >= 1.5
        # https://github.com/huggingface/transformers/issues/3936#issuecomment-793764416
        # extended_attention_mask = extended_attention_mask.to(self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, header_ids, token_type_ids, match_type_ids, l_hs,
            header_len, type_ids, col_dict_list, ids, header_flatten_tokens,
            header_flatten_index, header_flatten_output, token_column_id,
            token_column_mask, column_start_index, headers_length)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            all_schema_link_matrix=all_schema_link_matrix,
            all_schema_link_mask=all_schema_link_mask,
            output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class Seq2SQL(nn.Module):

    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, n_action_ops,
                 max_select_num, max_where_num, device):
        super(Seq2SQL, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        self.device = device

        self.n_agg_ops = n_agg_ops
        self.n_cond_ops = n_cond_ops
        self.n_action_ops = n_action_ops
        self.max_select_num = max_select_num
        self.max_where_num = max_where_num

        self.w_sss_model = nn.Linear(iS, max_where_num)
        self.w_sse_model = nn.Linear(iS, max_where_num)
        self.s_ht_model = nn.Linear(iS, max_select_num)
        self.wc_ht_model = nn.Linear(iS, max_where_num)

        self.select_agg_model = nn.Linear(iS * max_select_num,
                                          n_agg_ops * max_select_num)
        self.w_op_model = nn.Linear(iS * max_where_num,
                                    n_cond_ops * max_where_num)

        self.conn_model = nn.Linear(iS, 3)
        self.action_model = nn.Linear(iS, n_action_ops + 1)
        self.slen_model = nn.Linear(iS, max_select_num + 1)
        self.wlen_model = nn.Linear(iS, max_where_num + 1)

    def set_device(self, device):
        self.device = device

    def forward(self, wemb_layer, l_n, l_hs, start_index, column_index, tokens,
                ids):
        # chunk input lists for multi-gpu
        max_l_n = max(l_n)
        max_l_hs = max(l_hs)
        l_n = np.array(l_n)[ids.cpu().numpy()].tolist()
        l_hs = np.array(l_hs)[ids.cpu().numpy()].tolist()
        start_index = np.array(start_index)[ids.cpu().numpy()].tolist()
        column_index = np.array(column_index)[ids.cpu().numpy()].tolist()
        # tokens = np.array(tokens)[ids.cpu().numpy()].tolist()

        conn_index = []
        slen_index = []
        wlen_index = []
        action_index = []
        where_op_index = []
        select_agg_index = []
        header_pos_index = []
        query_index = []
        for ib, elem in enumerate(start_index):
            # [SEP] conn [SEP] wlen [SEP] (wop [SEP])*wn slen [SEP] (agg [SEP])*sn
            action_index.append(elem + 1)
            conn_index.append(elem + 2)
            wlen_index.append(elem + 3)
            woi = [elem + 4 + i for i in range(self.max_where_num)]

            slen_index.append(elem + 4 + self.max_where_num)
            sai = [
                elem + 5 + self.max_where_num + i
                for i in range(self.max_select_num)
            ]
            where_op_index.append(woi)
            select_agg_index.append(sai)

            qilist = [i for i in range(l_n[ib] + 2)] + [l_n[ib] + 1] * (
                max_l_n - l_n[ib])
            query_index.append(qilist)

            index = [column_index[ib] + i for i in range(0, l_hs[ib], 1)]
            index += [index[0] for _ in range(max_l_hs - len(index))]
            header_pos_index.append(index)

        # print("tokens: ", tokens)
        # print("conn_index: ", conn_index, "start_index: ", start_index)
        conn_index = torch.tensor(conn_index, dtype=torch.long).to(self.device)
        slen_index = torch.tensor(slen_index, dtype=torch.long).to(self.device)
        wlen_index = torch.tensor(wlen_index, dtype=torch.long).to(self.device)
        action_index = torch.tensor(
            action_index, dtype=torch.long).to(self.device)
        where_op_index = torch.tensor(
            where_op_index, dtype=torch.long).to(self.device)
        select_agg_index = torch.tensor(
            select_agg_index, dtype=torch.long).to(self.device)
        query_index = torch.tensor(
            query_index, dtype=torch.long).to(self.device)
        header_index = torch.tensor(
            header_pos_index, dtype=torch.long).to(self.device)

        bS = len(l_n)
        conn_emb = torch.zeros([bS, self.iS]).to(self.device)
        slen_emb = torch.zeros([bS, self.iS]).to(self.device)
        wlen_emb = torch.zeros([bS, self.iS]).to(self.device)
        action_emb = torch.zeros([bS, self.iS]).to(self.device)
        wo_emb = torch.zeros([bS, self.max_where_num, self.iS]).to(self.device)
        sa_emb = torch.zeros([bS, self.max_select_num,
                              self.iS]).to(self.device)
        qv_emb = torch.zeros([bS, max_l_n + 2, self.iS]).to(self.device)
        ht_emb = torch.zeros([bS, max_l_hs, self.iS]).to(self.device)
        for i in range(bS):
            conn_emb[i, :] = wemb_layer[i].index_select(0, conn_index[i])
            slen_emb[i, :] = wemb_layer[i].index_select(0, slen_index[i])
            wlen_emb[i, :] = wemb_layer[i].index_select(0, wlen_index[i])
            action_emb[i, :] = wemb_layer[i].index_select(0, action_index[i])

            wo_emb[i, :, :] = wemb_layer[i].index_select(
                0, where_op_index[i, :])
            sa_emb[i, :, :] = wemb_layer[i].index_select(
                0, select_agg_index[i, :])
            qv_emb[i, :, :] = wemb_layer[i].index_select(0, query_index[i, :])
            ht_emb[i, :, :] = wemb_layer[i].index_select(0, header_index[i, :])

        s_cco = self.conn_model(conn_emb.reshape(-1, self.iS)).reshape(bS, 3)
        s_slen = self.slen_model(slen_emb.reshape(-1, self.iS)).reshape(
            bS, self.max_select_num + 1)
        s_wlen = self.wlen_model(wlen_emb.reshape(-1, self.iS)).reshape(
            bS, self.max_where_num + 1)
        s_action = self.action_model(action_emb.reshape(-1, self.iS)).reshape(
            bS, self.n_action_ops + 1)
        wo_output = self.w_op_model(
            wo_emb.reshape(-1, self.iS * self.max_where_num)).reshape(
                bS, -1, self.n_cond_ops)

        wc_output = self.wc_ht_model(ht_emb.reshape(-1, self.iS)).reshape(
            bS, -1, self.max_where_num).transpose(1, 2)

        wv_ss = self.w_sss_model(qv_emb.reshape(-1, self.iS)).reshape(
            bS, -1, self.max_where_num).transpose(1, 2)
        wv_se = self.w_sse_model(qv_emb.reshape(-1, self.iS)).reshape(
            bS, -1, self.max_where_num).transpose(1, 2)

        sc_output = self.s_ht_model(ht_emb.reshape(-1, self.iS)).reshape(
            bS, -1, self.max_select_num).transpose(1, 2)
        sa_output = self.select_agg_model(
            sa_emb.reshape(-1, self.iS * self.max_select_num)).reshape(
                bS, -1, self.n_agg_ops)

        return s_action, sc_output, sa_output, s_cco, wc_output, wo_output, (
            wv_ss, wv_se), (s_slen, s_wlen)
