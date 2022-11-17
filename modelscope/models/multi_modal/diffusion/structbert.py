# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team and Alibaba inc.
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

import json
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 emb_size=-1,
                 num_hidden_layers=12,
                 transformer_type='original',
                 transition_function='linear',
                 weighted_transformer=0,
                 num_rolled_layers=3,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 attention_type='self',
                 rezero=False,
                 pre_ln=False,
                 squeeze_excitation=False,
                 transfer_matrix=False,
                 dim_dropout=False,
                 roberta_style=False,
                 set_mask_zero=False,
                 init_scale=False,
                 safer_fp16=False,
                 grad_checkpoint=False):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.num_hidden_layers = num_hidden_layers
        self.transformer_type = transformer_type
        self.transition_function = transition_function
        self.weighted_transformer = weighted_transformer
        self.num_rolled_layers = num_rolled_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.attention_type = attention_type
        self.rezero = rezero
        self.pre_ln = pre_ln
        self.squeeze_excitation = squeeze_excitation
        self.transfer_matrix = transfer_matrix
        self.dim_dropout = dim_dropout
        self.set_mask_zero = set_mask_zero
        self.roberta_style = roberta_style
        self.init_scale = init_scale
        self.safer_fp16 = safer_fp16
        self.grad_checkpoint = grad_checkpoint

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class BERTLayerNorm(nn.Module):

    def __init__(self, config, variance_epsilon=1e-12, special_size=None):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.config = config
        hidden_size = special_size if special_size is not None else config.hidden_size
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon if not config.roberta_style else 1e-5

    def forward(self, x):
        previous_type = x.type()
        if self.config.safer_fp16:
            x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        if self.config.safer_fp16:
            return (self.gamma * x + self.beta).type(previous_type)
        else:
            return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):

    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        hidden_size = config.hidden_size if config.emb_size < 0 else config.emb_size
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            hidden_size,
            padding_idx=1 if config.roberta_style else None)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            hidden_size,
            padding_idx=1 if config.roberta_style else None)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  hidden_size)
        self.config = config
        self.proj = None if config.emb_size < 0 else nn.Linear(
            config.emb_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config, special_size=hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, adv_embedding=None):
        seq_length = input_ids.size(1)
        if not self.config.roberta_style:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        else:
            mask = input_ids.ne(1).int()
            position_ids = (torch.cumsum(mask, dim=1).type_as(mask)
                            * mask).long() + 1
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(
            input_ids) if adv_embedding is None else adv_embedding
        if self.config.set_mask_zero:
            words_embeddings[input_ids == 103] = 0.
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if not self.config.roberta_style:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        if self.proj is not None:
            embeddings = self.proj(embeddings)
            embeddings = self.dropout(embeddings)
        else:
            return embeddings, words_embeddings


class BERTFactorizedAttention(nn.Module):

    def __init__(self, config):
        super(BERTFactorizedAttention, self).__init__()
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

    def transpose_for_scores(self, x, *size):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(size)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, 0, 2, 3, 1)
        key_layer = self.transpose_for_scores(mixed_key_layer, 0, 2, 1, 3)
        value_layer = self.transpose_for_scores(mixed_value_layer, 0, 2, 1, 3)

        s_attention_scores = query_layer + attention_mask
        s_attention_probs = nn.Softmax(dim=-1)(s_attention_scores)
        s_attention_probs = self.dropout(s_attention_probs)

        c_attention_probs = nn.Softmax(dim=-1)(key_layer)
        s_context_layer = torch.matmul(s_attention_probs, value_layer)
        context_layer = torch.matmul(c_attention_probs, s_context_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


def dim_dropout(x, p=0, dim=-1, training=False):
    if not training or p == 0:
        return x
    a = (1 - p)
    b = (x.data.new(x.size()).zero_() + 1)
    dropout_mask = torch.bernoulli(a * b)
    return dropout_mask * (dropout_mask.size(dim) / torch.sum(
        dropout_mask, dim=dim, keepdim=True)) * x


class BERTSelfAttention(nn.Module):

    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
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
        self.config = config
        if config.pre_ln:
            self.LayerNorm = BERTLayerNorm(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        if self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
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
        if head_mask is not None and not self.training:
            for i, mask in enumerate(head_mask):
                if head_mask[i] == 1:
                    attention_scores[:, i, :, :] = 0.
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.config.dim_dropout:
            attention_probs = self.dropout(attention_probs)
        else:
            attention_probs = dim_dropout(
                attention_probs,
                p=self.config.attention_probs_dropout_prob,
                dim=-1,
                training=self.training)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):

    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if not config.pre_ln and not config.rezero:
            self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.rezero:
            self.res_factor = nn.Parameter(
                torch.Tensor(1).fill_(0.99).to(
                    dtype=next(self.parameters()).dtype))
            self.factor = nn.Parameter(
                torch.ones(1).to(dtype=next(self.parameters()).dtype))

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if not self.config.rezero and not self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.config.rezero:
            hidden_states = hidden_states + self.factor * input_tensor
        else:
            pass
        return hidden_states


class BERTAttention(nn.Module):

    def __init__(self, config):
        super(BERTAttention, self).__init__()
        if config.attention_type.lower() == 'self':
            self.self = BERTSelfAttention(config)
        elif config.attention_type.lower() == 'factorized':
            self.self = BERTFactorizedAttention(config)
        else:
            raise ValueError(
                'Attention type must in [self, factorized], but got {}'.format(
                    config.attention_type))
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None):
        self_output = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class DepthwiseSeparableConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(DepthwiseSeparableConv1d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias)
        self.pointwise = nn.Conv1d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BERTIntermediate(nn.Module):

    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.config = config
        if self.config.pre_ln:
            self.LayerNorm = BERTLayerNorm(config)
        self.intermediate_act_fn = gelu
        if config.transition_function.lower() == 'linear':
            self.dense = nn.Linear(config.hidden_size,
                                   config.intermediate_size)
        elif config.transition_function.lower() == 'cnn':
            self.cnn = DepthwiseSeparableConv1d(
                config.hidden_size, 4 * config.hidden_size, kernel_size=7)
        elif config.config.hidden_size.lower() == 'rnn':
            raise NotImplementedError(
                'rnn transition function is not implemented yet')
        else:
            raise ValueError('Only support linear/cnn/rnn')

    def forward(self, hidden_states):
        if self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states)
        if self.config.transition_function.lower() == 'linear':
            hidden_states = self.dense(hidden_states)
        elif self.config.transition_function.lower() == 'cnn':
            hidden_states = self.cnn(hidden_states.transpose(-1,
                                                             -2)).transpose(
                                                                 -1, -2)
        else:
            pass
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SqueezeExcitationBlock(nn.Module):

    def __init__(self, config):
        super(SqueezeExcitationBlock, self).__init__()
        self.down_sampling = nn.Linear(config.hidden_size,
                                       config.hidden_size // 4)
        self.up_sampling = nn.Linear(config.hidden_size // 4,
                                     config.hidden_size)

    def forward(self, hidden_states):
        squeeze = torch.mean(hidden_states, 1, keepdim=True)
        excitation = torch.sigmoid(
            self.up_sampling(gelu(self.down_sampling(squeeze))))
        return hidden_states * excitation


class BERTOutput(nn.Module):

    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.config = config
        if config.transition_function.lower() == 'linear':
            self.dense = nn.Linear(config.intermediate_size,
                                   config.hidden_size)
        elif config.transition_function.lower() == 'cnn':
            self.cnn = DepthwiseSeparableConv1d(
                4 * config.hidden_size, config.hidden_size, kernel_size=7)
        elif config.config.hidden_size.lower() == 'rnn':
            raise NotImplementedError(
                'rnn transition function is not implemented yet')
        else:
            raise ValueError('Only support linear/cnn/rnn')
        if not config.pre_ln and not config.rezero:
            self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.squeeze_excitation:
            self.SEblock = SqueezeExcitationBlock(config)
        if config.rezero:
            self.res_factor = nn.Parameter(
                torch.Tensor(1).fill_(0.99).to(
                    dtype=next(self.parameters()).dtype))
            self.factor = nn.Parameter(
                torch.ones(1).to(dtype=next(self.parameters()).dtype))

    def forward(self, hidden_states, input_tensor):
        if self.config.transition_function.lower() == 'linear':
            hidden_states = self.dense(hidden_states)
        elif self.config.transition_function.lower() == 'cnn':
            hidden_states = self.cnn(hidden_states.transpose(-1,
                                                             -2)).transpose(
                                                                 -1, -2)
        else:
            pass
        hidden_states = self.dropout(hidden_states)
        if self.config.squeeze_excitation:
            hidden_states = self.SEblock(hidden_states)
        if not self.config.rezero and not self.config.pre_ln:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        elif self.config.rezero:
            hidden_states = hidden_states + self.factor * input_tensor
        else:
            pass
        return hidden_states


class BERTLayer(nn.Module):

    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_output = self.attention(hidden_states, attention_mask,
                                          head_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return attention_output, layer_output


class BERTWeightedLayer(nn.Module):

    def __init__(self, config):
        super(BERTWeightedLayer, self).__init__()
        self.config = config
        self.self = BERTSelfAttention(config)
        self.attention_head_size = self.self.attention_head_size

        self.w_o = nn.ModuleList([
            nn.Linear(self.attention_head_size, config.hidden_size)
            for _ in range(config.num_attention_heads)
        ])
        self.w_kp = torch.rand(config.num_attention_heads)
        self.w_kp = nn.Parameter(self.w_kp / self.w_kp.sum())
        self.w_a = torch.rand(config.num_attention_heads)
        self.w_a = nn.Parameter(self.w_a / self.w_a.sum())

        self.intermediate = BERTIntermediate(config)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        self_outputs = self_output.split(self.self.attention_head_size, dim=-1)
        self_outputs = [
            self.w_o[i](self_outputs[i]) for i in range(len(self_outputs))
        ]
        self_outputs = [
            self.dropout(self_outputs[i]) for i in range(len(self_outputs))
        ]
        self_outputs = [
            kappa * output for kappa, output in zip(self.w_kp, self_outputs)
        ]
        self_outputs = [
            self.intermediate(self_outputs[i])
            for i in range(len(self_outputs))
        ]
        self_outputs = [
            self.output(self_outputs[i]) for i in range(len(self_outputs))
        ]
        self_outputs = [
            self.dropout(self_outputs[i]) for i in range(len(self_outputs))
        ]
        self_outputs = [
            alpha * output for alpha, output in zip(self.w_a, self_outputs)
        ]
        output = sum(self_outputs)
        return self.LayerNorm(hidden_states + output)


class BERTEncoder(nn.Module):

    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        self.layer = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            if config.weighted_transformer:
                self.layer.append(BERTWeightedLayer(config))
            else:
                self.layer.append(BERTLayer(config))
        if config.rezero:
            for index, layer in enumerate(self.layer):
                layer.output.res_factor = nn.Parameter(
                    torch.Tensor(1).fill_(1.).to(
                        dtype=next(self.parameters()).dtype))
                layer.output.factor = nn.Parameter(
                    torch.Tensor(1).fill_(1).to(
                        dtype=next(self.parameters()).dtype))
                layer.attention.output.res_factor = layer.output.res_factor
                layer.attention.output.factor = layer.output.factor
        self.config = config

    def forward(self,
                hidden_states,
                attention_mask,
                epoch_id=-1,
                head_masks=None):
        all_encoder_layers = [hidden_states]
        if epoch_id != -1:
            detach_index = int(len(self.layer) / 3) * (2 - epoch_id) - 1
        else:
            detach_index = -1
        for index, layer_module in enumerate(self.layer):
            if head_masks is None:
                if not self.config.grad_checkpoint:
                    self_out, hidden_states = layer_module(
                        hidden_states, attention_mask, None)
                else:
                    self_out, hidden_states = torch.utils.checkpoint.checkpoint(
                        layer_module, hidden_states, attention_mask, None)
            else:
                self_out, hidden_states = layer_module(hidden_states,
                                                       attention_mask,
                                                       head_masks[index])
            if detach_index == index:
                hidden_states.detach_()
            all_encoder_layers.append(self_out)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTEncoderRolled(nn.Module):

    def __init__(self, config):
        super(BERTEncoderRolled, self).__init__()
        layer = BERTLayer(config)
        self.config = config
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_rolled_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                epoch_id=-1,
                head_masks=None):
        all_encoder_layers = [hidden_states]
        for i in range(self.config.num_hidden_layers):
            if self.config.transformer_type.lower() == 'universal':
                hidden_states = self.layer[i % self.config.num_rolled_layers](
                    hidden_states, attention_mask)
            elif self.config.transformer_type.lower() == 'albert':
                a = i // (
                    self.config.num_hidden_layers
                    // self.config.num_rolled_layers)
                hidden_states = self.layer[a](hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTEncoderACT(nn.Module):

    def __init__(self, config):
        super(BERTEncoderACT, self).__init__()
        self.layer = BERTLayer(config)
        p = nn.Linear(config.hidden_size, 1)
        self.p = nn.ModuleList(
            [copy.deepcopy(p) for _ in range(config.num_hidden_layers)])
        # Following act paper, set bias init ones
        for module in self.p:
            module.bias.data.fill_(1.)
        self.config = config
        self.act_max_steps = config.num_hidden_layers
        self.threshold = 0.99

    def should_continue(self, halting_probability, n_updates):
        return (halting_probability.lt(self.threshold).__and__(
            n_updates.lt(self.act_max_steps))).any()

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        batch_size, seq_len, hdim = hidden_states.size()
        halting_probability = torch.zeros(batch_size, seq_len).cuda()
        remainders = torch.zeros(batch_size, seq_len).cuda()
        n_updates = torch.zeros(batch_size, seq_len).cuda()
        for i in range(self.act_max_steps):
            p = torch.sigmoid(self.p[i](hidden_states).squeeze(2))
            still_running = halting_probability.lt(1.0).float()
            new_halted = (halting_probability + p * still_running).gt(
                self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running).le(
                self.threshold).float() * still_running
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = (p * still_running
                              + new_halted * remainders).unsqueeze(2)
            transformed_states = self.layer(hidden_states, attention_mask)
            hidden_states = transformed_states * update_weights + hidden_states * (
                1 - update_weights)
            all_encoder_layers.append(hidden_states)
            if not self.should_continue(halting_probability, n_updates):
                break
        return all_encoder_layers, torch.mean(n_updates + remainders)


class BERTPooler(nn.Module):

    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.config = config
        self.embeddings = BERTEmbeddings(config)
        if config.transformer_type.lower() == 'original':
            self.encoder = BERTEncoder(config)
        elif config.transformer_type.lower() == 'universal':
            self.encoder = BERTEncoderRolled(config)
        elif config.transformer_type.lower() == 'albert':
            self.encoder = BERTEncoderRolled(config)
        elif config.transformer_type.lower() == 'act':
            self.encoder = BERTEncoderACT(config)
        elif config.transformer_type.lower() == 'textnas':
            from textnas_final import input_dict, op_dict, skip_dict
            self.encoder = TextNASEncoder(config, op_dict, input_dict,
                                          skip_dict)
        else:
            raise ValueError('Not support transformer type: {}'.format(
                config.transformer_type.lower()))
        self.pooler = BERTPooler(config)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                epoch_id=-1,
                head_masks=None,
                adv_embedding=None):
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
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output, word_embeddings = self.embeddings(
            input_ids, token_type_ids, adv_embedding)
        if self.config.transformer_type.lower() == 'act':
            all_encoder_layers, act_loss = self.encoder(
                embedding_output, extended_attention_mask)
        elif self.config.transformer_type.lower() == 'reformer':
            sequence_output = self.encoder(embedding_output)
            all_encoder_layers = [sequence_output, sequence_output]
        else:
            all_encoder_layers = self.encoder(embedding_output,
                                              extended_attention_mask,
                                              epoch_id, head_masks)
        all_encoder_layers.insert(0, word_embeddings)
        sequence_output = all_encoder_layers[-1]
        if not self.config.safer_fp16:
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = sequence_output[:, 0]
        return all_encoder_layers, pooled_output


class BertForSequenceClassificationMultiTask(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, label_list, core_encoder):
        super(BertForSequenceClassificationMultiTask, self).__init__()
        if core_encoder.lower() == 'bert':
            self.bert = BertModel(config)
        elif core_encoder.lower() == 'lstm':
            self.bert = LSTMModel(config)
        else:
            raise ValueError(
                'Only support lstm or bert, but got {}'.format(core_encoder))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.ModuleList()
        for label in label_list:
            self.classifier.append(nn.Linear(config.hidden_size, len(label)))
        self.label_list = label_list

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(
                    mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(
                    mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(
                    mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                labels=None,
                labels_index=None,
                epoch_id=-1,
                head_masks=None,
                adv_embedding=None,
                return_embedding=False,
                loss_weight=None):
        all_encoder_layers, pooled_output = self.bert(input_ids,
                                                      token_type_ids,
                                                      attention_mask, epoch_id,
                                                      head_masks,
                                                      adv_embedding)
        pooled_output = self.dropout(pooled_output)
        logits = [classifier(pooled_output) for classifier in self.classifier]
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none')
            regression_loss_fct = nn.MSELoss(reduction='none')
            labels_lst = torch.unbind(labels, 1)
            loss_lst = []
            for index, (label, logit) in enumerate(zip(labels_lst, logits)):
                if len(self.label_list[index]) != 1:
                    loss = loss_fct(logit, label.long())
                else:
                    loss = regression_loss_fct(logit.squeeze(-1), label)
                labels_mask = (labels_index == index).to(
                    dtype=next(self.parameters()).dtype)
                if loss_weight is not None:
                    loss = loss * loss_weight[index]
                loss = torch.mean(loss * labels_mask)
                loss_lst.append(loss)
            if not return_embedding:
                return sum(loss_lst), logits
            else:
                return sum(loss_lst), logits, all_encoder_layers[0]
        else:
            return logits
