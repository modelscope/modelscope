# Copyright 2021-2022 The Alibaba PAI Team Authors.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

import math
import os
from typing import Optional, Union

import addict
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_utils import PreTrainedModel

from modelscope.utils.constant import ModelFile
from .configuration import GPTMoEConfig


class GPTMoESelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        # Per attention head
        self.hidden_size_per_attention_head = \
            self.hidden_size // self.num_attention_heads

        self.query_key_value = nn.Linear(self.hidden_size,
                                         3 * self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(
            config.attention_probs_dropout_prob)

        # Output.
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads, self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _split_tensor_along_last_dim(self,
                                     tensor,
                                     num_partitions,
                                     contiguous_split_chunks=False):
        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def forward(self, hidden_states, ltor_mask, is_infer=False):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Attention heads. [b, s, hp]
        tgt_len = hidden_states.size(1)
        ltor_mask = torch.reshape(ltor_mask, [1, 1, tgt_len, tgt_len])
        mixed_x_layer = self.query_key_value(hidden_states)
        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = \
            self._split_tensor_along_last_dim(mixed_x_layer, 3)

        # Reshape and transpose [b, np, s, hn]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        previous_type = value_layer.type()

        # Raw attention scores. [b, np, s, s]
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(
            self.hidden_size_per_attention_head)
        # Apply the left to right attention mask.
        if is_infer:
            src_len = key_layer.size(2)
            ltor_mask = torch.tril(
                torch.ones((1, tgt_len, src_len),
                           device=hidden_states.device)).view(
                               1, 1, tgt_len, src_len).type(previous_type)
        converted_mask = 10000.0 * (1.0 - ltor_mask)
        attention_scores = (torch.mul(attention_scores, ltor_mask)
                            - converted_mask).type(previous_type)

        # Attention probabilities. [b, np, s, s]
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        # [b, np, s, hn]
        context_layer = torch.matmul(attention_probs, value_layer)
        # [b, s, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size, )
        # [b, s, hp]
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output. [b, s, h]
        output = self.dense(context_layer)
        output = self.output_dropout(output)

        return output


class GPTMoEMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super().__init__()

        hidden_size = config.hidden_size
        # Project to 4h.
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.activation_func = F.gelu
        # Project back to h.
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output = self.dropout(output)
        return output


class GPTMoETransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config):
        super().__init__()

        # Layernorm on the input data.
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

        # Self attention.
        self.attention = GPTMoESelfAttention(config)

        # Layernorm on the attention output
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

        # MLP
        self.mlp = GPTMoEMLP(config)

    def forward(self, hidden_states, ltor_mask):
        # hidden_states: [b, s, h]
        # ltor_mask: [1, 1, s, s]

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.attention(layernorm_output, ltor_mask)
        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output)
        # Second residual connection.
        output = layernorm_input + mlp_output

        return output


class GPTMoETransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super().__init__()

        self.input_tensor = None

        # Number of layers.
        self.num_layers = config.num_hidden_layers

        self.layers = torch.nn.ModuleList(
            [GPTMoETransformerLayer(config) for _ in range(self.num_layers)])

        # Final layer norm before output.
        self.final_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(self, hidden_states, attention_mask):
        # hidden_states: [s, b, h]

        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm.
        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class GPTMoETransformerLanguageModel(nn.Module):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self, config):
        super().__init__()

        # Embeddings.
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer.
        self.transformer = GPTMoETransformer(config)

    def forward(self, input_ids, attention_mask, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        transformer_input = self.embedding_dropout(embeddings)
        transformer_output = self.transformer(transformer_input,
                                              attention_mask)

        logits = F.linear(transformer_output, self.word_embeddings.weight)
        return logits


class GPTMoEModel(PreTrainedModel):

    config_class = GPTMoEConfig

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

    def __init__(self, config):
        super().__init__(config)
        self.language_model = GPTMoETransformerLanguageModel(config)

    def forward(self,
                input_ids,
                attention_mask=None,
                position_ids=None,
                labels=None,
                **kwargs):
        seq_length = input_ids.size(1)
        attention_mask = torch.tril(
            torch.ones((1, 1, seq_length, seq_length),
                       dtype=torch.long,
                       device=input_ids.device))
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        logits = self.language_model(input_ids, attention_mask, position_ids)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1))
        return addict.Dict(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]]):
        config = cls.config_class.from_pretrained(
            pretrained_model_name_or_path)
        model = cls(config)
        state_dict_file = os.path.join(pretrained_model_name_or_path,
                                       ModelFile.TORCH_MODEL_BIN_FILE)
        state_dict = torch.load(state_dict_file)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        state_dict = {
            k.replace('model.language_model', 'language_model'): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        return {'input_ids': input_ids}
