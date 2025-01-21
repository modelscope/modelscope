# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model."""

import copy
import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import (PreTrainedModel,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)
from transformers.utils import (DUMMY_INPUTS, DUMMY_MASK, add_start_docstrings,
                                add_start_docstrings_to_model_forward,
                                is_torch_fx_proxy, replace_return_docstrings)
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from modelscope.metainfo import Models
from modelscope.models.base import Model, Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput, Seq2SeqModelOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .configuration import T5Config

logger = get_logger()


###################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################
def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
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
    logger.info(f'Converting TensorFlow checkpoint from {tf_path}')
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f'Loading TF weight {name} with shape {shape}')
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in [
                'adam_v', 'adam_m', 'AdamWeightDecayOptimizer',
                'AdamWeightDecayOptimizer_1', 'global_step'
        ] for n in name):
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        if '_slot_' in name[-1]:
            logger.info(f"Skipping {'/'.join(name)}")
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]

        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                scope_names = re.split(r'_(\d+)', m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ['kernel', 'scale', 'embedding']:
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'self_attention':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[0]
            elif scope_names[0] == 'enc_dec_attention':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[1]
            elif scope_names[0] == 'dense_relu_dense':
                pointer = getattr(pointer, 'layer')
                pointer = pointer[2]
            elif scope_names[0] == 'rms_norm':
                if hasattr(pointer, 'layer_norm'):
                    pointer = getattr(pointer, 'layer_norm')
                elif hasattr(pointer, 'final_layer_norm'):
                    pointer = getattr(pointer, 'final_layer_norm')
            elif scope_names[0] == 'scale':
                pointer = getattr(pointer, 'weight')
            elif scope_names[0] == 'output_bias' or scope_names[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif scope_names[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            elif scope_names[0] == 'decoder' and name[1] == 'logits':
                continue
            elif scope_names[0] == 'logits':
                pointer = getattr(pointer, 'lm_head')
            elif scope_names[0] == 'wi' and len(
                    scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f'wi_{scope_names[1]}')
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ['kernel', 'scale', 'embedding']:
            pointer = getattr(pointer, 'weight')
        if scope_names[0] != 'embedding':
            logger.info(
                f'Transposing numpy weight of shape {array.shape} for {name}')
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f'Pointer shape {pointer.shape} and array shape {array.shape} mismatched'
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f'Initialize PyTorch weight {name}')
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info(
        f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}."
    )
    return model


class T5LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(
            -1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance
                                                    + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedGeluDense(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN['gelu_new']

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(self, config: T5Config):
        super().__init__()
        if config.feed_forward_proj == 'relu':
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == 'gated-gelu':
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f'{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`'
            )

        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):

    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(
                torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position,
                                           torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in
        # positions up to max_distance
        relateive_pos_log = torch.log(relative_position.float() / max_exact)
        max_dis_log = math.log(max_distance / max_exact)
        origin_relative_position = relateive_pos_log / max_dis_log * (
            num_buckets - max_exact)
        relative_postion_if_large = max_exact + origin_relative_position.to(
            torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1))

        relative_buckets += torch.where(is_small, relative_position,
                                        relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(
            key_length,
            dtype=torch.long,
            device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f'past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states'
            real_seq_length += past_key_value[0].shape[
                2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[
            1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads,
                               self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(
                batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states,
                    past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states],
                                              dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(
            hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states,
            past_key_value[0] if past_key_value is not None else None)
        value_states = project(
            hidden_states, self.v, key_value_states,
            past_key_value[1] if past_key_value is not None else None)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1):, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(
            scores.float(), dim=-1).type_as(
                scores)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(
            attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states,
                                   value_states) if (self.is_decoder
                                                     and use_cache) else None
        outputs = (attn_output, ) + (present_key_value_state, ) + (
            position_bias, )

        if output_attentions:
            outputs = outputs + (attn_weights, )
        return outputs


class T5LayerSelfAttention(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,
                   ) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    '`past_key_values` is passed to the encoder. Please make sure this is intended.'
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f'There should be {expected_num_past_key_values} past states. '
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f'Got {len(past_key_value)} past key / value states')

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(
                hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16 and torch.isinf(
                    hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[
                    1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(
                hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        if use_cache:
            outputs = outputs + (present_key_value_state, ) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        # hidden-states, present_key_value_states, (self-attention position
        # bias), (self-attention weights), (cross-attention position bias),
        # (cross-attention weights)
        return outputs


class T5PreTrainedModel(TorchModel, PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = 'transformer'
    is_parallelizable = True
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config.name_or_path, **kwargs)
        super(Model, self).__init__(config)

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            'decoder_input_ids': input_ids,
            'input_ids': input_ids,
            'decoder_attention_mask': input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, T5Model):
            # Mesh TensorFlow embeddings initialization See
            # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization See
            # https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and
            # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model)**-0.5))
            if hasattr(module.wi, 'bias') and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff)**-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model)**-0.5))
            if hasattr(module.wi_0, 'bias') and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model)**-0.5))
            if hasattr(module.wi_1, 'bias') and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff)**-0.5))
            if hasattr(module.wo, 'bias') and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before
            # softmax See
            # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim)**-0.5))
            module.k.weight.data.normal_(
                mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(
                mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim)**-0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(
                    mean=0.0, std=factor * ((d_model)**-0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), 'self.model.config.decoder_start_token_id has to be defined.'

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1, ),
                                           decoder_start_token_id)
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, 'self.model.config.pad_token_id has to be defined.'
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(
        ), 'Verify that `shifted_input_ids` has only positive values'

        return shifted_input_ids

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

        model_dir = kwargs.get('model_dir', None)
        if model_dir is None:
            config = T5Config(**kwargs)
            model = cls(config)
        else:
            model_kwargs = {}
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_kwargs)
        model.model_dir = model_dir
        return model


class T5Stack(T5PreTrainedModel):

    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=bool(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        r"""
        This is an experimental feature and is a subject to change at a
        moment's notice.

        Uses a device map to distribute attention modules of the model
        across several devices. If no device map is given, it will evenly
        distribute blocks across all devices.

        Args:
            device_map (`Dict[int, list]`, optional, defaults to None):
                A dictionary that maps attention modules to devices. Note
                that the embedding module and LMHead are always
                automatically mapped to the first device (for esoteric
                reasons). That means that the first device should have fewer
                attention modules mapped to it than other devices. For
                reference, the t5 models have the following number of
                attention modules:

                    - t5-small: 6
                    - t5-base: 12
                    - t5-large: 24
                    - t5-3b: 24
                    - t5-11b: 24

        Example:

        >>> # Here is an example of a device map on a machine with 4 GPUs
        >>> # using t5-3b, which has a total of 24 attention modules:
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-3b")
        >>> device_map = {
        >>>     0: [0, 1, 2], 1: [3, 4, 5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14,
        >>>     15, 16], 3: [17, 18, 19, 20, 21, 22, 23],
        >>> }
        >>> model.parallelize(device_map)
        >>> # all of the parallelize methods in this file are the same

        """
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None else device_map)
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = 'cpu' if 'cpu' in self.device_map.keys(
        ) else 'cuda:' + str(min(self.device_map.keys()))
        self.last_device = 'cuda:' + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = 'cuda:' + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        r"""
        Moves the model to cpu from a model parallel state.

        Example:

        >>> # On a 4 GPU machine with t5-3b:
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-3b")
        >>> device_map = {
        >>>     0: [0, 1, 2], 1: [3, 4, 5, 6, 7, 8, 9], 2: [10, 11, 12, 13, 14,
        >>>     15, 16], 3: [17, 18, 19, 20, 21, 22, 23],
        >>> }
        >>> model.parallelize(device_map)
        >>> # Splits the model across several devices model.deparallelize()
        >>> # Put the model back on cpu and
        >>> # cleans memory by calling torch.cuda.empty_cache()
        >>> # all of the deparallelize methods in this file are the same
        """
        self.model_parallel = False
        self.device_map = None
        self.first_device = 'cpu'
        self.last_device = 'cpu'
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to('cpu')
        self.embed_tokens = self.embed_tokens.to('cpu')
        self.final_layer_norm = self.final_layer_norm.to('cpu')
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(
                f'You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = 'decoder_' if self.is_decoder else ''
            raise ValueError(
                f'You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds'
            )

        if inputs_embeds is None:
            assert self.embed_tokens is not None, 'You have to initialize the model with valid token embeddings'
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[
            2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f'`use_cache` can only be set to `True` if {self} is used as a decoder'

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask,
                                                  self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions
                                      and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module,
                past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                        hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                    )
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return tuple(
                            module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with: hidden-states, key-value-states,
            # (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (
                    None, ) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer
            # store them layer_outputs = hidden-states, key-value-states
            # (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state, )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3], )
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[5], )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and 'cuda:' + str(k) != self.last_device:
                        hidden_states = hidden_states.to('cuda:' + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and
`decoder_head_mask`. Currently, `decoder_head_mask` is set to copy `head_mask`,
but this feature is deprecated and will be removed in future versions. If you do
not want to use any `decoder_head_mask` now, please set `decoder_head_mask =
torch.ones(num_layers, num_heads)`.
"""


@MODELS.register_module(group_key=Tasks.backbone, module_name=Models.T5)
class T5Model(T5PreTrainedModel):
    """The bare T5 Model transformer outputting raw hidden-states without any
    specific head on top.

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with
    a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by
    Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder
    transformer pre-trained in a text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass
    documentation for the generic methods the library implements for all its
    model (such as downloading or saving, resizing the input embeddings, pruning
    heads etc.)

    This model is also a PyTorch
    [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch
    documentation for all matter related to general usage and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters
        of the model.
            Initializing with a config file does not load the weights associated
            with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.
    """
    _keys_to_ignore_on_load_missing = [
        r'encoder\.embed_tokens\.weight',
        r'decoder\.embed_tokens\.weight',
    ]
    _keys_to_ignore_on_load_unexpected = [
        r'decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight',
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(
                len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None else device_map)
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to('cpu')
        self.decoder = self.decoder.to('cpu')
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of
        heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model
            with relative position embeddings so you should be able to pad the
            inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a
            look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size,
        sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask
            values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size,
        target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`T5Tokenizer`]. See
            [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`]
            for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for
            `decoder_input_ids` generation. If `past_key_values` is used,
            optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining
            take a look at [T5 Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size,
        target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in
            `decoder_input_ids`. Causal mask will also be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers,
        num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the
            encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or
        `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the
            decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or
        `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in
                the decoder. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*,
            `optional`: *attentions*) `last_hidden_state` of shape `(batch_size,
            sequence_length, hidden_size)` is a sequence of hidden states at the
            output of the last layer of the encoder. Used in the cross-attention
            of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length
        `config.n_layers` with each tuple having 4 tensors of shape
        `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention
            blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only
            the last `decoder_input_ids` (those that don't have their past key
            value states given to this model) of shape `(batch_size, 1)` instead
            of all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size,
        sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to
            directly pass an embedded representation. This is useful if you want
            more control over how to convert `input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size,
        target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to
            directly pass an embedded representation. If `past_key_values` is
            used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more
            control over how to convert `decoder_input_ids` indices into
            associated vectors than the model's internal embedding lookup
            matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset,
            `decoder_inputs_embeds` takes the value of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned
            and can be used to speed up decoding (see `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention
            layers. See `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See
            `hidden_states` under returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain
            tuple.
        Returns:

        Example:

        >>> from transformers import T5Tokenizer, T5Model

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs,
                                            AttentionBackboneModelOutput):
            encoder_outputs = AttentionBackboneModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1]
                if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2]
                if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
