# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import logging
import math

import torch
import torch.nn.functional as F
from megatron_util import mpu
from torch import nn

from modelscope.utils.nlp.distributed import (normal_init_method,
                                              scaled_init_method)
from .configuration import PlugNLGConfig, PlugNLUConfig

logger = logging.getLogger(__name__)


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
        self.word_embeddings = mpu.VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            init_method=normal_init_method(
                mean=0.0, std=config.initializer_range))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.fp32_layernorm = config.fp32_layernorm
        self.fp32_embedding = config.fp32_embedding
        self.fp32_tokentypes = config.fp32_tokentypes
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if not self.fp32_tokentypes:

            embeddings = words_embeddings + position_embeddings + token_type_embeddings
            if self.fp32_embedding and not self.fp32_layernorm:
                embeddings = embeddings.half()
            previous_type = embeddings.type()
            if self.fp32_layernorm:
                embeddings = embeddings.float()
            embeddings = self.LayerNorm(embeddings)
            if self.fp32_layernorm:
                if self.fp32_embedding:
                    embeddings = embeddings.half()
                else:
                    embeddings = embeddings.type(previous_type)
        else:
            embeddings = words_embeddings.float() + position_embeddings.float(
            ) + token_type_embeddings.float()
            if self.fp32_tokentypes and not self.fp32_layernorm:
                embeddings = embeddings.half()
            previous_type = embeddings.type()
            if self.fp32_layernorm:
                embeddings = embeddings.float()
            embeddings = self.LayerNorm(embeddings)
            if self.fp32_layernorm:
                if self.fp32_tokentypes:
                    embeddings = embeddings.half()
                else:
                    embeddings = embeddings.type(previous_type)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        if hasattr(config, 'deep_init') and config.deep_init:
            init_method = scaled_init_method(
                mean=0.0,
                std=config.initializer_range,
                num_layers=config.num_hidden_layers)
        else:
            init_method = normal_init_method(
                mean=0.0, std=config.initializer_range)
        self.dense = mpu.RowParallelLinear(
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            bias=True,
            input_is_parallel=True,
            stride=1,
            init_method=init_method)
        self.fp32_layernorm = config.fp32_layernorm
        if not config.pre_ln:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        input_tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        if self.LayerNorm is not None:
            previous_type = ln_input.type()
            if self.fp32_layernorm:
                ln_input = ln_input.float()
            hidden_states = self.LayerNorm(ln_input)
            if self.fp32_layernorm:
                hidden_states = hidden_states.type(previous_type)
        else:
            hidden_states = ln_input
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.fp32_layernorm = config.fp32_layernorm
        if config.pre_ln:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.self = mpu.BertParallelSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_probs_dropout_prob,
            output_parallel=True,
            init_method=normal_init_method(
                mean=0.0, std=config.initializer_range),
            separate=config.attn_separate)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor,
        attention_mask,
    ):
        if self.LayerNorm is not None:
            ln_input = input_tensor
            previous_type = input_tensor.type()
            if self.fp32_layernorm:
                ln_input = input_tensor.float()
            ln_output = self.LayerNorm(ln_input)
            if self.fp32_layernorm:
                ln_output = ln_output.type(previous_type)
            self_output = self.self(
                ln_output,
                attention_mask,
            )
        else:
            self_output = self.self(
                input_tensor,
                attention_mask,
            )

        attention_output = self.output(
            self_output,
            input_tensor,
        )
        return attention_output


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = mpu.ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.intermediate_size,
            bias=True,
            gather_output=False,
            stride=1,
            init_method=normal_init_method(
                mean=0.0, std=config.initializer_range))
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super(BertOutput, self).__init__()
        if hasattr(config, 'deep_init') and config.deep_init:
            init_method = scaled_init_method(
                mean=0.0,
                std=config.initializer_range,
                num_layers=config.num_hidden_layers)
        else:
            init_method = normal_init_method(
                mean=0.0, std=config.initializer_range)
        self.dense = mpu.RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=True,
            input_is_parallel=True,
            stride=1,
            init_method=init_method)
        self.fp32_layernorm = config.fp32_layernorm
        if not config.pre_ln:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states,
        input_tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        ln_input = hidden_states + input_tensor
        if self.LayerNorm is not None:
            previous_type = ln_input.type()
            if self.fp32_layernorm:
                ln_input = ln_input.float()
            hidden_states = self.LayerNorm(ln_input)
            if self.fp32_layernorm:
                hidden_states = hidden_states.type(previous_type)
        else:
            hidden_states = ln_input
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.fp32_layernorm = config.fp32_layernorm
        if config.pre_ln:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        if self.LayerNorm is not None:
            ln_input = attention_output
            previous_type = attention_output.type()
            if self.fp32_layernorm:
                ln_input = attention_output.float()
            ln_output = self.LayerNorm(ln_input)
            if self.fp32_layernorm:
                ln_output = ln_output.type(previous_type)
            intermediate_output = self.intermediate(ln_output)
        else:
            intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.fp32_layernorm = config.fp32_layernorm
        if config.pre_ln:
            self.LayerNorm = BertLayerNorm(
                config.hidden_size, eps=config.layernorm_epsilon)
        else:
            self.LayerNorm = None

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        detach_index=-1,
    ):
        all_encoder_layers = []

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_

            return custom_forward

        if checkpoint_activations:
            layer_idx = 0
            num_layers = len(self.layer)
            chunk_length = 1
            while layer_idx < num_layers:
                hidden_states = mpu.checkpoint(
                    custom(layer_idx, layer_idx + chunk_length), hidden_states,
                    attention_mask * 1)
                if detach_index == layer_idx:
                    hidden_states.detach_()
                layer_idx += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
                if detach_index == i:
                    hidden_states.detach_()
                if i == len(self.layer) - 1 and self.LayerNorm is not None:
                    previous_type = hidden_states.type()
                    if self.fp32_layernorm:
                        hidden_states = hidden_states.float()
                    hidden_states = self.LayerNorm(hidden_states)
                    if self.fp32_layernorm:
                        hidden_states = hidden_states.type(previous_type)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not output_all_encoded_layers or checkpoint_activations:
            if self.LayerNorm is not None:
                previous_type = hidden_states.type()
                if self.fp32_layernorm:
                    hidden_states = hidden_states.float()
                hidden_states = self.LayerNorm(hidden_states)
                if self.fp32_layernorm:
                    hidden_states = hidden_states.type(previous_type)
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
        self.LayerNorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.fp32_layernorm = config.fp32_layernorm

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        previous_type = hidden_states.type()
        if self.fp32_layernorm:
            hidden_states = hidden_states.float()
        hidden_states = self.LayerNorm(hidden_states)
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        return hidden_states


class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_weight = bert_model_embedding_weights
        self.bias = nn.Parameter(
            torch.zeros(bert_model_embedding_weights.size(0)))
        self.bias.model_parallel = True
        self.fp32_embedding = config.fp32_embedding
        self.fp32_layernorm = config.fp32_layernorm

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
                if self.fp32_layernorm:
                    self.transform.LayerNorm.float()
        hidden_states = self.transform(self.type_converter(hidden_states))
        hidden_states = mpu.copy_to_model_parallel_region(hidden_states)
        hidden_states = F.linear(
            self.type_converter(hidden_states),
            self.type_converter(self.decoder_weight),
            self.type_converter(self.bias))
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 3)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        for p in self.seq_relationship.parameters():
            if p is None:
                continue
            pooled_output = pooled_output.type_as(p)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, PlugNLUConfig) and not isinstance(
                config, PlugNLGConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `BertConfig`. '
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


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

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
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as
            described below. Default: `True`.

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

    Examples:
        >>> # Already been converted into WordPiece token ids
        >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        >>> config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        >>>     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        >>> model = modeling.BertModel(config=config)
        >>> all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
        detach_index=-1,
    ):
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
            dtype=next(self.encoder.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
            detach_index=detach_index)
        sequence_output = encoded_layers[-1]
        for p in self.pooler.parameters():
            if p is None:
                continue
            sequence_output = sequence_output.type_as(p)
            break

        pooled_output = sequence_output[:, 0]
        if not output_all_encoded_layers or checkpoint_activations:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class DecodeLayer(nn.Module):

    def __init__(self, config):
        super(DecodeLayer, self).__init__()
        init_method = normal_init_method(
            mean=0.0, std=config.initializer_range)
        output_layer_init_method = scaled_init_method(
            mean=0.0,
            std=config.initializer_range,
            num_layers=config.num_hidden_layers)

        self.attention = mpu.GPT2ParallelSelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout_prob=config.attention_probs_dropout_prob,
            output_dropout_prob=config.hidden_dropout_prob,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
        )

        self.cross_attention = mpu.PalmParallelCrossAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            attention_dropout_prob=config.attention_probs_dropout_prob,
            output_dropout_prob=config.hidden_dropout_prob,
            init_method=init_method,
            attn_separate=False,
            output_layer_init_method=output_layer_init_method,
        )

        self.input_layernorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.post_cross_attention_layernorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)

        self.intermediate = mpu.ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            gather_output=False,
            init_method=init_method,
        )
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.output = mpu.RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
        )

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.fp32_layernorm = config.fp32_layernorm

        def convert_to_type(tensor):
            if self.fp32_layernorm:
                return tensor.float()
            else:
                return tensor

        self.type_converter = convert_to_type

    # def forward(self, hidden_states, enc_attn_mask, dec_attn_mask):
    def forward(self,
                hidden_states,
                enc_hidden_states,
                enc_attn_mask,
                dec_attn_mask,
                is_infer=False):
        residual = hidden_states
        previous_type = hidden_states.type()
        hidden_states = self.input_layernorm(
            self.type_converter(hidden_states))
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        hidden_states = self.attention(
            hidden_states, dec_attn_mask, is_infer=is_infer)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            self.type_converter(hidden_states))
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        hidden_states = self.cross_attention(hidden_states, enc_hidden_states,
                                             enc_attn_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(
            self.type_converter(hidden_states))
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BertDecoder(nn.Module):

    def __init__(self, config):
        super(BertDecoder, self).__init__()
        self.layer = nn.ModuleList(
            [DecodeLayer(config) for _ in range(config.dec_hidden_layers)])

        self.final_layernorm = BertLayerNorm(
            config.hidden_size, eps=config.layernorm_epsilon)
        self.fp32_layernorm = config.fp32_layernorm

    def forward(self,
                hidden_states,
                enc_hidden_states,
                enc_attn_mask,
                dec_attn_mask,
                checkpoint_activations=False,
                output_all_encoded_layers=False,
                is_infer=False):

        def custom(start, end):

            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(
                        x_,
                        inputs[1],
                        inputs[2],
                        dec_attn_mask * 1,
                        is_infer=is_infer)
                return x_

            return custom_forward

        pre_enc_hidden = enc_hidden_states.data
        if checkpoint_activations:
            layer_idx = 0
            num_layers = len(self.layer)
            chunk_length = 1
            while layer_idx < num_layers:
                hidden_states = mpu.checkpoint(
                    custom(layer_idx, layer_idx + chunk_length), hidden_states,
                    enc_hidden_states, enc_attn_mask * 1)
                enc_hidden_states.data = pre_enc_hidden
                layer_idx += chunk_length
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states,
                    enc_hidden_states,
                    enc_attn_mask,
                    dec_attn_mask,
                    is_infer=is_infer)

        previous_type = hidden_states.type()
        if self.fp32_layernorm:
            hidden_states = hidden_states.float()
        hidden_states = self.final_layernorm(hidden_states)
        if self.fp32_layernorm:
            hidden_states = hidden_states.type(previous_type)

        return [hidden_states]


class DecodeModel(PreTrainedBertModel):

    def __init__(self, config):
        super(DecodeModel, self).__init__(config)
        self.decoder = BertDecoder(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                embeddings,
                sequence_output,
                decode_input_ids,
                position_ids=None,
                enc_attn_mask=None,
                dec_attn_mask=None,
                checkpoint_activations=False,
                is_infer=False):
        extended_attention_mask = enc_attn_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.decoder.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = embeddings(decode_input_ids)
        sequence_output = self.decoder(
            embedding_output,
            sequence_output,
            extended_attention_mask,
            dec_attn_mask,
            checkpoint_activations=False,
            is_infer=is_infer)
        return sequence_output[-1]


class PalmForPreTraining(PreTrainedBertModel):

    def __init__(self, config):
        super(PalmForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.decoder = DecodeModel(config)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                decode_input_ids=None,
                position_ids=None,
                decode_attention_mask=None,
                lm_labels=None,
                checkpoint_activations=False,
                is_infer=False,
                sequence_output=None,
                parallel_output=True):
        if sequence_output is None:
            sequence_output, pooled_output = self.bert(
                input_ids,
                token_type_ids,
                attention_mask,
                output_all_encoded_layers=False,
                checkpoint_activations=checkpoint_activations)
            prediction_scores, seq_relationship_score = self.cls(
                sequence_output, pooled_output)
        else:
            prediction_scores = None
            sequence_output = sequence_output.to(
                dtype=next(self.decoder.parameters()).dtype)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        decode_output = self.decoder(
            self.bert.embeddings,
            sequence_output,
            decode_input_ids,
            position_ids,
            attention_mask,
            decode_attention_mask,
            checkpoint_activations=checkpoint_activations,
            is_infer=is_infer)

        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            decode_output)

        logits_parallel = F.linear(transformer_output_parallel,
                                   self.bert.embeddings.word_embeddings.weight)

        if parallel_output:
            return prediction_scores, logits_parallel
        if is_infer:
            return prediction_scores, mpu.gather_from_model_parallel_region(
                logits_parallel), sequence_output
        return prediction_scores, mpu.gather_from_model_parallel_region(
            logits_parallel)


class PlugModel(torch.nn.Module):
    """
    The bare Plug Model transformer outputting raw hidden-states without any specific head on top.
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`PlugNLGConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~DistributedPlug.initialize_model`] method to load the model weights.
    Examples:

    >>> # The PLUG model has 27B parameters and usually need to run on multiple GPUs. The example given
    >>> # here only initializes a slice of the model on a single GPU.
    >>> # Check out the [`~DistributedPipeline.__init__`] method to initialize entire PLUG model.
    >>> from modelscope.models.nlp.plug import PlugNLGConfig, PlugModel

    >>> # Initializing a Plug configuration
    >>> configuration = PlugNLGConfig()

    >>> # Initializing a model from the configuration
    >>> model = PlugModel(configuration)
    """

    def __init__(self, config):
        super(PlugModel, self).__init__()
        self.config = config
        self.model = PalmForPreTraining(self.config)

    def forward(self,
                input_tokens,
                token_type_ids=None,
                attention_mask=None,
                target_tokens=None,
                position_ids=None,
                decode_attention_mask=None,
                checkpoint_activations=False,
                is_infer=False,
                sequence_output=None,
                parallel_output=True):
        """
        Parameters:
            input_tokens (`torch.LongTensor` of shape `(batch_size, input_tokens_length)`):
                `input_tokens_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using transformers [`BertTokenizer`]. See
                [`TextGenerationPreprocessor.__call__`] for details.
            token_type_ids (`torch.LongTensor` of shape `(batch_size, input_tokens_length)`, *optional*, defaults to
            None):
               Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
               1]`:

               - 0 corresponds to a *sentence A* token,
               - 1 corresponds to a *sentence B* token.

            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to None):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            target_tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to None):
                Target token ids(labels) for language modeling. Note that the labels **are shifted** inside the model,
                i.e. you can set `target_tokens = input_tokens` Indices are selected in
                `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only
                computed for labels in `[0, ..., config.vocab_size]`

            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults to None):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.max_position_embeddings - 1]`.

            decode_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*, defaults
            to None):
                Mask to avoid performing attention on padding token indices of target tokens. Mask values selected in
                `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            checkpoint_activations (`boolean`, *optional*, defaults to `False`):
                Whether gradient checkpointing is activated for this model or not.
            is_infer (`boolean`, *optional*, defaults to `False`):
                Whether or not to perform single inference.
            sequence_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*,
            defaults to None):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model. A single forward() call can produce one single token. To generate the current token, the
                sequence_output generated by the `forward()` of the previous token is required.
            parallel_output (`boolean`, *optional*, defaults to `True`):
                To parallel return output, or gather it before return.


        """
        return self.model(
            input_tokens,
            token_type_ids,
            attention_mask,
            target_tokens,
            position_ids,
            decode_attention_mask,
            checkpoint_activations=checkpoint_activations,
            is_infer=is_infer,
            sequence_output=sequence_output,
            parallel_output=parallel_output)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
