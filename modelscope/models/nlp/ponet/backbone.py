# Copyright 2021-2022 The Alibaba DAMO Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# All rights reserved.
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
"""PyTorch PoNet model. """

import math
from distutils.version import LooseVersion

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import (PreTrainedModel,
                                         apply_chunking_to_forward,
                                         find_pruneable_heads_and_indices,
                                         prune_linear_layer)

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionBackboneModelOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from .configuration import PoNetConfig

logger = get_logger()

is_pytorch_12plus = LooseVersion(torch.__version__) >= LooseVersion('1.12.0')

CLS_ID = 101
EOS_ID = 102


def segment_max(src, index, dim=1):
    if is_pytorch_12plus:
        out = torch.zeros_like(src).scatter_reduce(
            dim,
            index[:, :, None].expand_as(src),
            src,
            reduce='amax',
            include_self=False)
    else:
        dummy_scatter_index = index[:, :, None].expand_as(src)
        min_value = src.min() - 1
        dummpy_scatter_shape = (*src.shape[:-1], index.max() + 1,
                                src.shape[-1])
        dummy_scatter_index_expand = dummy_scatter_index.unsqueeze(-2).expand(
            *dummpy_scatter_shape)
        index_reconstruct_expand = torch.arange(
            index.max() + 1,
            device=src.device)[None, None, :,
                               None].expand(*dummpy_scatter_shape)
        src_expand = src.unsqueeze(-2).expand(*dummpy_scatter_shape)
        out, _ = src_expand.masked_scatter(
            dummy_scatter_index_expand != index_reconstruct_expand,
            torch.full_like(src_expand, min_value.item())).max(dim=1)

    dummy = index.unsqueeze(-1).expand(*index.shape[:2], out.size(-1))
    return torch.gather(out, dim, dummy).to(dtype=src.dtype)


def get_segment_index(input_ids, cls_id=CLS_ID, eos_id=EOS_ID):
    mask = (input_ids == cls_id).to(
        dtype=torch.long) + (input_ids == eos_id).to(dtype=torch.long)
    mask = mask + torch.cat([torch.zeros_like(mask[:, 0:1]), mask[:, :-1]],
                            dim=1)
    return mask.cumsum(dim=1) - 1


def get_token_type_mask(input_ids, cls_id=CLS_ID, eos_id=EOS_ID):
    mask = (input_ids == cls_id) | (input_ids == eos_id)
    return mask


def get_win_max(hidden_states, kernel_size=3):
    m = nn.MaxPool1d(kernel_size, stride=1, padding=kernel_size // 2)
    out = m(hidden_states.permute(0, 2, 1)).permute(0, 2, 1)
    return out


class PoNetEmbeddings(nn.Module):
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
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               'absolute')
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse('1.6.0'):
            self.register_buffer(
                'token_type_ids',
                torch.zeros(
                    self.position_ids.size(),
                    dtype=torch.long,
                    device=self.position_ids.device),
                persistent=False,
            )

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
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape,
                    dtype=torch.long,
                    device=self.position_ids.device)

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


class PoNetSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense_local = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_segment = nn.Linear(config.hidden_size, config.hidden_size)

        self.num_attention_heads = config.num_attention_heads
        self.clsgsepg = getattr(config, 'clsgsepg', True)
        self.attention_head_size = int(config.hidden_size
                                       / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dense_q = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense_k = nn.Linear(config.hidden_size, self.all_head_size)
        self.dense_o = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # bz, head, len, head_size

    def forward(
        self,
        hidden_states,
        segment_index,
        token_type_mask,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        context_layer_q = self.transpose_for_scores(
            self.dense_q(hidden_states))
        context_layer_k = self.transpose_for_scores(
            self.dense_k(hidden_states))
        context_layer_v = context_layer_k
        context_layer_o = self.transpose_for_scores(
            self.dense_o(hidden_states))

        if attention_mask is not None:
            _attention_mask = (attention_mask.squeeze(1).unsqueeze(-1) < -1)

        if attention_mask is not None:
            context_layer_q.masked_fill_(_attention_mask, 0.0)
            q = context_layer_q.sum(dim=-2) / torch.ones_like(
                _attention_mask).to(dtype=context_layer_q.dtype).masked_fill(
                    _attention_mask, 0.0).sum(dim=-2)
        else:
            q = context_layer_q.mean(dim=-2)
        att = torch.einsum('bdh,bdlh -> bdl', q, context_layer_k) / math.sqrt(
            context_layer_q.shape[-1])
        if attention_mask is not None:
            att = att + attention_mask.squeeze(1)
        att_prob = att.softmax(dim=-1)
        v = torch.einsum('bdlh,bdl->bdh', context_layer_v, att_prob)

        context_layer_segment = self.dense_segment(hidden_states)
        context_layer_local = self.dense_local(hidden_states)
        if attention_mask is not None:
            context_layer_local.masked_fill_(
                _attention_mask.squeeze(1), -10000)
            context_layer_segment.masked_fill_(
                _attention_mask.squeeze(1), -10000)

        if self.clsgsepg:
            # XXX: a trick to make sure the segment and local information will not leak
            context_layer_local = get_win_max(
                context_layer_local.masked_fill(
                    token_type_mask.unsqueeze(dim=-1), -10000))
            context_layer_segment = segment_max(
                context_layer_segment, index=segment_index)

            context_layer_segment.masked_fill_(
                token_type_mask.unsqueeze(dim=-1), 0.0)
            context_layer_local.masked_fill_(
                token_type_mask.unsqueeze(dim=-1), 0.0)
        else:
            context_layer_local = get_win_max(context_layer_local)
            context_layer_segment = segment_max(
                context_layer_segment, index=segment_index)

        context_layer_local = self.transpose_for_scores(context_layer_local)
        context_layer_segment = self.transpose_for_scores(
            context_layer_segment)

        context_layer = (v.unsqueeze(dim=-2) + context_layer_segment
                         ) * context_layer_o + context_layer_local
        context_layer = context_layer.permute(0, 2, 1, 3).reshape(
            *hidden_states.shape[:2], -1)

        if attention_mask is not None:
            context_layer.masked_fill_(_attention_mask.squeeze(1), 0.0)

        outputs = (context_layer,
                   att_prob) if output_attentions else (context_layer, )
        return outputs


class PoNetSelfOutput(nn.Module):

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


class PoNetIntermediate(nn.Module):

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


class PoNetOutput(nn.Module):

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


class PoNetAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = PoNetSelfAttention(config)
        self.output = PoNetSelfOutput(config)
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
        segment_index,
        token_type_mask,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            segment_index,
            token_type_mask,
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


class PoNetLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PoNetAttention(config)

        config.is_decoder = False  # XXX: Decoder is not yet impletemented.
        self.is_decoder = config.is_decoder

        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f'{self} should be used as a decoder model if cross attention is added'
            self.crossattention = PoNetAttention(config)
        self.intermediate = PoNetIntermediate(config)
        self.output = PoNetOutput(config)

    def forward(
        self,
        hidden_states,
        segment_index,
        token_type_mask,
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
            segment_index,
            token_type_mask,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
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
            ), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'  # noqa *

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


class PoNetEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [PoNetLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        segment_index,
        token_type_mask,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
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

            if getattr(self.config, 'gradient_checkpointing',
                       False) and self.training:

                if use_cache:
                    logger.warning(
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
                    segment_index,
                    token_type_mask,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    segment_index,
                    token_type_mask,
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
        return AttentionBackboneModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PoNetPooler(nn.Module):

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


class PoNetPreTrainedModel(TorchModel, PreTrainedModel):
    """
    A base class to handle weights initialization and a simple interface for loading pretrained models.
    """

    config_class = PoNetConfig
    base_model_prefix = 'ponet'
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

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.pop('model_dir', None)
        if model_dir is None:
            ponet_config = PoNetConfig(**kwargs)
            model = cls(ponet_config)
        else:
            model = super(
                Model,
                cls).from_pretrained(pretrained_model_name_or_path=model_dir)
        return model


@MODELS.register_module(Tasks.backbone, module_name=Models.ponet)
class PoNetModel(PoNetPreTrainedModel):
    """The bare PoNet Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~modelscope.models.nlp.ponet.PoNetConfig`):
            Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

        self.embeddings = PoNetEmbeddings(config)
        self.encoder = PoNetEncoder(config)

        self.pooler = PoNetPooler(config) if add_pooling_layer else None

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

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        segment_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using :class:`~modelscope.models.nlp.ponet.PoNetTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

            attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

            token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Segment token indices to indicate first and second portions of the inputs. Indices are selected in
                ``[0,1]``:

                - 0 corresponds to a `sentence A` token,
                - 1 corresponds to a `sentence B` token.

            position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                ``[0,config.max_position_embeddings - 1]``.

            head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`,
                `optional`):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`,
                `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids`
                indices into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            encoder_hidden_states
                (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
                the model is configured as a decoder.
            encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used
                in the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers`
                with each tuple having 4 tensors of shape :obj:
                `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
                (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
                instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
            use_cache (:obj:`bool`, `optional`):
                If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
                decoding (see :obj:`past_key_values`).

        Returns:
            Returns `modelscope.outputs.AttentionBackboneModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_ponet_fill-mask_chinese-base', task='backbone')
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_ponet_fill-mask_chinese-base')
            >>> print(model(**preprocessor('这是个测试')))
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

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
            attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size(
            )
            encoder_hidden_shape = (encoder_batch_size,
                                    encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
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

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        segment_index = get_segment_index(
            input_ids) if segment_ids is None else segment_ids
        token_type_mask = get_token_type_mask(input_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            segment_index,
            token_type_mask,
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
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return AttentionBackboneModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
