# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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

import copy
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import xavier_uniform_
from transformers import (BertConfig, BertModel, BertTokenizer, RobertaConfig,
                          RobertaModel, RobertaTokenizer, logging)
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel

from .configuration import PlugConfig

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


class MultiHeadedAttention(nn.Module):  # SelfAttention
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self,
                 head_count,
                 model_dim,
                 dropout=0.1,
                 use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super().__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self,
                key,
                value,
                query,
                mask=None,
                layer_cache=None,
                type=None,
                predefined_graph_1=None,
                return_attn=False):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == 'self':
                query, key, value = self.linear_query(query), self.linear_keys(
                    query), self.linear_values(query)

                key = shape(key)
                value = shape(value)

                device = key.device
                if layer_cache['self_keys'] is not None:
                    key = torch.cat((layer_cache['self_keys'].to(device), key),
                                    dim=2)
                if layer_cache['self_values'] is not None:
                    value = torch.cat(
                        (layer_cache['self_values'].to(device), value), dim=2)
                layer_cache['self_keys'] = key
                layer_cache['self_values'] = value
            elif type == 'context':
                query = self.linear_query(query)
                if layer_cache['memory_keys'] is None:
                    key, value = self.linear_keys(key), self.linear_values(
                        value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache['memory_keys'], layer_cache[
                        'memory_values']
                layer_cache['memory_keys'] = key
                layer_cache['memory_values'] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, float('-inf'))

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (predefined_graph_1 is not None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (
                torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            if return_attn:
                return output, attn
            else:
                return output
        else:
            context = torch.matmul(drop_attn, value)
            if return_attn:
                return context, attn
            else:
                return context


class PositionwiseFeedForward(nn.Module):  # Output
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.w_1 = nn.Linear(d_model, d_ff)
        self.actv = ACT2FN['gelu_new']
        self.dropout_1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerDecoderLayer(nn.Module):  # Layer
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """
    MAX_SIZE = 5000

    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=dropout)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(self.MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self,
                inputs,
                memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                previous_input=None,
                layer_cache=None,
                step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(
            tgt_pad_mask.type(torch.uint8)
            + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)].type(
                torch.uint8), 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None

        query = self.self_attn(
            all_input,
            all_input,
            input_norm,
            mask=dec_mask,
            layer_cache=layer_cache,
            type='self')

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            type='context',
            return_attn=True)
        output = self.feed_forward(self.drop(mid) + query)

        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float)
                              * -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerDecoderState:

    def __init__(self, src: Tensor, cache_num_layers: int = -1):
        self.src: Tensor = src
        self.previous_input: Tensor = None
        self.previous_layer_inputs: Tensor = None
        self.cache: Optional[Dict[str, Any]] = None
        if cache_num_layers != -1:
            self._init_cache(cache_num_layers)

    def update_state(self, new_input, previous_layer_inputs):
        self.previous_input = new_input
        self.previous_layer_inputs = previous_layer_inputs
        self.cache = None

    def _init_cache(self, num_layers):
        self.cache = {}
        for layer in range(num_layers):
            layer_cache = {'memory_keys': None, 'memory_values': None}
            layer_cache['self_keys'] = None
            layer_cache['self_values'] = None
            self.cache['layer_{}'.format(layer)] = layer_cache

    def map_batch_fn(self, fn):

        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)


class TransformerDecoder(nn.Module):  # Decoder
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       d_model (int): size of the model
       heads (int): number of heads
       d_ff (int): size of the inner FF layer
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
       attn_type (str): if using a seperate copy attention
    """
    decoder_type = 'transformer'

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super().__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,
                                          self.embeddings.embedding_dim)

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.state = None

    def forward(self,
                state: TransformerDecoderState,
                tgt: Tensor,
                memory_bank: Tensor,
                step: int = None,
                memory_masks: Tensor = None):
        src_words = state.src
        tgt_words = tgt
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)
        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim
        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        if (memory_masks is not None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)
        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)

        if state.cache is None:
            saved_inputs = []
        attns = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.cache is None:
                if state.previous_input is not None:
                    prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    previous_input=prev_layer_input,
                    layer_cache=state.cache['layer_{}'.format(i)]
                    if state.cache is not None else None,
                    step=step)
            if state.cache is None:
                saved_inputs.append(all_input)
            attns.append(attn)

        if state.cache is None:
            saved_inputs = torch.stack(saved_inputs)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        if state.cache is None:
            state.update_state(tgt, saved_inputs)

        return output, attns, state


class PlugPointerGenerator(nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.gen_func = nn.LogSoftmax(-1)

    def forward(self, x):
        x = self.dense(x)
        x = self.gen_func(x)
        return x


class PlugPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PlugConfig
    base_model_prefix = 'plug'

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path: Optional[Union[str,
                                                               os.PathLike]]):
        config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        config = PlugConfig.from_json_file(config_file) if os.path.isfile(
            config_file) else PlugConfig()
        config.encoder_pth = os.path.join(pretrained_model_name_or_path,
                                          config.encoder_pth)
        checkpoint_file = os.path.join(pretrained_model_name_or_path,
                                       WEIGHTS_NAME)
        checkpoint = torch.load(checkpoint_file) if os.path.isfile(
            checkpoint_file) else None
        return cls(config, checkpoint)


class PlugModel(PlugPreTrainedModel):  # Model

    def __init__(self, config, checkpoint=None):
        super().__init__(config)
        self.config = config
        if config.encoder == 'bert' or config.encoder == 'zh_bert':
            self.bert = BertModel(
                BertConfig.from_pretrained(config.encoder_pth))
        elif config.encoder == 'roberta':
            self.bert = RobertaModel(
                RobertaConfig.from_pretrained(config.encoder_pth))

        if (config.max_pos > 512):
            my_pos_embeddings = nn.Embedding(
                config.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:
                                          512] = self.bert.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[
                512:] = self.bert.embeddings.position_embeddings.weight.data[
                    -1][None, :].repeat(config.max_pos - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings
        self.vocab_size = self.bert.config.vocab_size
        tgt_embeddings = nn.Embedding(
            self.vocab_size,
            self.bert.config.hidden_size,
            padding_idx=1 if config.encoder == 'roberta' else 0)

        if config.share_emb:
            tgt_embeddings.weight = copy.deepcopy(
                self.bert.model.embeddings.word_embeddings.weight)
        self.decoder = TransformerDecoder(
            config.dec_layers,
            config.dec_hidden_size,
            heads=config.dec_heads,
            d_ff=config.dec_ff_size,
            dropout=config.dec_dropout,
            embeddings=tgt_embeddings)
        self.generator = PlugPointerGenerator(config.dec_hidden_size,
                                              self.vocab_size)
        self.generator.dense.weight = self.decoder.embeddings.weight

        if checkpoint is not None:
            for key in list(checkpoint['model'].keys()):
                if key.startswith('module.'):
                    checkpoint['model'][key.replace(
                        'module.', '')] = checkpoint['model'][key]
                    checkpoint['model'].pop(key)
                if key.startswith('plug.'):
                    checkpoint['model'][key.replace(
                        'plug.', '')] = checkpoint['model'][key]
                    checkpoint['model'].pop(key)
            msg = self.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
        else:
            for module in self.decoder.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                elif isinstance(module, nn.LayerNorm):
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            for p in self.generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
                else:
                    p.data.zero_()
            if config.use_bert_emb:
                if config.encoder == 'roberta':
                    tgt_embeddings = nn.Embedding(
                        self.vocab_size,
                        self.bert.config.hidden_size,
                        padding_idx=1)
                else:
                    tgt_embeddings = nn.Embedding(
                        self.vocab_size,
                        self.bert.config.hidden_size,
                        padding_idx=0)
                tgt_embeddings.weight = copy.deepcopy(
                    self.bert.embeddings.word_embeddings.weight)
                self.decoder.embeddings = tgt_embeddings
            self.generator.dense.weight = self.decoder.embeddings.weight

    def forward(self, src, tgt, mask_src, token_type_ids):
        top_vec, _ = self.bert(
            src, mask_src, token_type_ids=token_type_ids, return_dict=False)
        state = TransformerDecoderState(src)
        decoder_outputs, attns, _ = self.decoder(state, tgt[:, :-1], top_vec)
        return decoder_outputs, attns[-1], top_vec


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size, ), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size, label_smoothing=0.0):
        super().__init__()
        self.generator = generator
        self.padding_idx = symbols['PAD']
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx)
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum')

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, tgt, output):
        target = tgt[:, 1:]
        batch_size, decoder_length = target.size(0), target.size(1)
        normalization = target.ne(self.padding_idx).sum()
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth = target.contiguous().view(-1)
        loss = self.criterion(scores, gtruth)
        loss = loss.div(float(normalization))
        return loss, scores.view(batch_size, decoder_length, -1)


class PlugForConditionalGeneration(PlugPreTrainedModel):

    @dataclass
    class Batch:
        batch_size: int
        src: torch.Tensor
        tgt: torch.Tensor
        mask_src: torch.Tensor
        token_type_ids: torch.Tensor
        query_id: List[None] = None
        src_str: List[List[str]] = None
        tgt_str: List[str] = None

    def __init__(self, config, checkpoint=None, dataset: str = 'default'):
        super().__init__(config)
        self.logger = logging.get_logger(__name__)
        self.config = config
        if config.encoder == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(
                config.encoder_pth, do_lower_case=False)
            symbols = {
                'BOS': tokenizer.cls_token_id,
                'EOS': tokenizer.sep_token_id,
                'PAD': tokenizer.pad_token_id,
                'EOQ': tokenizer.unk_token_id
            }
        elif config.encoder == 'bert' or config.encoder == 'zh_bert':
            tokenizer = BertTokenizer.from_pretrained(
                config.encoder_pth, do_lower_case=True)
            symbols = {
                'BOS': tokenizer.vocab['[CLS]'],
                'EOS': tokenizer.vocab['[SEP]'],
                'PAD': tokenizer.vocab['[PAD]'],
                'EOQ': tokenizer.vocab['[unused2]']
            }
        self.tokenizer = tokenizer
        self.symbols = symbols
        self.plug = PlugModel(config, checkpoint)
        self.loss = NMTLossCompute(self.plug.generator, symbols,
                                   self.plug.vocab_size,
                                   config.label_smoothing)
        # for generation
        self.config.dataset = dataset
        self.start_token = self.symbols['BOS']
        self.end_token = self.symbols['EOS']

    def forward(self, src, tgt, mask_src=None, token_type_ids=None):
        if mask_src is None:
            mask_src = src.ne(self.symbols['PAD']).long()
        output = self.plug(src, tgt, mask_src, token_type_ids)[0]
        loss = self.loss(tgt, output)
        return loss

    def translate_batch(self,
                        batch: 'Batch',
                        fast: bool = False,
                        *args,
                        **kwargs):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        self.plug.eval()
        with torch.no_grad():
            return self._fast_translate_batch(batch, *args, **kwargs)

    def _tile(self, x, count, dim=0):
        perm = list(range(len(x.size())))
        if dim != 0:
            perm[0], perm[dim] = perm[dim], perm[0]
            x = x.permute(perm).contiguous()
        out_size = list(x.size())
        out_size[0] *= count
        batch = x.size(0)
        x = x.view(batch, -1) \
            .transpose(0, 1) \
            .repeat(count, 1) \
            .transpose(0, 1) \
            .contiguous() \
            .view(*out_size)
        if dim != 0:
            x = x.permute(perm).contiguous()
        return x

    def _top_k_top_p_filtering(self,
                               logits,
                               top_k=10,
                               top_p=1.0,
                               filter_value=-float('Inf'),
                               min_tokens_to_keep=1):
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep),
                        logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1,
                                                                      None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def _fast_translate_batch(self,
                              batch: 'Batch',
                              max_length: int = 80,
                              min_length: int = 10,
                              bad_words_ids=None,
                              early_stopping=True,
                              num_beams=3,
                              length_penalty=1.2,
                              repetition_penalty=1.2,
                              no_repeat_ngram_size=4,
                              *args,
                              **kwargs):
        # TODO: faster code path for beam_size == 1.
        # TODO: support these blacklisted features.

        num_beams = num_beams
        batch_size = batch.batch_size
        src = batch.src
        mask_src = batch.mask_src
        token_type_ids = batch.token_type_ids

        src_features, _ = self.plug.bert(
            src, mask_src, token_type_ids=token_type_ids, return_dict=False)
        state = TransformerDecoderState(src, self.plug.decoder.num_layers)
        device = src_features.device

        # Tile states and memory beam_size times.
        state.map_batch_fn(
            lambda state, dim: self._tile(state, num_beams, dim=dim))
        src_features = self._tile(src_features, num_beams, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * num_beams,
            step=num_beams,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full([batch_size * num_beams, 1],
                               self.start_token,
                               dtype=torch.long,
                               device=device)

        # cal bad_words_ids pre dict
        bad_words_prefix_dict = {}
        bad_words_prefix_len = set([])
        if bad_words_ids is not None:
            for bw_id in bad_words_ids:
                key = tuple(bw_id[:-1])
                value = bw_id[-1]
                bad_words_prefix_dict[key] = bad_words_prefix_dict.get(
                    key, []) + [value]
                bad_words_prefix_len.add(len(key))

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor(
                [0.0] + [float('-inf')] * (num_beams - 1),
                device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results['predictions'] = [[] for _ in range(batch_size)]  # noqa: F812
        results['scores'] = [[] for _ in range(batch_size)]  # noqa: F812
        results['gold_score'] = [0] * batch_size
        results['batch'] = batch

        for step in range(max_length):
            self.logger.info(f'step: {step + 1} / {max_length}')
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)
            dec_out, attns, state = self.plug.decoder(
                state, decoder_input, src_features, step=step)

            # Generator forward.
            log_probs = self.plug.generator.forward(
                dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # filter bad word
            if len(bad_words_prefix_dict) > 0:
                # cal bad word banned token: batch_size * num_beams
                num_hypos = alive_seq.size(0)
                bad_word_banned_token = []
                for i in range(num_hypos):
                    curr_banned_token = []
                    for pre_len in bad_words_prefix_len:
                        pre_key = tuple(alive_seq[i, step + 1 - pre_len:step
                                                  + 1].cpu().numpy().tolist())
                        curr_banned_token += bad_words_prefix_dict.get(
                            pre_key, [])
                    bad_word_banned_token.append(set(curr_banned_token))
                # set banned word prob=-1e20
                assert log_probs.size(0) == num_hypos
                for i in range(num_hypos):
                    for banned_token in bad_word_banned_token[i]:
                        log_probs[i, banned_token] = -1e20

            # do repetition_penalty
            if repetition_penalty > 1.0:
                """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
                # calculate prev_output_tokens for repetition_penalty: batch_size * num_beams
                prev_output_tokens = self.calc_banned_tokens(
                    alive_seq, alive_seq.size(0), no_repeat_ngram_size,
                    step + 1)
                # batch_size * num_beams
                for i in range(log_probs.size(0)):
                    for previous_token in set(prev_output_tokens[i]):
                        if log_probs[i, previous_token] < 0:
                            log_probs[i, previous_token] *= repetition_penalty
                        else:
                            log_probs[i, previous_token] /= repetition_penalty

            # Multiply probs by the beam probability.

            curr_length_penalty = (step + 1)**length_penalty
            # '''
            if self.config.sample_topk:
                temperature = self.config.temperature
                _scores = log_probs / temperature
                _scores = self._top_k_top_p_filtering(
                    _scores,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    min_tokens_to_keep=1
                )  # (batch_size * num_beams, vocab_size)
                # Sample 2 next words for each beam (so we have some spare tokens
                # and match output of greedy beam search)
                topk_ids = torch.multinomial(
                    F.softmax(_scores, dim=-1),
                    num_samples=1)  # (batch_size * num_beams, 2)
                # Compute next scores
                _scores = F.log_softmax(
                    _scores, dim=1)  # (batch_size * num_beams, vocab_size)

                _scores += topk_log_probs.view(-1).unsqueeze(1)
                _scores = _scores / curr_length_penalty
                topk_scores = torch.gather(
                    _scores, -1, topk_ids)  # (batch_size * num_beams, 2)
                # log_probs +=   # (batch_size * num_beams, 2)
                # Match shape of greedy beam search
                topk_ids = topk_ids.view(
                    -1, num_beams)  # (batch_size, 2 * num_beams)
                topk_scores = topk_scores.view(
                    -1, num_beams)  # (batch_size, 2 * num_beams)
            # '''
            else:
                log_probs += topk_log_probs.view(-1).unsqueeze(1)
                curr_scores = log_probs / curr_length_penalty

                curr_scores = curr_scores.reshape(-1, num_beams * vocab_size)
                topk_scores, topk_ids = curr_scores.topk(num_beams, dim=-1)
            if (self.config.block_trigram):
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        if self.config.encoder == 'roberta':
                            # words = [self.vocab.convert_ids_to_tokens[w] for w in words]
                            words = self.tokenizer.decode(
                                words).strip().split()
                        else:
                            words = [
                                self.tokenizer.ids_to_tokens[w] for w in words
                            ]
                            words = ' '.join(words).replace(' ##', '').split()
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1])
                                    for i in range(1,
                                                   len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20
            # Recover log probs.
            topk_log_probs = topk_scores * curr_length_penalty

            # Resolve beam origin and true word ids.
            # topk_beam_index = topk_ids.div(vocab_size)
            topk_beam_index = topk_ids // vocab_size
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                topk_beam_index
                + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat([
                alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)
            ], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(self.end_token)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, num_beams, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(self.end_token)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append(
                            (topk_scores[i, j], predictions[i, j, 1:]))
                        if early_stopping and len(hypotheses) == num_beams:
                            end_condition[i] = True
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        if self.config.dataset == 'qg_ranking_test' or (
                                self.config.dataset == 'paraphrase'
                                and not self.config.sample_topk):
                            for each in best_hyp[:num_beams]:
                                score, pred = each
                                results['scores'][b].append(score)
                                results['predictions'][b].append(pred)
                        else:
                            score, pred = best_hyp[0]
                            results['scores'][b].append(score)
                            results['predictions'][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            state.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

    def calc_banned_tokens(self, prev_input_ids, num_hypos,
                           no_repeat_ngram_size, cur_len):
        # Copied from fairseq for no_repeat_ngram in beam_search"""
        if cur_len + 1 < no_repeat_ngram_size:
            # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return [[] for _ in range(num_hypos)]
        generated_ngrams = [{} for _ in range(num_hypos)]
        for idx in range(num_hypos):
            gen_tokens = prev_input_ids[idx].cpu().numpy().tolist()
            generated_ngram = generated_ngrams[idx]
            for ngram in zip(
                    *[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                    prev_ngram_tuple, []) + [ngram[-1]]

        def _get_generated_ngrams(hypo_idx):
            # Before decoding the next token, prevent decoding of ngrams that have already appeared
            start_idx = cur_len + 1 - no_repeat_ngram_size
            ngram_idx = tuple(
                prev_input_ids[hypo_idx,
                               start_idx:cur_len].cpu().numpy().tolist())
            return generated_ngrams[hypo_idx].get(ngram_idx, [])

        banned_tokens = [
            _get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)
        ]
        return banned_tokens

    def translate(self,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor = None,
                  token_type_ids=None,
                  *args,
                  **kwargs) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.symbols['PAD']).long()
        batch = self.Batch(
            batch_size=input_ids.size()[0],
            src=input_ids,
            tgt=None,
            token_type_ids=token_type_ids,
            mask_src=attention_mask)
        translation_batch = self.translate_batch(batch, *args, **kwargs)

        preds = translation_batch['predictions']
        return {'predictions': preds}
