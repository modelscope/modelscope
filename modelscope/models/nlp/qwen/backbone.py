# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import math
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from transformers import (GenerationConfig, PreTrainedTokenizer,
                          StoppingCriteriaList)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_utils import set_seed
from transformers.utils import (ModelOutput, add_code_sample_docstrings,
                                add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging)
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from modelscope import Model, TorchModel
from modelscope.metainfo import Models
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from ... import MODELS
from .configuration import QWenConfig
from .qwen_generation_utils import (HistoryType, StopWordsLogitsProcessor,
                                    decode_tokens, get_stop_words_ids,
                                    make_context)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
    from einops import rearrange

    use_flash_rotary = True
except ImportError:
    use_flash_rotary = False
    print(
        'Warning: import flash_attn rotary fail, please install FlashAttention rotary to get better performance '
        'https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary')

try:
    from flash_attn.ops.rms_norm import rms_norm
except ImportError:
    rms_norm = None
    print(
        'Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get better performance '
        'https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm'
    )

logger = get_logger()

_CHECKPOINT_FOR_DOC = 'qwen-7b'
_CONFIG_FOR_DOC = 'QWenConfig'

QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ['qwen-7b']

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    flash_attn_unpadded_func = None
    print('Warning: import flash_attn fail, please install FlashAttention '
          'https://github.com/Dao-AILab/flash-attention')


class FlashSelfAttention(torch.nn.Module):

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
    ):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            'Please install FlashAttention first, '
            'e.g., with pip install flash-attn')
        assert (rearrange is not None
                ), 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        assert all(
            (i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )

        if self.training:
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        else:
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * seqlen_k,
                step=seqlen_k,
                dtype=torch.int32,
                device=q.device,
            )
            self.dropout_p = 0
        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class QWenAttention(nn.Module):

    def __init__(self, config, layer_number=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            'bias',
            torch.tril(
                torch.ones((max_positions, max_positions),
                           dtype=torch.bool)).view(1, 1, max_positions,
                                                   max_positions),
            persistent=False,
        )
        self.register_buffer(
            'masked_bias', torch.tensor(-1e4), persistent=False)
        self.layer_number = max(1, layer_number)
        self.params_dtype = config.params_dtype
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.use_flash_attn = config.use_flash_attn
        self.scale_attn_weights = True

        self.layer_idx = None

        self.projection_size = config.kv_channels * config.num_attention_heads

        assert self.projection_size % config.num_attention_heads == 0
        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads)

        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)

        self.c_proj = nn.Linear(
            config.hidden_size, self.projection_size, bias=not config.no_bias)

        self.is_fp32 = not (config.bf16 or config.fp16)
        if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attn_pdrop)

        self.bf16 = config.bf16

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(self.hidden_size_per_attention_head
                                    * config.rotary_pct)
        dim = (
            self.rotary_ndims if self.rotary_ndims is not None else
            self.hidden_size_per_attention_head)
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        self.logn_tensor = torch.Tensor(logn_list)[None, :, None, None]
        self._ntk_cached = 1.0

        self.attn_dropout = nn.Dropout(config.attn_pdrop)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [],
                value.size(-1)**0.5,
                dtype=attn_weights.dtype,
                device=attn_weights.device,
            )

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length
                                - query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(
            attn_weights.device)
        attn_weights = torch.where(causal_mask,
                                   attn_weights.to(attn_weights.dtype),
                                   mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self,
                                   query,
                                   key,
                                   value,
                                   attention_mask=None,
                                   head_mask=None):
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        attn_weights = torch.empty(
            bsz * num_heads,
            q_seq_len,
            k_seq_len,
            dtype=torch.float32,
            device=query.device,
        )

        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1))**0.5

        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len,
                                 dk), key.transpose(-1, -2).reshape(
                                     -1, dk, k_seq_len)
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len,
                                                k_seq_len)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length
                                - query_length:key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(
            mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                'Error with upcasting, attn_weights does not have dtype torch.float32'
            )
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size, )
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):

        mixed_x_layer = self.c_attn(hidden_states)
        query, key, value = mixed_x_layer.split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        kv_seq_len = hidden_states.size()[1]
        if layer_past:
            kv_seq_len += layer_past[0].shape[1]
        if (self.use_dynamic_ntk and kv_seq_len == hidden_states.size()[1]
                and not self.training):
            context_value = math.log(kv_seq_len / self.seq_length, 2) + 1
            ntk_alpha = 2**math.ceil(context_value) - 1
            ntk_alpha = max(ntk_alpha, 1)
            self._ntk_cached = ntk_alpha
        else:
            ntk_alpha = self._ntk_cached
        rotary_pos_emb = self.rotary_emb(
            kv_seq_len, ntk_alpha=ntk_alpha).to(hidden_states.device)

        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb, ) * 2

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            cur_len = query.shape[1]
            q_pos_emb = q_pos_emb[:, -cur_len:, :, :]
            k_pos_emb = k_pos_emb[:, -cur_len:, :, :]
            query = apply_rotary_pos_emb(query, q_pos_emb)
            key = apply_rotary_pos_emb(key, k_pos_emb)

        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)

        if use_cache:
            present = (key, value)
        else:
            present = None

        if self.use_logn_attn and not self.training:
            if self.logn_tensor.device != query.device:
                self.logn_tensor = self.logn_tensor.to(
                    query.device).type_as(query)
            seq_start = key.size(1) - query.size(1)
            seq_end = key.size(1)
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
            query = query * logn_tensor.expand_as(query)

        if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32 and query.is_cuda:
            q, k, v = query, key, value
            context_layer = self.core_attention_flash(q, k, v)

            context_layer = rearrange(context_layer,
                                      'b s h d -> b s (h d)').contiguous()
        else:
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            attn_output, attn_weight = self._attn(query, key, value,
                                                  attention_mask, head_mask)
            context_layer = self._merge_heads(attn_output, self.num_heads,
                                              self.head_dim)

        attn_output = self.c_proj(context_layer)
        outputs = (attn_output, present)
        if output_attentions:
            if self.use_flash_attn and flash_attn_unpadded_func is not None and not self.is_fp32:
                raise ValueError(
                    'Cannot output attentions while using flash-attn')
            else:
                outputs += (attn_weight, )

        return outputs


class QWenMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size // 2,
            bias=not config.no_bias)
        self.w2 = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size // 2,
            bias=not config.no_bias)
        ff_dim_in = config.ffn_hidden_size // 2
        self.c_proj = nn.Linear(
            ff_dim_in, config.hidden_size, bias=not config.no_bias)

    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        return output


class QWenBlock(nn.Module):

    def __init__(self, config, layer_idx=None, num_expert=1):
        super().__init__()
        self.num_expert = num_expert
        self.layer_number = layer_idx
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)
        hidden_size = config.hidden_size
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm)
        self.bf16 = config.bf16

        self.ln_1 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.attn = QWenAttention(config, layer_number=layer_idx)
        self.ln_2 = RMSNorm(
            hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.mlp = QWenMLP(config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            layernorm_output,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = attn_output + residual

        layernorm_output = self.ln_2(layernorm_input)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        mlp_output = self.mlp(layernorm_output)
        hidden_states = residual + mlp_output

        if use_cache:
            outputs = (hidden_states, ) + outputs
        else:
            outputs = (hidden_states, ) + outputs[1:]

        return outputs


class QWenPreTrainedModel(TorchModel, PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = 'transformer'
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ['QWenBlock']

    def __init__(self, config, **kwargs):
        super().__init__(config.name_or_path, **kwargs)
        super(Model, self).__init__(config)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == 'c_proj.weight':
                p.data.normal_(
                    mean=0.0,
                    std=(self.config.initializer_range
                         / math.sqrt(2 * self.config.n_layer)),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value

    @classmethod
    def _instantiate(cls, **kwargs):
        model_dir = kwargs.pop('model_dir', None)
        if model_dir is None:
            config = QWenConfig(**kwargs)
            model = cls(config)
        else:
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **kwargs)
        model.model_dir = model_dir
        return model


@MODELS.register_module(Tasks.backbone, module_name=Models.qwen_7b)
class QWenModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['attn.masked_bias']

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = config.padded_vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size

        max_sequence_length = config.max_position_embeddings
        self.position_embedding_type = config.pos_emb
        self.gradient_checkpointing = False

        if self.position_embedding_type == 'learned':
            self.wpe = nn.Embedding(max_sequence_length, self.embed_dim)
            self.init_method(self.position_embeddings.weight)
            self._position_embeddings_key = 'position_embeddings'
            self.init_method(self.position_embeddings.weight)
        else:
            self.wpe = None
            self._position_embeddings_key = ''

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([
            QWenBlock(
                config,
                layer_idx=i,
            ) for i in range(config.num_hidden_layers)
        ])
        self.ln_f = RMSNorm(
            self.embed_dim,
            eps=config.layer_norm_epsilon,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else
            self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else
            self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(
                self.dtype).min

        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        hidden_states = inputs_embeds
        if self.wpe is not None:
            position_embeds = self.wpe(position_ids)
            hidden_states = hidden_states + position_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1), )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (
                    outputs[2 if output_attentions else 1], )

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1], )

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        if not return_dict:
            return tuple(v
                         for v in [hidden_states, presents, all_hidden_states]
                         if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 / (base**(torch.arange(0, dim, 2).float() / dim))
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError('einops is required for Rotary Embedding')

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self,
                                    max_seq_len,
                                    offset=0,
                                    ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha**(self.dim / (self.dim - 2))
            '''
            self.inv_freq = 1.0 / (
                base**(torch.arange(
                    0, self.dim, 2, device=self.inv_freq.device).float()
                       / self.dim))
            '''
            self.inv_freq = torch.arange(
                0, self.dim, 2, device=self.inv_freq.device).float() / self.dim
            self.inv_freq = 1.0 / (base**self.inv_freq)
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            self._rotary_pos_emb_cache = rearrange(emb, 'n d -> 1 n 1 d')

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[:, offset:offset + max_seq_len]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, use_flash_rotary=False):
    if use_flash_rotary:
        t_ = t.float()
        freqs = freqs.squeeze(0).squeeze(1)
        cos = freqs[:, :freqs.shape[-1] // 2].cos()
        sin = freqs[:, :freqs.shape[-1] // 2].sin()
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs.shape[-1]
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
