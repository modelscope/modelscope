# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger()


class GPT3Config(PretrainedConfig):
    r"""
    Configuration classes for GPT-3 model.

    Class attributes:

    - **model_type** (`str`) -- An identifier for the model type, serialized into the JSON file, can be used to recreate
      the correct object in [`~transformers.AutoConfig`].

    Args:
        vocab_size (`int`, *optional*, defaults to 25600):
            Vocabulary size of the GPT model. Defines the number of different
            tokens that can be represented by the `inputs_ids` passed when
            calling [`GPT3Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the decoder layers and the pooler layer.
        ffn_hidden_size (`int`, *optional*, defaults to None):
            Dimensionality of the ffn layer, None defaults to four times the hidden_size.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the
            Transformer decoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward)
            layer in the Transformer decoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the
            decoder and pooler. If string, `"gelu"`, `"relu"`, `"silu"` and
            `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the
            embeddings, decoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or
            1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling
            [`GPT3Model`].
        layernorm_epsilon (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        bias_gelu_fusion (`bool`, *optional*, defaults to True):
            Whether to use gelu activation function when mixing bias.
        fp32_residual_connection (`bool`, *optional*, defaults to False):
            Whether to use fp32 for residual connection
            between layers to improve accuracy.
        sequence_parallel (`bool`, *optional*, defaults to False):
            Whether to use sequence parallel during training.
        bf16 (`bool`, *optional*, defaults to `False`):
            Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training.
            Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda).
            This is an experimental API and it may change.
        fp16 (`bool`, *optional*, defaults to `False`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `True`):
            Whether to scale query and key layer parameters during training.
        init_method_std (`float`, *optional*, defaults to `0.02`):
            The standard deviation of the normal distribution for initialization process.
        eod_id (`int`, *optional*, defaults to `1`):
            The end of text label for tokenizer, also indicates the end of the generation.
        tokens_to_generate (`int`, *optional*, defaults to 100):
            Number of tokens to generate.
        top_k (`int`, *optional*, defaults to 0):
            Number of highest probability vocabulary tokens to keep for
            top-k-filtering that will be used by default in
            the `generate` method of the model.
        top_p (`float`, *optional*, defaults to 0.9):
            Value that will be used by default in the `generate` method of the model
            for `top_p`. If set to float < 1,
            only the most probable tokens with probabilities that add up to `top_p`
            or higher are kept for generation.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to module the next token probabilities that will be used
            by default in the `generate` method of the model. Must be strictly positive.
    """

    model_type = 'gpt3'

    def __init__(
            self,
            vocab_size=25600,
            hidden_size=768,
            ffn_hidden_size=None,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=2048,
            type_vocab_size=2,
            layernorm_epsilon=1e-12,
            bias_gelu_fusion=True,
            fp32_residual_connection=False,
            sequence_parallel=False,
            fp16=False,
            bf16=False,
            apply_query_key_layer_scaling=True,
            attention_softmax_in_fp32=False,
            kv_channels=None,
            masked_softmax_fusion=True,
            attention_dropout=0.1,
            bias_dropout_fusion=True,
            apply_residual_connection_post_layernorm=False,
            hidden_dropout=0.1,
            init_method_std=0.02,
            # generate
            eod_id=1,
            tokens_to_generate=100,
            top_k=0,
            top_p=0.9,
            temperature=1.0,
            **kwargs):
        super().__init__(layer_norm_eps=layernorm_epsilon, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = 4 * hidden_size \
            if ffn_hidden_size is None else ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layernorm_epsilon = layernorm_epsilon
        self.bias_gelu_fusion = bias_gelu_fusion
        self.fp32_residual_connection = fp32_residual_connection
        self.sequence_parallel = sequence_parallel
        self.fp16 = fp16
        self.bf16 = bf16
        assert not (fp16 and bf16)
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        if kv_channels is None:
            assert hidden_size % num_attention_heads == 0
            self.kv_channels = hidden_size // num_attention_heads
        self.masked_softmax_fusion = masked_softmax_fusion
        self.attention_dropout = attention_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.apply_residual_connection_post_layernorm = \
            apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.init_method_std = init_method_std
        self.eod_id = eod_id
        self.tokens_to_generate = tokens_to_generate
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        self.no_persist_layer_norm = \
            TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11)

    @property
    def params_dtype(self):
        if self.fp16:
            return torch.half
        elif self.bf16:
            return torch.bfloat16
        else:
            return torch.float
