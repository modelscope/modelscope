# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import copy

import json
from transformers import PretrainedConfig

from modelscope.utils import logger as logging

logger = logging.get_logger()


class PlugNLUConfig(PretrainedConfig):
    model_type = 'plugNLU'

    def __init__(self,
                 vocab_size=21504,
                 original_vocab_size=21128,
                 hidden_size=8192,
                 num_hidden_layers=24,
                 num_attention_heads=128,
                 intermediate_size=32768,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=2048,
                 type_vocab_size=3,
                 initializer_range=0.00707,
                 lr_decay_style='linear',
                 weight_decay=1e-2,
                 clip_grad=1.0,
                 warmup=0.0333,
                 pre_ln=True,
                 fp16=True,
                 fp32_layernorm=True,
                 fp32_embedding=False,
                 fp32_tokentypes=False,
                 layernorm_epsilon=1e-5,
                 dec_hidden_layers=6,
                 attn_separate=False,
                 **kwargs):
        super().__init__(layer_norm_eps=layernorm_epsilon, **kwargs)

        self.vocab_size = vocab_size
        self.original_vocab_size = original_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.lr_decay_style = lr_decay_style
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.warmup = warmup
        self.pre_ln = pre_ln
        self.fp16 = fp16
        self.fp32_layernorm = fp32_layernorm
        self.fp32_embedding = fp32_embedding
        self.layernorm_epsilon = layernorm_epsilon
        self.fp32_tokentypes = fp32_tokentypes
        self.dec_hidden_layers = dec_hidden_layers
        self.attn_separate = attn_separate

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = PlugNLUConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r', encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def merge_args(self, args):
        """merge values a `BertConfig` from a json file of parameters."""
        local_keys = self.__dict__.keys()
        for key, value in args.__dict__.items():
            if key in local_keys:
                continue
            self.__dict__[key] = value
        return self

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class PlugNLGConfig(PlugNLUConfig):
    """
    This is the configuration class to store the configuration of a [`PlugModel`]. It is used to instantiate a
    PLUG understanding model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the PLUG
    [PLUG](https://modelscope.cn/models/damo/nlp_plug_text-generation_27B/summary) architecture.

    Configuration objects inherit from [`PlugNLUConfig`] and can be used to control the model outputs. Read the
    documentation from [`PlugNLUConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 21504):
            Padded vocabulary size of the PLUG model for vocab tensor parallel. Defines the number of different tokens
            that can be represented by the `inputs_ids` passed when calling [`PlugModel`].
        original_vocab_size (`int`, *optional*, defaults to 21128):
            True vocabulary size of the PLUG model. Defines the number of different tokens that can be represented.
        hidden_size (`int`, *optional*, defaults to 8192):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        dec_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 32768):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the Transformer Attention.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 3):
            The vocabulary size of the `token_type_ids` passed when calling [`PlugModel`].
        initializer_range (`float`, *optional*, defaults to 0.00707):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        lr_decay_style (`str`, *optional*, defaults to 'linear'):
            The decay style of learning rate during fine-tunining. If string, `"linear"`, `"cosine"`, `"exponential"`,
            `"constant"`, `"None"` are supported.
        weight_decay (`float`, *optional*, defaults to 1e-2):
            Decoupled weight decay to apply.
        clip_grad (`float`, *optional*, defaults to 1.0):
            Maximum gradient norm for gradient clipping.
        warmup (`float`, *optional*, defaults to 0.01):
            Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.
        pre_ln (`boolean`, *optional*, defaults to `True`):
            Whether or not to apply LayerNorm to the input instead of the output in the blocks.
        fp16 (`boolean`, *optional*, defaults to `True`):
            Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        fp32_layernorm (`boolean`, *optional*, defaults to `True`):
            Whether to use fp32 32-bit precision LayerNorm training while the argument `fp16` set to `True`.
        fp32_embedding (`boolean`, *optional*, defaults to `False`):
            Whether to use fp32 32-bit precision Embedding training while the argument `fp16` set to `True`.
        fp32_tokentypes (`boolean`, *optional*, defaults to `False`):
            Whether to use fp32 32-bit precision token types training while the argument `fp16` set to `True`.
        layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        attn_separate (`boolean`, *optional*, defaults to `False`):
            Whether or not to separate query-key-value to query, key, value in the Attention.

    Example:

    >>> # The PLUG model has 27B parameters and usually need to run on multiple GPUs. The example given
    >>> # here only initializes a slice of the model on a single GPU.
    >>> # Check out the [`~DistributedPipeline.__init__`] method to initialize entire PLUG model.
    >>> from modelscope.models.nlp.plug import PlugNLGConfig, PlugModel

    >>> # Initializing a Plug configuration
    >>> configuration = PlugNLGConfig()

    >>> # Initializing a model from the configuration
    >>> model = PlugModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    """

    model_type = 'plugNLG'

    def __init__(self,
                 vocab_size=21504,
                 original_vocab_size=21128,
                 hidden_size=8192,
                 num_hidden_layers=24,
                 dec_hidden_layers=6,
                 num_attention_heads=128,
                 intermediate_size=32768,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=2048,
                 type_vocab_size=3,
                 initializer_range=0.00707,
                 lr_decay_style='linear',
                 weight_decay=1e-2,
                 clip_grad=1.0,
                 warmup=0.01,
                 pre_ln=True,
                 fp16=True,
                 fp32_layernorm=True,
                 fp32_embedding=False,
                 fp32_tokentypes=False,
                 layernorm_epsilon=1e-12,
                 attn_separate=False,
                 **kwargs):
        super().__init__(layer_norm_eps=layernorm_epsilon, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.lr_decay_style = lr_decay_style
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.warmup = warmup
        self.pre_ln = pre_ln
        self.fp16 = fp16
        self.fp32_layernorm = fp32_layernorm
        self.fp32_embedding = fp32_embedding
        self.layernorm_epsilon = layernorm_epsilon
        self.fp32_tokentypes = fp32_tokentypes
        self.dec_hidden_layers = dec_hidden_layers
        self.attn_separate = attn_separate
