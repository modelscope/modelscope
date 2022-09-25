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

logger = logging.get_logger(__name__)


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
                 deep_init=False,
                 deepspeed=False,
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
                 pruning_method=None,
                 pruning_mask_init='constant',
                 pruning_mask_scale=0.0,
                 pruning_initial_threshold=1.0,
                 pruning_final_threshold=0.01,
                 pruning_initial_warmup=1,
                 pruning_final_warmup=20,
                 pruning_module='decoder',
                 pruning_decay_step=50,
                 pruning_decay_type='exp',
                 ft_module=None,
                 attn_separate=False,
                 LR_weight_rank=8,
                 LR_mask_rank=8,
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
        self.deep_init = deep_init
        self.deepspeed = deepspeed
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
        self.pruning_method = pruning_method
        self.pruning_mask_init = pruning_mask_init
        self.pruning_mask_scale = pruning_mask_scale
        self.pruning_module = pruning_module
        self.pruning_initial_threshold = pruning_initial_threshold
        self.pruning_final_threshold = pruning_final_threshold
        self.pruning_initial_warmup = pruning_initial_warmup
        self.pruning_final_warmup = pruning_final_warmup
        self.pruning_decay_step = pruning_decay_step
        self.pruning_decay_type = pruning_decay_type
        self.ft_module = ft_module
        self.attn_separate = attn_separate
        self.LR_weight_rank = LR_weight_rank
        self.LR_mask_rank = LR_mask_rank

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
    model_type = 'plugNLG'

    def __init__(self,
                 vocab_size=21504,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.00707,
                 deep_init=False,
                 deepspeed=False,
                 lr_decay_style='linear',
                 weight_decay=1e-2,
                 clip_grad=1.0,
                 warmup=0.01,
                 pre_ln=False,
                 fp16=False,
                 fp32_layernorm=False,
                 fp32_embedding=False,
                 fp32_tokentypes=False,
                 layernorm_epsilon=1e-12,
                 dec_hidden_layers=6,
                 pruning_method=None,
                 pruning_mask_init='constant',
                 pruning_mask_scale=0.0,
                 pruning_initial_threshold=1.0,
                 pruning_final_threshold=0.01,
                 pruning_initial_warmup=1,
                 pruning_final_warmup=20,
                 pruning_module='decoder',
                 pruning_decay_step=50,
                 pruning_decay_type='exp',
                 ft_module=None,
                 attn_separate=False,
                 LR_weight_rank=8,
                 LR_mask_rank=8,
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
        self.deep_init = deep_init
        self.deepspeed = deepspeed
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
        self.pruning_method = pruning_method
        self.pruning_mask_init = pruning_mask_init
        self.pruning_mask_scale = pruning_mask_scale
        self.pruning_module = pruning_module
        self.pruning_initial_threshold = pruning_initial_threshold
        self.pruning_final_threshold = pruning_final_threshold
        self.pruning_initial_warmup = pruning_initial_warmup
        self.pruning_final_warmup = pruning_final_warmup
        self.pruning_decay_step = pruning_decay_step
        self.pruning_decay_type = pruning_decay_type
        self.ft_module = ft_module
        self.attn_separate = attn_separate
        self.LR_weight_rank = LR_weight_rank
        self.LR_mask_rank = LR_mask_rank
