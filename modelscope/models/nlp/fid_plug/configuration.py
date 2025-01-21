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
""" PLUG model configuration """
from transformers.configuration_utils import PretrainedConfig


class PlugConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layernorm_epsilon (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        dec_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        attn_separate (:obj:`bool`, `optional`, defaults to false):
            Whether or not to separate the q, k, v of attention.

    Examples::

        >>> import PlugModel, PlugConfig
        >>> configuration = PlugConfig()

        >>> # Initializing a model from the configuration
        >>> model = PlugModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'plug'

    def __init__(self,
                 encoder='roberta',
                 encoder_pth='roberta-base',
                 max_pos=512,
                 share_emb=False,
                 dec_layers=12,
                 dec_hidden_size=768,
                 dec_heads=8,
                 dec_ff_size=3072,
                 dec_dropout=0.2,
                 use_bert_emb=True,
                 label_smoothing=0.1,
                 block_trigram=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.encoder_pth = encoder_pth
        self.max_pos = max_pos
        self.share_emb = share_emb
        self.dec_layers = dec_layers
        self.dec_hidden_size = dec_hidden_size
        self.dec_heads = dec_heads
        self.dec_ff_size = dec_ff_size
        self.dec_dropout = dec_dropout
        self.use_bert_emb = use_bert_emb
        self.label_smoothing = label_smoothing
        # Translator
        self.block_trigram = block_trigram
