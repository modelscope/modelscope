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
""" PEER model configuration """

# modified the path according to the structure in my directory csssl_4_15/cssl/ and its env
from transformers.configuration_utils import PretrainedConfig

from modelscope.utils import logger as logging

logger = logging.get_logger()


class PeerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.PeerModel` or a
    :class:`~transformers.TFPeerModel`. It is used to instantiate a PEER model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the PEER `google/peer-small-discriminator
    <https://huggingface.co/google/peer-small-discriminator>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `onal`, defaults to 30522)
            Vocabulary size of the PEER model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.PeerModel` or
            :class:`~transformers.TFPeerModel`.
        embedding_size (:obj:`int`, `onal`, defaults to 128)
            Dimensionality of the encoder layers and the pooler layer.
        hidden_size (:obj:`int`, `onal`, defaults to 256)
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `onal`, defaults to 12)
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `onal`, defaults to 4)
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `onal`, defaults to 1024)
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `onal`, defaults to :obj:`"gelu"`)
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `onal`, defaults to 0.1)
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `onal`, defaults to 0.1)
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `onal`, defaults to 512)
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `onal`, defaults to 2)
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.PeerModel` or
            :class:`~transformers.TFPeerModel`.
        initializer_range (:obj:`float`, `onal`, defaults to 0.02)
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `onal`, defaults to 1e-12)
            The epsilon used by the layer normalization layers.
        summary_type (:obj:`str`, `onal`, defaults to :obj:`"first"`)
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Has to be one of the following ons

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `onal`, defaults to :obj:`True`)
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `onal`)
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            Pass :obj:`"gelu"` for a gelu activation to the output, any other value will result in no activation.
        summary_last_dropout (:obj:`float`, `onal`, defaults to 0.0)
            Argument used when doing sequence summary. Used in the sequence classification and multiple choice models.

            The dropout ratio to be used after the projection and activation.
        position_embedding_type (:obj:`str`, `onal`, defaults to :obj:`"absolute"`)
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.

    Examples::

        >>> from transformers import PeerModel, PeerConfig

        >>> # Initializing a PEER peer-base-uncased style configuration
        >>> configuration = PeerConfig()

        >>> # Initializing a model from the peer-base-uncased style configuration
        >>> model = PeerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'peer'

    def __init__(self,
                 vocab_size=30522,
                 embedding_size=128,
                 hidden_size=256,
                 num_hidden_layers=12,
                 num_hidden_layers_shared=3,
                 num_hidden_layers_gen=6,
                 num_attention_heads=4,
                 intermediate_size=1024,
                 hidden_act='gelu',
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 summary_type='first',
                 summary_use_proj=True,
                 summary_activation='gelu',
                 summary_last_dropout=0.1,
                 pad_token_id=0,
                 position_embedding_type='absolute',
                 gen_weight=1,
                 dis_weight=50,
                 dis_weight_scheduler=1,
                 augmentation_copies=1,
                 augmentation_temperature=1,
                 absolute_position_embedding=1,
                 relative_position_embedding=32,
                 seq_side_info_embeddings=0,
                 cold_start_epochs=1.25,
                 debug_config=dict(),
                 rtd_levels=2,
                 rtd_level_thresholds='',
                 ranking_start_epoch=1.0,
                 real_token_rank_for_good_estimate=5,
                 rank_sampl_prop=0.3,
                 rank_sampl_range=100,
                 rank_delta_factor=0.0,
                 rank_level_compare_method=0,
                 weight_loss_low_levels=1.0,
                 weight_loss_low_levels_setting='1.0-1.0',
                 weight_loss_low_levels_scheduler=0,
                 weight_loss_level_compos=1,
                 mask_da=0,
                 mask_da_start_epoch=0.0,
                 mask_da_mlm_topk_val=0,
                 mask_ratio_setting='0.15-0.15',
                 mask_ratio_scheduler=0,
                 mask_ratio_stage1_epochs=0.0,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layers_shared = num_hidden_layers_shared
        self.num_hidden_layers_gen = num_hidden_layers_gen
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_last_dropout = summary_last_dropout
        if type(position_embedding_type) == str:
            position_embedding_type = position_embedding_type.split('+')
        self.position_embedding_type = position_embedding_type
        self.augmentation_temperature = augmentation_temperature

        self.gen_weight = gen_weight
        self.dis_weight = dis_weight
        self.dis_weight_scheduler = dis_weight_scheduler
        self.augmentation_copies = augmentation_copies

        self.absolute_position_embedding = absolute_position_embedding
        self.relative_position_embedding = relative_position_embedding
        self.seq_side_info_embeddings = seq_side_info_embeddings

        self.cold_start_epochs = cold_start_epochs
        self.debug_config = debug_config

        self.rtd_levels = rtd_levels
        self.rtd_level_thresholds = rtd_level_thresholds
        self.ranking_start_epoch = ranking_start_epoch
        self.real_token_rank_for_good_estimate = real_token_rank_for_good_estimate
        self.rank_sampl_prop = rank_sampl_prop
        self.rank_sampl_range = rank_sampl_range
        self.rank_delta_factor = rank_delta_factor
        self.rank_level_compare_method = rank_level_compare_method
        self.weight_loss_low_levels = weight_loss_low_levels
        self.weight_loss_low_levels_setting = weight_loss_low_levels_setting
        self.weight_loss_low_levels_scheduler = weight_loss_low_levels_scheduler
        self.weight_loss_level_compos = weight_loss_level_compos

        self.mask_da = mask_da
        self.mask_da_start_epoch = mask_da_start_epoch
        self.mask_da_mlm_topk_val = mask_da_mlm_topk_val

        self.mask_ratio_setting = mask_ratio_setting
        self.mask_ratio_scheduler = mask_ratio_scheduler
        self.mask_ratio_stage1_epochs = mask_ratio_stage1_epochs
