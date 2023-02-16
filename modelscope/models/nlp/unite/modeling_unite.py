# Copyright (c) Alibaba, Inc. and its affiliates.
"""PyTorch UniTE model."""

import warnings
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from packaging import version
from torch.nn import (Dropout, Linear, Module, Parameter, ParameterList,
                      Sequential)
from torch.nn.functional import softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import XLMRobertaConfig, XLMRobertaModel
from transformers.activations import ACT2FN

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['UniTEForTranslationEvaluation']


def _layer_norm_all(tensor, mask_float):
    broadcast_mask = mask_float.unsqueeze(dim=-1)
    num_elements_not_masked = broadcast_mask.sum() * tensor.size(-1)
    tensor_masked = tensor * broadcast_mask

    mean = tensor_masked.sum([-1, -2, -3],
                             keepdim=True) / num_elements_not_masked
    variance = (((tensor_masked - mean) * broadcast_mask)**2).sum(
        [-1, -2, -3], keepdim=True) / num_elements_not_masked

    return (tensor - mean) / torch.sqrt(variance + 1e-12)


class LayerwiseAttention(Module):

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        dropout: float = None,
    ) -> None:
        super(LayerwiseAttention, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.dropout = dropout

        self.scalar_parameters = Parameter(
            torch.zeros((num_layers, ), requires_grad=True))
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(
                self.scalar_parameters)).fill_(-1e20)
            self.register_buffer('dropout_mask', dropout_mask)
            self.register_buffer('dropout_fill', dropout_fill)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        tensors = torch.cat(list(x.unsqueeze(dim=0) for x in tensors), dim=0)
        normed_weights = softmax(
            self.scalar_parameters, dim=0).view(-1, 1, 1, 1)

        mask_float = mask.float()
        weighted_sum = (normed_weights
                        * _layer_norm_all(tensors, mask_float)).sum(dim=0)
        weighted_sum = weighted_sum[:, 0, :]

        return self.gamma * weighted_sum


class FeedForward(Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = [3072, 768],
        activations: str = 'Sigmoid',
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        """
        Feed Forward Neural Network.

        Args:
        in_dim (:obj:`int`):
            Number of input features.
        out_dim (:obj:`int`, defaults to 1):
            Number of output features. Default is 1 -- a single scalar.
        hidden_sizes (:obj:`List[int]`, defaults to `[3072, 768]`):
            List with hidden layer sizes.
        activations (:obj:`str`, defaults to `Sigmoid`):
            Name of the activation function to be used in the hidden layers.
        final_activation (:obj:`str`, Optional, defaults to `None`):
            Name of the final activation function if any.
        dropout (:obj:`float`, defaults to 0.1):
            Dropout ratio to be used in the hidden layers.
        """
        super().__init__()
        modules = []
        modules.append(Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(Dropout(dropout))

        modules.append(Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = Sequential(*modules)

    def build_activation(self, activation: str) -> Module:
        return ACT2FN[activation]

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)


@MODELS.register_module(Tasks.translation_evaluation, module_name=Models.unite)
class UniTEForTranslationEvaluation(TorchModel):

    def __init__(self,
                 attention_probs_dropout_prob: float = 0.1,
                 bos_token_id: int = 0,
                 eos_token_id: int = 2,
                 pad_token_id: int = 1,
                 hidden_act: str = 'gelu',
                 hidden_dropout_prob: float = 0.1,
                 hidden_size: int = 1024,
                 initializer_range: float = 0.02,
                 intermediate_size: int = 4096,
                 layer_norm_eps: float = 1e-05,
                 max_position_embeddings: int = 512,
                 num_attention_heads: int = 16,
                 num_hidden_layers: int = 24,
                 type_vocab_size: int = 1,
                 use_cache: bool = True,
                 vocab_size: int = 250002,
                 mlp_hidden_sizes: List[int] = [3072, 1024],
                 mlp_act: str = 'tanh',
                 mlp_final_act: Optional[str] = None,
                 mlp_dropout: float = 0.1,
                 **kwargs):
        r"""The UniTE Model which outputs the scalar to describe the corresponding
            translation quality of hypothesis. The model architecture includes two
            modules: a pre-trained language model (PLM) to derive representations,
            and a multi-layer perceptron (MLP) to give predicted score.

            Args:
                attention_probs_dropout_prob (:obj:`float`, defaults to 0.1):
                    The dropout ratio for attention weights inside PLM.
                bos_token_id (:obj:`int`, defaults to 0):
                    The numeric id representing beginning-of-sentence symbol.
                eos_token_id (:obj:`int`, defaults to 2):
                    The numeric id representing ending-of-sentence symbol.
                pad_token_id (:obj:`int`, defaults to 1):
                    The numeric id representing padding symbol.
                hidden_act (:obj:`str`, defaults to :obj:`"gelu"`):
                    Activation inside PLM.
                hidden_dropout_prob (:obj:`float`, defaults to 0.1):
                    The dropout ratio for activation states inside PLM.
                hidden_size (:obj:`int`, defaults to 1024):
                    The dimensionality of PLM.
                initializer_range (:obj:`float`, defaults to 0.02):
                    The hyper-parameter for initializing PLM.
                intermediate_size (:obj:`int`, defaults to 4096):
                    The dimensionality of PLM inside feed-forward block.
                layer_norm_eps (:obj:`float`, defaults to 1e-5):
                    The value for setting epsilon to avoid zero-division inside
                        layer normalization.
                max_position_embeddings: (:obj:`int`, defaults to 512):
                    The maximum value for identifying the length of input sequence.
                num_attention_heads (:obj:`int`, defaults to 16):
                    The number of attention heads inside multi-head attention layer.
                num_hidden_layers (:obj:`int`, defaults to 24):
                    The number of layers inside PLM.
                type_vocab_size (:obj:`int`, defaults to 1):
                    The number of type embeddings.
                use_cache (:obj:`bool`, defaults to :obj:`True`):
                    Whether to use cached buffer to initialize PLM.
                vocab_size (:obj:`int`, defaults to 250002):
                    The size of vocabulary.
                mlp_hidden_sizes (:obj:`List[int]`, defaults to `[3072, 1024]`):
                    The size of hidden states inside MLP.
                mlp_act (:obj:`str`, defaults to :obj:`"tanh"`):
                    Activation inside MLP.
                mlp_final_act (:obj:`str`, `optional`, defaults to :obj:`None`):
                    Activation at the end of MLP.
                mlp_dropout (:obj:`float`, defaults to 0.1):
                    The dropout ratio for MLP.
            """
        super().__init__(**kwargs)

        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.use_cache = use_cache
        self.vocab_size = vocab_size
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.mlp_act = mlp_act
        self.mlp_final_act = mlp_final_act
        self.mlp_dropout = mlp_dropout

        self.encoder_config = XLMRobertaConfig(
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            use_cache=self.use_cache)

        self.encoder = XLMRobertaModel(
            self.encoder_config, add_pooling_layer=False)

        self.layerwise_attention = LayerwiseAttention(
            num_layers=self.num_hidden_layers + 1,
            model_dim=self.hidden_size,
            dropout=self.mlp_dropout)

        self.estimator = FeedForward(
            in_dim=self.hidden_size,
            out_dim=1,
            hidden_sizes=self.mlp_hidden_sizes,
            activations=self.mlp_act,
            final_activation=self.mlp_final_act,
            dropout=self.mlp_dropout)

        return

    def forward(self, input_sentences: List[torch.Tensor]):
        input_ids = self.combine_input_sentences(input_sentences)
        attention_mask = input_ids.ne(self.pad_token_id).long()
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True)
        mix_states = self.layerwise_attention(outputs['hidden_states'],
                                              attention_mask)
        pred = self.estimator(mix_states)
        return pred.squeeze(dim=-1)

    def load_checkpoint(self, path: str, device: torch.device):
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
        logger.info('Loading checkpoint parameters from %s' % path)
        return

    def combine_input_sentences(self, input_sent_groups: List[torch.Tensor]):
        for input_sent_group in input_sent_groups[1:]:
            input_sent_group[:, 0] = self.eos_token_id

        if len(input_sent_groups) == 3:
            cutted_sents = self.cut_long_sequences3(input_sent_groups)
        else:
            cutted_sents = self.cut_long_sequences2(input_sent_groups)
        return cutted_sents

    @staticmethod
    def cut_long_sequences2(all_input_concat: List[List[torch.Tensor]],
                            maximum_length: int = 512,
                            pad_idx: int = 1):
        all_input_concat = list(zip(*all_input_concat))
        collected_tuples = list()
        for tensor_tuple in all_input_concat:
            all_lens = tuple(len(x) for x in tensor_tuple)

            if sum(all_lens) > maximum_length:
                lengths = dict(enumerate(all_lens))
                lengths_sorted_idxes = list(x[0] for x in sorted(
                    lengths.items(), key=lambda d: d[1], reverse=True))

                offset = ceil((sum(lengths.values()) - maximum_length) / 2)

                if min(all_lens) > (maximum_length
                                    // 2) and min(all_lens) > offset:
                    lengths = dict((k, v - offset) for k, v in lengths.items())
                else:
                    lengths[lengths_sorted_idxes[
                        0]] = maximum_length - lengths[lengths_sorted_idxes[1]]

                new_lens = list(lengths[k]
                                for k in range(0, len(tensor_tuple)))
                new_tensor_tuple = tuple(
                    x[:y] for x, y in zip(tensor_tuple, new_lens))
                for x, y in zip(new_tensor_tuple, tensor_tuple):
                    x[-1] = y[-1]
                collected_tuples.append(new_tensor_tuple)
            else:
                collected_tuples.append(tensor_tuple)

        concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
        all_input_concat_padded = pad_sequence(
            concat_tensor, batch_first=True, padding_value=pad_idx)

        return all_input_concat_padded

    @staticmethod
    def cut_long_sequences3(all_input_concat: List[List[torch.Tensor]],
                            maximum_length: int = 512,
                            pad_idx: int = 1):
        all_input_concat = list(zip(*all_input_concat))
        collected_tuples = list()
        for tensor_tuple in all_input_concat:
            all_lens = tuple(len(x) for x in tensor_tuple)

            if sum(all_lens) > maximum_length:
                lengths = dict(enumerate(all_lens))
                lengths_sorted_idxes = list(x[0] for x in sorted(
                    lengths.items(), key=lambda d: d[1], reverse=True))

                offset = ceil((sum(lengths.values()) - maximum_length) / 3)

                if min(all_lens) > (maximum_length
                                    // 3) and min(all_lens) > offset:
                    lengths = dict((k, v - offset) for k, v in lengths.items())
                else:
                    while sum(lengths.values()) > maximum_length:
                        if lengths[lengths_sorted_idxes[0]] > lengths[
                                lengths_sorted_idxes[1]]:
                            offset = maximum_length - lengths[
                                lengths_sorted_idxes[1]] - lengths[
                                    lengths_sorted_idxes[2]]
                            if offset > lengths[lengths_sorted_idxes[1]]:
                                lengths[lengths_sorted_idxes[0]] = offset
                            else:
                                lengths[lengths_sorted_idxes[0]] = lengths[
                                    lengths_sorted_idxes[1]]
                        elif lengths[lengths_sorted_idxes[0]] == lengths[
                                lengths_sorted_idxes[1]] > lengths[
                                    lengths_sorted_idxes[2]]:
                            offset = (maximum_length
                                      - lengths[lengths_sorted_idxes[2]]) // 2
                            if offset > lengths[lengths_sorted_idxes[2]]:
                                lengths[lengths_sorted_idxes[0]] = lengths[
                                    lengths_sorted_idxes[1]] = offset
                            else:
                                lengths[lengths_sorted_idxes[0]] = lengths[
                                    lengths_sorted_idxes[1]] = lengths[
                                        lengths_sorted_idxes[2]]
                        else:
                            lengths[lengths_sorted_idxes[0]] = lengths[
                                lengths_sorted_idxes[1]] = lengths[
                                    lengths_sorted_idxes[
                                        2]] = maximum_length // 3

                new_lens = list(lengths[k] for k in range(0, len(lengths)))
                new_tensor_tuple = tuple(
                    x[:y] for x, y in zip(tensor_tuple, new_lens))

                for x, y in zip(new_tensor_tuple, tensor_tuple):
                    x[-1] = y[-1]
                collected_tuples.append(new_tensor_tuple)
            else:
                collected_tuples.append(tensor_tuple)

        concat_tensor = list(torch.cat(x, dim=0) for x in collected_tuples)
        all_input_concat_padded = pad_sequence(
            concat_tensor, batch_first=True, padding_value=pad_idx)

        return all_input_concat_padded
