# Modified by Zhipu.AI
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
"""GPT-2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron_util import mpu, print_rank_0

from modelscope.models.nlp.mglm.model.prompt import PromptSpell
from .transformer import GPT2ParallelTransformer


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GLMModel(torch.nn.Module):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(
        self,
        num_layers,
        vocab_size,
        hidden_size,
        num_attention_heads,
        embedding_dropout_prob,
        attention_dropout_prob,
        output_dropout_prob,
        max_sequence_length,
        max_memory_length,
        checkpoint_activations,
        checkpoint_num_layers=1,
        parallel_output=True,
        relative_encoding=False,
        block_position_encoding=False,
        output_predict=True,
        spell_length=None,
        spell_func='lstm',
        attention_scale=1.0,
    ):

        super(GLMModel, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = GPT2ParallelTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            max_sequence_length,
            max_memory_length,
            embedding_dropout_prob,
            attention_dropout_prob,
            output_dropout_prob,
            checkpoint_activations,
            checkpoint_num_layers,
            attention_scale=attention_scale,
            relative_encoding=relative_encoding,
            block_position_encoding=block_position_encoding)
        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size,
                                            spell_func)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = 'Freeze transformer'
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f' tune {tune_prefix_layers} prefix layers'
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self,
                input_ids,
                position_ids,
                attention_mask,
                *mems,
                return_memory=False,
                detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(
                batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        transformer_output = self.transformer(
            embeddings,
            position_ids,
            attention_mask,
            mems,
            return_memory=return_memory,
            detach_memory=detach_memory)
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        if self.output_predict:
            # Parallel logits.
            logits_parallel = mpu.copy_to_model_parallel_region(logits)
            logits_parallel = F.linear(logits_parallel,
                                       self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel, *outputs)

            return (mpu.gather_from_model_parallel_region(logits_parallel),
                    *outputs)
        else:
            return (logits, *outputs)


class EncoderDecoder(torch.nn.Module):
    """Seq2Seq Transformer Model
    The output of the forward method are the logits (parallel or serial depending on the `parallel_output` flag).
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 output_predict=True):
        super(EncoderDecoder, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.encoder = GPT2ParallelTransformer(
            num_layers, hidden_size, num_attention_heads, max_sequence_length,
            max_memory_length, embedding_dropout_prob, attention_dropout_prob,
            output_dropout_prob, checkpoint_activations, checkpoint_num_layers)
        self.decoder = GPT2ParallelTransformer(
            num_layers,
            hidden_size,
            num_attention_heads,
            max_sequence_length,
            max_memory_length,
            embedding_dropout_prob,
            attention_dropout_prob,
            output_dropout_prob,
            checkpoint_activations,
            checkpoint_num_layers,
            use_decoder_layer=True)

    def forward(self, source_ids, target_ids, source_position_ids,
                target_position_ids, source_mask, target_mask):
        # Embeddings.
        source_embeddings = self.word_embeddings(source_ids)
        target_embeddings = self.word_embeddings(target_ids)

        # Transformer.
        encoder_output, _ = self.encoder(source_embeddings,
                                         source_position_ids, source_mask)
        decoder_output, _ = self.decoder(target_embeddings,
                                         target_position_ids, target_mask)
        if self.output_predict:
            # Parallel logits.
            output_parallel = mpu.copy_to_model_parallel_region(decoder_output)
            logits_parallel = F.linear(output_parallel,
                                       self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel, )

            return (mpu.gather_from_model_parallel_region(logits_parallel), )
        else:
            return (decoder_output, )


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend([
                p for p in list(module_._parameters.values())
                if p is not None and p.requires_grad
            ])
        else:
            weight_decay_params['params'].extend([
                p for n, p in list(module_._parameters.items())
                if p is not None and p.requires_grad and n != 'bias'
            ])
            no_weight_decay_params['params'].extend([
                p for n, p in list(module_._parameters.items())
                if p is not None and p.requires_grad and n == 'bias'
            ])

    return weight_decay_params, no_weight_decay_params
