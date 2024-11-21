# Copyright 2021-2022 The Alibaba DAMO Team Authors. All rights reserved.
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

from __future__ import absolute_import, division, print_function
import os.path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (AutoConfig, DPRConfig, DPRQuestionEncoder,
                          MT5ForConditionalGeneration, RagTokenForGeneration,
                          XLMRobertaForSequenceClassification, XLMRobertaModel,
                          XLMRobertaTokenizer)

from modelscope.utils.logger import get_logger

logger = get_logger()


class Wrapper(nn.Module):

    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        return self.encoder(input_ids, attention_mask).pooler_output


class DPRModel(nn.Module):

    def __init__(self, model_dir, config):
        super().__init__()
        self.config = config

        qry_encoder = XLMRobertaModel(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'qry_encoder')))
        ctx_encoder = XLMRobertaModel(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'ctx_encoder')))
        self.qry_encoder = Wrapper(qry_encoder)
        self.ctx_encoder = Wrapper(ctx_encoder)
        self.loss_fct = nn.CrossEntropyLoss()

    @staticmethod
    def encode(model, input_ids, attention_mask, gck_segment=32):
        dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        pooled_output = []
        for mini_batch in range(0, input_ids.shape[0], gck_segment):
            mini_batch_input_ids = input_ids[mini_batch:mini_batch
                                             + gck_segment]
            mini_batch_attention_mask = attention_mask[mini_batch:mini_batch
                                                       + gck_segment]
            mini_batch_pooled_output = checkpoint(model, mini_batch_input_ids,
                                                  mini_batch_attention_mask,
                                                  dummy_tensor)
            pooled_output.append(mini_batch_pooled_output)
        return torch.cat(pooled_output, dim=0)

    def forward(self,
                query_input_ids,
                query_attention_mask,
                context_input_ids,
                context_attention_mask,
                labels,
                gck_segment=32):
        query_vector = self.encode(self.qry_encoder, query_input_ids,
                                   query_attention_mask, gck_segment)
        context_vector = self.encode(self.ctx_encoder, context_input_ids,
                                     context_attention_mask, gck_segment)
        logits = torch.matmul(query_vector, context_vector.T)
        loss = self.loss_fct(logits, labels)
        return loss, logits


class ClassifyRerank(nn.Module):

    def __init__(self, model_dir):
        super().__init__()
        self.base_model = XLMRobertaForSequenceClassification.from_pretrained(
            model_dir)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                *args,
                **kwargs):
        outputs = self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return outputs


class Rerank(nn.Module):

    def __init__(self, encoder, top_k):
        super().__init__()
        self.encoder = encoder
        self.top_k = top_k

    def forward(self, inputs):
        model = self.encoder
        logits = F.log_softmax(model(**inputs)[0], dim=-1)[:, 1]
        logits = logits.view(-1, self.top_k)
        logprobs = F.log_softmax(logits, dim=-1)
        return logprobs


class Re2GModel(nn.Module):

    def __init__(self, model_dir, config):
        super(Re2GModel, self).__init__()
        self.config = config
        self.top_k = self.config['top_k']
        encoder = XLMRobertaForSequenceClassification(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'rerank')))
        generator = MT5ForConditionalGeneration(
            config=AutoConfig.from_pretrained(
                os.path.join(model_dir, 'generation')))

        self.rerank = Rerank(encoder, self.top_k)

        dpr_config = DPRConfig()
        dpr_config.vocab_size = encoder.config.vocab_size
        rag_model = RagTokenForGeneration(
            question_encoder=DPRQuestionEncoder(dpr_config),
            generator=generator)
        rag_model.rag.question_encoder = None
        self.generator = rag_model

    def forward(self, rerank_input_ids, input_ids, attention_mask, label_ids):
        doc_scores = self.rerank(rerank_input_ids)

        outputs = self.generator(
            labels=label_ids,
            context_input_ids=input_ids,
            context_attention_mask=attention_mask,
            doc_scores=doc_scores,
            n_docs=self.top_k)
        return outputs

    def generate(self, rerank_input_ids, input_ids, attention_mask):
        doc_scores = self.rerank(rerank_input_ids)

        beam_search_output = self.generator.generate(
            n_docs=self.top_k,
            encoder_input_ids=input_ids,
            context_input_ids=input_ids,
            context_attention_mask=attention_mask,
            doc_scores=doc_scores,
            num_beams=self.config['num_beams'],
            max_length=self.config['target_sequence_length'],
            early_stopping=True,
            no_repeat_ngram_size=self.config['no_repeat_ngram_size'],
            return_dict_in_generate=True,
            output_scores=True)
        generated_ids = beam_search_output.detach().cpu().numpy()

        return generated_ids
