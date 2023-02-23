# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
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
from torch import nn

from modelscope.utils.logger import get_logger

logger = get_logger()


def _symmetric_kl_div(logits1, logits2, attention_mask=None):
    """
    Calclate two logits' the KL div value symmetrically.
    :param logits1: The first logit.
    :param logits2: The second logit.
    :param attention_mask: An optional attention_mask which is used to mask some element out.
    This is usually useful in token_classification tasks.
    If the shape of logits is [N1, N2, ... Nn, D], the shape of attention_mask should be [N1, N2, ... Nn]
    :return: The mean loss.
    """
    labels_num = logits1.shape[-1]
    KLDiv = nn.KLDivLoss(reduction='none')
    loss = torch.sum(
        KLDiv(nn.LogSoftmax(dim=-1)(logits1),
              nn.Softmax(dim=-1)(logits2)),
        dim=-1) + torch.sum(
            KLDiv(nn.LogSoftmax(dim=-1)(logits2),
                  nn.Softmax(dim=-1)(logits1)),
            dim=-1)
    if attention_mask is not None:
        loss = torch.sum(
            loss * attention_mask) / torch.sum(attention_mask) / labels_num
    else:
        loss = torch.mean(loss) / labels_num
    return loss


def compute_adv_loss(embedding,
                     model,
                     ori_logits,
                     ori_loss,
                     adv_grad_factor,
                     adv_bound=None,
                     sigma=5e-6,
                     **kwargs):
    """
    Calculate the adv loss of the model.
    :param embedding: Original sentense embedding
    :param model: The model, or the forward function(including decoder/classifier),
            accept kwargs as input, output logits
    :param ori_logits: The original logits outputed from the model function
    :param ori_loss: The original loss
    :param adv_grad_factor: This factor will be multipled by the KL loss grad and then the result will be added to
            the original embedding.
            More details please check:https://arxiv.org/abs/1908.04577
            The range of this value always be 1e-3~1e-7
    :param adv_bound: adv_bound is used to cut the top and the bottom bound of the produced embedding.
            If not proveded, 2 * sigma will be used as the adv_bound factor
    :param sigma: The std factor used to produce a 0 mean normal distribution.
            If adv_bound not proveded, 2 * sigma will be used as the adv_bound factor
    :param kwargs: the input param used in model function
    :return: The original loss adds the adv loss
    """
    adv_bound = adv_bound if adv_bound is not None else 2 * sigma
    embedding_1 = embedding + embedding.data.new(embedding.size()).normal_(
        0, sigma)  # 95% in +- 1e-5
    kwargs.pop('input_ids')
    if 'inputs_embeds' in kwargs:
        kwargs.pop('inputs_embeds')
    with_attention_mask = False if 'with_attention_mask' not in kwargs else kwargs[
        'with_attention_mask']
    attention_mask = kwargs['attention_mask']
    if not with_attention_mask:
        attention_mask = None
    if 'with_attention_mask' in kwargs:
        kwargs.pop('with_attention_mask')
    outputs = model(**kwargs, inputs_embeds=embedding_1)
    v1_logits = outputs.logits
    loss = _symmetric_kl_div(ori_logits, v1_logits, attention_mask)
    emb_grad = torch.autograd.grad(loss, embedding_1)[0].data
    emb_grad_norm = emb_grad.norm(
        dim=2, keepdim=True, p=float('inf')).max(
            1, keepdim=True)[0]
    is_nan = torch.any(torch.isnan(emb_grad_norm))
    if is_nan:
        logger.warning('Nan occurred when calculating adv loss.')
        return ori_loss
    emb_grad = emb_grad / (emb_grad_norm + 1e-6)
    embedding_2 = embedding_1 + adv_grad_factor * emb_grad
    embedding_2 = torch.max(embedding_1 - adv_bound, embedding_2)
    embedding_2 = torch.min(embedding_1 + adv_bound, embedding_2)
    outputs = model(**kwargs, inputs_embeds=embedding_2)
    adv_logits = outputs.logits
    adv_loss = _symmetric_kl_div(ori_logits, adv_logits, attention_mask)
    return ori_loss + adv_loss


def compute_adv_loss_pair(embedding,
                          model,
                          start_logits,
                          end_logits,
                          ori_loss,
                          adv_grad_factor,
                          adv_bound=None,
                          sigma=5e-6,
                          **kwargs):
    """
    Calculate the adv loss of the model. This function is used in the pair logits scenerio.
    :param embedding: Original sentense embedding
    :param model: The model, or the forward function(including decoder/classifier),
            accept kwargs as input, output logits
    :param start_logits: The original start logits outputed from the model function
    :param end_logits: The original end logits outputed from the model function
    :param ori_loss: The original loss
    :param adv_grad_factor: This factor will be multipled by the KL loss grad and then the result will be added to
            the original embedding.
            More details please check:https://arxiv.org/abs/1908.04577
            The range of this value always be 1e-3~1e-7
    :param adv_bound: adv_bound is used to cut the top and the bottom bound of the produced embedding.
            If not proveded, 2 * sigma will be used as the adv_bound factor
    :param sigma: The std factor used to produce a 0 mean normal distribution.
            If adv_bound not proveded, 2 * sigma will be used as the adv_bound factor
    :param kwargs: the input param used in model function
    :return: The original loss adds the adv loss
    """
    adv_bound = adv_bound if adv_bound is not None else 2 * sigma
    embedding_1 = embedding + embedding.data.new(embedding.size()).normal_(
        0, sigma)  # 95% in +- 1e-5
    kwargs.pop('input_ids')
    if 'inputs_embeds' in kwargs:
        kwargs.pop('inputs_embeds')
    outputs = model(**kwargs, inputs_embeds=embedding_1)
    v1_logits_start, v1_logits_end = outputs.logits
    loss = _symmetric_kl_div(start_logits,
                             v1_logits_start) + _symmetric_kl_div(
                                 end_logits, v1_logits_end)
    loss = loss / 2
    emb_grad = torch.autograd.grad(loss, embedding_1)[0].data
    emb_grad_norm = emb_grad.norm(
        dim=2, keepdim=True, p=float('inf')).max(
            1, keepdim=True)[0]
    is_nan = torch.any(torch.isnan(emb_grad_norm))
    if is_nan:
        logger.warning('Nan occurred when calculating pair adv loss.')
        return ori_loss
    emb_grad = emb_grad / emb_grad_norm
    embedding_2 = embedding_1 + adv_grad_factor * emb_grad
    embedding_2 = torch.max(embedding_1 - adv_bound, embedding_2)
    embedding_2 = torch.min(embedding_1 + adv_bound, embedding_2)
    outputs = model(**kwargs, inputs_embeds=embedding_2)
    adv_logits_start, adv_logits_end = outputs.logits
    adv_loss = _symmetric_kl_div(start_logits,
                                 adv_logits_start) + _symmetric_kl_div(
                                     end_logits, adv_logits_end)
    return ori_loss + adv_loss
