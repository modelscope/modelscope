# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Github: https://github.com/kakaobrain/bassl
# ------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertEncoder


class ShotEmbedding(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        nn_size = cfg.neighbor_size + 2  # +1 for center shot, +1 for cls
        self.shot_embedding = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.position_embedding = nn.Embedding(nn_size, cfg.hidden_size)
        self.mask_embedding = nn.Embedding(2, cfg.input_dim, padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(cfg.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)

        self.register_buffer('pos_ids',
                             torch.arange(nn_size, dtype=torch.long))

    def forward(
        self,
        shot_emb: torch.Tensor,
        mask: torch.Tensor = None,
        pos_ids: torch.Tensor = None,
    ) -> torch.Tensor:

        assert len(shot_emb.size()) == 3

        if pos_ids is None:
            pos_ids = self.pos_ids

        # this for mask embedding (un-masked ones remain unchanged)
        if mask is not None:
            self.mask_embedding.weight.data[0, :].fill_(0)
            mask_emb = self.mask_embedding(mask.long())
            shot_emb = (shot_emb * (1 - mask).float()[:, :, None]) + mask_emb

        # we set [CLS] token to averaged feature
        cls_emb = shot_emb.mean(dim=1)

        # embedding shots
        shot_emb = torch.cat([cls_emb[:, None, :], shot_emb], dim=1)
        shot_emb = self.shot_embedding(shot_emb)
        pos_emb = self.position_embedding(pos_ids)
        embeddings = shot_emb + pos_emb[None, :]
        embeddings = self.dropout(self.LayerNorm(embeddings))
        return embeddings


class TransformerCRN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.pooling_method = cfg.pooling_method
        self.shot_embedding = ShotEmbedding(cfg)
        self.encoder = BertEncoder(cfg)

        nn_size = cfg.neighbor_size + 2  # +1 for center shot, +1 for cls
        self.register_buffer(
            'attention_mask',
            self._get_extended_attention_mask(
                torch.ones((1, nn_size)).float()),
        )

    def forward(
        self,
        shot: torch.Tensor,
        mask: torch.Tensor = None,
        pos_ids: torch.Tensor = None,
        pooling_method: str = None,
    ):
        if self.attention_mask.shape[1] != (shot.shape[1] + 1):
            n_shot = shot.shape[1] + 1  # +1 for CLS token
            attention_mask = self._get_extended_attention_mask(
                torch.ones((1, n_shot), dtype=torch.float, device=shot.device))
        else:
            attention_mask = self.attention_mask

        shot_emb = self.shot_embedding(shot, mask=mask, pos_ids=pos_ids)
        encoded_emb = self.encoder(
            shot_emb, attention_mask=attention_mask).last_hidden_state

        return encoded_emb, self.pooler(
            encoded_emb, pooling_method=pooling_method)

    def pooler(self, sequence_output, pooling_method=None):
        if pooling_method is None:
            pooling_method = self.pooling_method

        if pooling_method == 'cls':
            return sequence_output[:, 0, :]
        elif pooling_method == 'avg':
            return sequence_output[:, 1:].mean(dim=1)
        elif pooling_method == 'max':
            return sequence_output[:, 1:].max(dim=1)[0]
        elif pooling_method == 'center':
            cidx = sequence_output.shape[1] // 2
            return sequence_output[:, cidx, :]
        else:
            raise ValueError

    def _get_extended_attention_mask(self, attention_mask):

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f'Wrong shape for attention_mask (shape {attention_mask.shape})'
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
