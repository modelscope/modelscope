# Copyright 2022-2023 The Alibaba Fundamental Vision  Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Q2VRankerStage1(nn.Module):
    """
        Used to calculate the qv_ctx_score with query embedding and multi anchor context embeddings as input.
        The qv_ctx_score is used to pre-rank and retain top-k related anchors.
    """

    def __init__(self, nscales, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nscales = nscales

    def forward(self, ctx_feats, qfeat):
        qfeat = self.fc(qfeat)
        qv_ctx_scores = list()
        for i in range(self.nscales):
            score = torch.einsum('bld,bd->bl',
                                 F.normalize(ctx_feats[i], p=2, dim=2),
                                 F.normalize(qfeat, p=2, dim=1))
            qv_ctx_scores.append(score)

        return qv_ctx_scores


class V2QRankerStage1(nn.Module):
    """
        Used to calculate the vq_ctx_score with anchor context embeddings and multi query embeddings as input.
    """

    def __init__(self, nscales, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nscales = nscales

    def forward(self, ctx_feats, qfeat):
        vq_ctx_scores = list()
        for i in range(self.nscales):
            score = torch.einsum(
                'bld,bd->bl', F.normalize(self.fc(ctx_feats[i]), p=2, dim=2),
                F.normalize(qfeat, p=2, dim=1))
            vq_ctx_scores.append(score)

        return vq_ctx_scores


class Q2VRankerStage2(nn.Module):
    """
        Used to calculate the qv_ctn_score with query embedding and video sequence embedding as input.
        The qv_ctn_score is used to re-rank anchors.
    """

    def __init__(self, nscales, hidden_dim, snippet_length=10):
        super().__init__()
        self.nscales = nscales
        self.snippet_length = snippet_length
        self.qfc = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = V2VAttention()

    def forward(self, vfeats, qfeat, hit_indices, qv_ctx_scores):
        qfeat = self.qfc(qfeat)

        qv_ctn_scores = list()
        qv_merge_scores = list()

        _, L, D = vfeats.size()
        ctn_feats = list()
        for i in range(self.nscales):
            anchor_length = self.snippet_length * 2**i
            assert L // anchor_length == qv_ctx_scores[i].size(1)
            qv_ctx_score = torch.index_select(qv_ctx_scores[i], 1,
                                              hit_indices[i])

            ctn_feat = vfeats.view(L // anchor_length, anchor_length,
                                   D).detach()
            ctn_feat = torch.index_select(ctn_feat, 0, hit_indices[i])
            ctn_feat = self.encoder(
                ctn_feat,
                torch.ones(ctn_feat.size()[:2], device=ctn_feat.device))
            ctn_feats.append(ctn_feat)

            qv_ctn_score = torch.einsum(
                'bkld,bd->bkl', F.normalize(ctn_feat.unsqueeze(0), p=2, dim=3),
                F.normalize(qfeat, p=2, dim=1))
            qv_ctn_score, _ = torch.max(qv_ctn_score, dim=2)
            qv_ctn_scores.append(qv_ctn_score)
            qv_merge_scores.append(qv_ctx_score + qv_ctn_score)

        return qv_merge_scores, qv_ctn_scores, ctn_feats


class V2QRankerStage2(nn.Module):
    """
        Used to calculate the vq_ctn_score with anchor content embeddings and multi query embeddings as input.
    """

    def __init__(self, nscales, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.nscales = nscales

    def forward(self, ctn_feats, qfeat):
        vq_ctn_scores = list()
        for i in range(self.nscales):
            score = torch.einsum(
                'bkld,bd->bkl',
                F.normalize(self.fc(ctn_feats[i]).unsqueeze(0), p=2, dim=3),
                F.normalize(qfeat, p=2, dim=1))
            score = torch.mean(score, dim=2)
            vq_ctn_scores.append(score)

        return vq_ctn_scores


class V2VAttention(nn.Module):
    """
        Self-attention encoder for anchor frame sequence to encode intra-anchor knowledge.
    """

    def __init__(self):
        super().__init__()
        self.posemb = PositionEncoding(max_len=400, dim=512, dropout=0.0)
        self.encoder = MultiHeadAttention(dim=512, n_heads=8, dropout=0.1)
        self.dropout = nn.Dropout(0.0)

    def forward(self, video_feats, video_masks):
        mask = torch.einsum('bm,bn->bmn', video_masks,
                            video_masks).unsqueeze(1)
        residual = video_feats
        video_feats = video_feats + self.posemb(video_feats)
        out = self.encoder(
            query=video_feats, key=video_feats, value=video_feats, mask=mask)
        video_feats = self.dropout(residual
                                   + out) * video_masks.unsqueeze(2).float()
        return video_feats


class BboxRegressor(nn.Module):
    """
        Predict the offset of bounding box for each candidate anchor.
    """

    def __init__(self, hidden_dim, enable_stage2=False):
        super().__init__()
        self.fc_ctx = nn.Linear(hidden_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)

        if enable_stage2:
            self.fc_ctn = nn.Linear(hidden_dim, hidden_dim)
            self.attn = SelfAttention(hidden_dim)
            self.predictor = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2))
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2))
        self.enable_stage2 = enable_stage2

    def forward(self, ctx_feats, ctn_feats, qfeat):
        qfeat = self.fc_q(qfeat)

        ctx_feats = torch.cat(ctx_feats, dim=1)
        ctx_fuse_feats = F.relu(self.fc_ctx(ctx_feats)) * F.relu(
            qfeat.unsqueeze(1))

        if self.enable_stage2 and ctn_feats:
            ctn_fuse_feats = list()
            for i in range(len(ctn_feats)):
                out = F.relu(self.fc_ctn(ctn_feats[i]).unsqueeze(0)) * F.relu(
                    qfeat.unsqueeze(1).unsqueeze(1))
                out = self.attn(out)
                ctn_fuse_feats.append(out)
            ctn_fuse_feats = torch.cat(ctn_fuse_feats, dim=1)
            fuse_feats = torch.cat([ctx_fuse_feats, ctn_fuse_feats], dim=-1)
        else:
            fuse_feats = ctx_fuse_feats

        out = self.predictor(fuse_feats)
        return out


class SelfAttention(nn.Module):
    """
        Obtain pooled features by self-attentive pooling.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        att = self.fc2(self.relu(self.fc1(x))).squeeze(3)
        att = F.softmax(att, dim=2).unsqueeze(3)
        out = torch.sum(x * att, dim=2)
        return out


class PositionEncoding(nn.Module):
    """
        An implementation of trainable positional embedding which is added to
        sequence features to inject time/position information.

        Args:
            max_len: The max number of trainable positional embeddings.
            dim: the dimension of positional embedding.
    """

    def __init__(self, max_len, dim, dropout=0.0):
        super(PositionEncoding, self).__init__()

        self.embed = nn.Embedding(max_len, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_ids = pos_ids.unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.dropout(self.relu(self.embed(pos_ids)))

        return pos_emb


class MultiHeadAttention(nn.Module):
    """
        An implementation of multi-head attention module, as described in
        'Attention Is All You Need <https://arxiv.org/abs/1706.03762>'

        Args:
            dim: the dimension of features of hidden layers.
            n_heads: the number of head.
    """

    def __init__(self, dim, n_heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query, key, value, mask):
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)

        q_trans = self.transpose_for_scores(q)
        k_trans = self.transpose_for_scores(k)
        v_trans = self.transpose_for_scores(v)

        att = torch.matmul(q_trans, k_trans.transpose(-1,
                                                      -2))  # (N, nh, Lq, L)
        att = att / math.sqrt(self.head_dim)
        att = mask_logits(att, mask)
        att = self.softmax(att)
        att = self.dropout(att)

        ctx_v = torch.matmul(att, v_trans)  # (N, nh, Lq, dh)
        ctx_v = ctx_v.permute(0, 2, 1, 3).contiguous()  # (N, Lq, nh, dh)
        shape = ctx_v.size()[:-2] + (self.dim, )
        ctx_v = ctx_v.view(*shape)  # (N, Lq, D)
        return ctx_v


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value
