# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------

import copy
import math
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(hidden_size, dropout=dropout)
        self.layers = get_clones(EncoderLayer(hidden_size, heads, dropout), N)
        self.norm = Norm(hidden_size)

    def forward(self, x, mask=None, require_att=False):
        att = None
        for i in range(self.N):
            if mask is None:
                if i == (self.N - 1):
                    x, att = self.layers[i](x, require_att=True)
                else:
                    x = self.layers[i](x)
            else:
                x = self.layers[i](x, mask)
        if require_att:
            return x, att
        else:
            return x


class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(inplace=True)  # newly added
        )

    def forward(self, x):
        out = self.linear(x)
        return out


class Transformer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, heads,
                 dropout):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.encoder = Encoder(input_size, hidden_size, n_layers, heads,
                               dropout)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x, mask=None, require_att=False):
        x = self.linear(x)
        att = None
        if mask is None:
            # evaluation model
            if require_att:
                embedding, att = self.encoder(x, require_att=True)
            else:
                embedding = self.encoder(x)

            output = self.decoder(embedding)

            if require_att:
                return output, att
            else:
                return output
        else:
            if require_att:
                embedding, att = self.encoder(x, mask, require_att=True)
            else:
                embedding = self.encoder(x, mask)

            output = self.decoder(embedding)
            return output


class Norm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            mask = mask.to(torch.float32)
            mask2d = torch.matmul(mask, mask.transpose(-2, -1)).expand(
                scores.shape[0], scores.shape[1], scores.shape[2],
                scores.shape[3])
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
            mask = mask.to(torch.float32)
            mask2d = mask.expand(scores.shape[0], scores.shape[1],
                                 scores.shape[2], scores.shape[3])

        scores = scores.masked_fill(mask2d == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


def attention_score(q, k, v, d_k):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    return scores


class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention_map(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention_score(q, k, v, self.d_k)

        return scores

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer

        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Embedder(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len=900, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                sin_coef = 10000**((2 * i) / d_model)
                cos_coef = 10000**((2 * (i + 1)) / d_model)
                pe[pos, i] = math.sin(pos / sin_coef)
                pe[pos, i + 1] = math.cos(pos / cos_coef)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, require_att=False):
        x2 = self.norm_1(x)
        xc = x2.clone()

        if mask is None:
            x = x + self.dropout_1(self.attn(x2, x2, x2))
        else:
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))

        if require_att:
            att = self.attn.attention_map(xc, xc, xc)
            return x, att
        else:
            return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Stacker(nn.Module):
    '''
    The architecture of the stacking regressor, which takes the dense representations and
    logical locations of table cells to make more accurate prediction of logical locations.
    '''

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 layers,
                 heads=8,
                 dropout=0.1):
        """
        Args:
            input_size : The dim of logical locations which is always 4.
            hidden_size : The dim of hidden states which is 256 by default.
            output_size : The dim of logical locations which is always 4.
            layers : Number of layers of self-attention mechanism, which is 4 in this implementation.
        """
        super(Stacker, self).__init__()
        self.logi_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True))
        self.tsfm = Transformer(2 * hidden_size, hidden_size, output_size,
                                layers, heads, dropout)

    def forward(self, outputs, logi, mask=None, require_att=False):
        """
        Args:
            outputs : The dense representation of table cells, a tensor of [batch_size, number_of_objects, hidden_size].
            logi : The logical location of table cells, a tensor of [batch_size, number_of_objects, 4].
            mask : The mask of cells, a tensor of [batch_size, number_of_objects], not None only in training stage.
            require_att :  If True, the model will also generate the attention maps of table cells.

        Returns:
            stacked_axis : The predicted logical location of cells, a tensor of [batch_size, number_of_objects, 4].
            att : The attention map of table cells.
        """
        logi_embeddings = self.logi_encoder(logi)

        cat_embeddings = torch.cat((logi_embeddings, outputs), dim=2)

        if mask is None:
            if require_att:
                stacked_axis, att = self.tsfm(cat_embeddings)
            else:
                stacked_axis = self.tsfm(cat_embeddings)
        else:
            stacked_axis = self.tsfm(cat_embeddings, mask=mask)

        if require_att:
            return stacked_axis, att
        else:
            return stacked_axis


class LoreProcessModel(nn.Module):
    '''
    The logical location prediction head of LORE. It contains a base regressor and a stacking regressor.
    They both consist of several self-attention blocks.
    See details in paper "LORE: Logical Location Regression Network for Table Structure Recognition"
    (https://arxiv.org/abs/2303.03730).
    '''

    def __init__(self, **kwargs):
        '''
            Args:
        '''
        super(LoreProcessModel, self).__init__()

        self.input_size = 256
        self.output_size = 4
        self.hidden_size = 256
        self.max_fmp_size = 256
        self.stacking_layers = 4
        self.tsfm_layers = 4
        self.num_heads = 8
        self.att_dropout = 0.1
        self.stacker = Stacker(self.output_size, self.hidden_size,
                               self.output_size, self.stacking_layers)
        self.tsfm_axis = Transformer(self.input_size, self.hidden_size,
                                     self.output_size, self.tsfm_layers,
                                     self.num_heads, self.att_dropout)
        self.x_position_embeddings = nn.Embedding(self.max_fmp_size,
                                                  self.hidden_size)
        self.y_position_embeddings = nn.Embedding(self.max_fmp_size,
                                                  self.hidden_size)

    def forward(self, outputs, batch=None, cc_match=None, dets=None):
        """
        Args:
            outputs : The dense representation of table cells from the detection part of LORE,
                      a tensor of [batch_size, number_of_objects, hidden_size].
            batch : The detection results of other source, such as external OCR systems.
            dets : The detection results of each table cells, a tensor of [batch_size, number_of_objects, 8].

        Returns:
            logi_axis : The output logical location of base regressor,
                        a tensor of [batch_size, number_of_objects, 4].
            stacked_axis : The output logical location of stacking regressor,
                           a tensor of [batch_size, number_of_objects, 4].
        """
        if batch is None:
            # evaluation mode
            vis_feat = outputs

        if batch is None:
            if dets is None:
                logic_axis = self.tsfm_axis(vis_feat)
                stacked_axis = self.stacker(vis_feat, logic_axis)
            else:
                left_pe = self.x_position_embeddings(dets[:, :, 0])
                upper_pe = self.y_position_embeddings(dets[:, :, 1])
                right_pe = self.x_position_embeddings(dets[:, :, 2])
                lower_pe = self.y_position_embeddings(dets[:, :, 5])
                feat = vis_feat + left_pe + upper_pe + right_pe + lower_pe

                logic_axis = self.tsfm_axis(feat)

                stacked_axis = self.stacker(feat, logic_axis)

        return logic_axis, stacked_axis
