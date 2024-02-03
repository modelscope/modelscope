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
"""PyTorch BERT model."""

import copy
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from modelscope.models.multi_modal.prost.models.until_config import \
    PreCrossConfig


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish}


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CrossEn(nn.Module):

    def __init__(self, config=None):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank:ctx.batch_size
                        * (ctx.rank + 1)],
            None,
        )


class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""
    # https://github.com/PyTorchLightning/lightning-bolts/blob/8d3fbf7782e3d3937ab8a1775a7092d7567f2933/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1)
                           * ctx.batch_size], None)


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PreCrossConfig):
            raise ValueError(
                'Parameter config in `{}(config)` should be an instance of class `PreCrossConfig`. '
                'To create a model from a Google pretrained model use '
                '`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`'.format(
                    self.__class__.__name__, self.__class__.__name__))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata,
                                         True, missing_keys, unexpected_keys,
                                         error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        # if prefix is None and (task_config is None or task_config.local_rank == 0):
        #     logger.info("-" * 20)
        #     if len(missing_keys) > 0:
        #         logger.info("Weights of {} not initialized from pretrained model: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
        #     if len(unexpected_keys) > 0:
        #         logger.info("Weights from pretrained model not used in {}: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
        #     if len(error_msgs) > 0:
        #         logger.error("Weights from pretrained model cause errors in {}: {}"
        #                      .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items()
                          if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model


class PatchShiftModule(nn.Module):

    def __init__(self, net, video_frame, n_div):
        super().__init__()
        self.net = net
        self.video_frame = video_frame
        self.n_div = n_div

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                need_weights=True,
                attn_mask=None):
        # here q == k == v, psm means patch shift output
        x = query  # shape here is LND, not NLD (50, 384, 768)
        x = x.permute(1, 0, 2)  # LND -> NLD
        patch_len = x.shape[-2]
        fold = patch_len // self.n_div
        x = x.reshape(-1, self.video_frame, x.shape[-2],
                      x.shape[-1])  # shape = [bs, frame, grid ** 2, width]
        psm = torch.zeros_like(x)  # shape = [bs, frame, grid ** 2, width]
        psm[:, :, :, :] = x[:, :, :, :]
        lshift_indices = torch.arange(start=1, end=patch_len, step=fold)
        psm[:, 1:, lshift_indices, :] = x[:, :-1,
                                          lshift_indices, :]  # f_t = f_t-1
        rshift_indices = torch.arange(start=1 + 3, end=patch_len, step=fold)
        psm[:, :-1, rshift_indices, :] = x[:, 1:,
                                           rshift_indices, :]  # f_t = f_t+1
        x = psm.reshape(-1, patch_len, x.shape[-1])
        x = x.permute(1, 0, 2)  # NLD -> LND

        return self.net(
            x, x, x, need_weights=need_weights, attn_mask=attn_mask)


def make_patch_shift(net, video_frame=12, shift_layers=4, n_div=7):
    '''
    Args:
    net: CLIP
    video_frame: need predefine here
    shift_layers: layers to be shift
    '''

    def make_trans_patch_shift(stage, shift_layers):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            if i >= 10 and i <= 11:
                blocks[i].attn = PatchShiftModule(
                    b.attn, video_frame=video_frame, n_div=n_div)
        return nn.Sequential(*blocks)

    net.clip.visual.transformer.resblocks = make_trans_patch_shift(
        net.clip.visual.transformer.resblocks, shift_layers=shift_layers)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Event_Layer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 is_weights=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_vis = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before
        self.is_weights = is_weights

    def forward(self, tgt, memory, pos=None, query_pos=None):

        tgt = self.norm1(tgt)
        memory = self.norm2(memory)
        tgt = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm3(tgt)

        tgt2, atten_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm5(tgt)

        return tgt, atten_weights


def adaptive_mask(aa, bb, ada_para):
    tensor = torch.zeros((aa, bb))
    adaptive_num = int(bb * ada_para)
    cc = int(bb / aa)
    for i in range(aa):
        start_col = i * cc
        end_col = start_col + cc + adaptive_num
        if end_col > bb - 1:
            tmp = end_col - (bb - 1)
            start_col = start_col - tmp
            if start_col < 0:
                start_col = 0
            end_col = bb
        tensor[i, start_col:end_col] = 1
    tensor = ~tensor.bool()
    return tensor


class Frame_Layer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 para=1.0,
                 dropout=0.1,
                 activation='relu',
                 normalize_before=False,
                 is_weights=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_vis = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)
        self.normalize_before = normalize_before
        self.is_weights = is_weights
        self.mask_para = para

    def forward(self, tgt, memory, pos=None, query_pos=None):
        tgt = self.norm1(tgt)
        memory = self.norm2(memory)
        mask_new = adaptive_mask(tgt.shape[0], memory.shape[0], ada_para=0.2)
        if torch.cuda.is_available():
            mask_new = mask_new.cuda()
        tgt2, atten_weights = self.multihead_attn(
            tgt, memory, memory, attn_mask=mask_new)
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm4(tgt)

        return tgt, atten_weights


class TransDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, pos=None, query_pos=None):
        output = tgt

        intermediate = []
        all_weights = []

        for layer in self.layers:
            output, weights = layer(
                output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                all_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(all_weights)
        return output.unsqueeze(0)


class Event_decoder(nn.Module):

    def __init__(self,
                 num_attris=3,
                 layers=1,
                 heads=1,
                 dim_ftr=512,
                 pos_emb=False,
                 length=1,
                 dim_feedforward=512,
                 without_init=False):
        super().__init__()
        embedding_dim = dim_ftr

        d_model = dim_ftr
        dim_feedforward = dim_feedforward

        self.V = nn.Parameter(
            torch.Tensor(num_attris, dim_feedforward), requires_grad=True)
        nn.init.xavier_uniform_(self.V)
        decoder_layer = Event_Layer(
            d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(
            decoder_layer,
            layers,
            nn.LayerNorm(d_model),
            return_intermediate=True)
        self.use_pos_enc = pos_emb

        if self.use_pos_enc:
            self.position_encoding_pre = positionalencoding2d(
                embedding_dim, 14, 14).unsqueeze(0)

    def forward(self, features):
        batch_size = features.shape[0]
        if self.use_pos_enc:  # False
            pos_encoding = self.position_encoding_pre(
                features,
                torch.zeros(features.shape[0], 14, 14,
                            dtype=torch.bool).cuda())
            features = features + pos_encoding

        enco_others = features.permute(1, 0, 2)
        h_attr = self.V
        h_attr_batch = h_attr.unsqueeze(0).repeat(batch_size, 1, 1)
        h_attr_batch = h_attr_batch.permute(1, 0, 2)

        hs, _ = self.event_decoder(h_attr_batch, enco_others)
        hs = hs[-1].permute(1, 0, 2)
        return hs


class Frame_decoder(nn.Module):

    def __init__(self,
                 num_attris=3,
                 layers=1,
                 heads=1,
                 dim_ftr=512,
                 pos_emb=False,
                 length=1,
                 dim_feedforward=512,
                 without_init=False):
        super().__init__()
        embedding_dim = dim_ftr
        d_model = dim_ftr
        dim_feedforward = dim_feedforward

        self.V = nn.Parameter(
            torch.Tensor(num_attris, dim_feedforward), requires_grad=True)
        nn.init.xavier_uniform_(self.V)
        decoder_layer = Frame_Layer(
            d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(
            decoder_layer,
            layers,
            nn.LayerNorm(d_model),
            return_intermediate=True)
        self.use_pos_enc = pos_emb

        if self.use_pos_enc:
            self.position_encoding_pre = positionalencoding2d(
                embedding_dim, 14, 14).unsqueeze(0)

    def forward(self, features):
        batch_size = features.shape[0]
        if self.use_pos_enc:
            pos_encoding = self.position_encoding_pre(
                features,
                torch.zeros(features.shape[0], 14, 14,
                            dtype=torch.bool).cuda())
            features = features + pos_encoding

        enco_others = features.permute(1, 0, 2)
        h_attr = self.V
        h_attr_batch = h_attr.unsqueeze(0).repeat(batch_size, 1, 1)
        h_attr_batch = h_attr_batch.permute(1, 0, 2)

        hs, _ = self.event_decoder(h_attr_batch, enco_others)
        hs = hs[-1].permute(1, 0, 2)

        return hs
