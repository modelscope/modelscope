# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from unicore.modules import LayerNorm
from unicore.utils import (checkpoint_sequential, permute_final_dims,
                           tensor_tree_map)

from .attentions import (Attention, TriangleAttentionEnding,
                         TriangleAttentionStarting, gen_attn_mask)
from .common import (Linear, SimpleModuleList, Transition,
                     bias_dropout_residual, chunk_layer, residual,
                     tri_mul_residual)
from .featurization import build_template_pair_feat_v2
from .triangle_multiplication import (TriangleMultiplicationIncoming,
                                      TriangleMultiplicationOutgoing)


class TemplatePointwiseAttention(nn.Module):

    def __init__(self, d_template, d_pair, d_hid, num_heads, inf, **kwargs):
        super(TemplatePointwiseAttention, self).__init__()

        self.inf = inf

        self.mha = Attention(
            d_pair,
            d_template,
            d_template,
            d_hid,
            num_heads,
            gating=False,
        )

    def _chunk(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        mha_inputs = {
            'q': z,
            'k': t,
            'v': t,
            'mask': mask,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            num_batch_dims=len(z.shape[:-2]),
        )

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        template_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        mask = gen_attn_mask(template_mask, -self.inf)[..., None, None, None,
                                                       None, :]
        z = z.unsqueeze(-2)

        t = permute_final_dims(t, (1, 2, 0, 3))

        if chunk_size is not None:
            z = self._chunk(z, t, mask, chunk_size)
        else:
            z = self.mha(z, t, t, mask=mask)

        z = z.squeeze(-2)

        return z


class TemplateProjection(nn.Module):

    def __init__(self, d_template, d_pair, **kwargs):
        super(TemplateProjection, self).__init__()

        self.d_pair = d_pair
        self.act = nn.ReLU()
        self.output_linear = Linear(d_template, d_pair, init='relu')

    def forward(self, t, z) -> torch.Tensor:
        if t is None:
            # handle for non-template case
            shape = z.shape
            shape[-1] = self.d_pair
            t = torch.zeros(shape, dtype=z.dtype, device=z.device)
        t = self.act(t)
        z_t = self.output_linear(t)
        return z_t


class TemplatePairStackBlock(nn.Module):

    def __init__(
        self,
        d_template: int,
        d_hid_tri_att: int,
        d_hid_tri_mul: int,
        num_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        tri_attn_first: bool,
        inf: float,
        **kwargs,
    ):
        super(TemplatePairStackBlock, self).__init__()

        self.tri_att_start = TriangleAttentionStarting(
            d_template,
            d_hid_tri_att,
            num_heads,
        )
        self.tri_att_end = TriangleAttentionEnding(
            d_template,
            d_hid_tri_att,
            num_heads,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            d_template,
            d_hid_tri_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            d_template,
            d_hid_tri_mul,
        )

        self.pair_transition = Transition(
            d_template,
            pair_transition_n,
        )
        self.tri_attn_first = tri_attn_first
        self.dropout = dropout_rate
        self.row_dropout_share_dim = -3
        self.col_dropout_share_dim = -2

    def forward(
        self,
        s: torch.Tensor,
        mask: torch.Tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        if self.tri_attn_first:
            s = bias_dropout_residual(
                self.tri_att_start,
                s,
                self.tri_att_start(
                    s, attn_mask=tri_start_attn_mask, chunk_size=chunk_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
            )

            s = bias_dropout_residual(
                self.tri_att_end,
                s,
                self.tri_att_end(
                    s, attn_mask=tri_end_attn_mask, chunk_size=chunk_size),
                self.col_dropout_share_dim,
                self.dropout,
                self.training,
            )
            s = tri_mul_residual(
                self.tri_mul_out,
                s,
                self.tri_mul_out(s, mask=mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
                block_size=block_size,
            )

            s = tri_mul_residual(
                self.tri_mul_in,
                s,
                self.tri_mul_in(s, mask=mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
                block_size=block_size,
            )
        else:
            s = tri_mul_residual(
                self.tri_mul_out,
                s,
                self.tri_mul_out(s, mask=mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
                block_size=block_size,
            )

            s = tri_mul_residual(
                self.tri_mul_in,
                s,
                self.tri_mul_in(s, mask=mask, block_size=block_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
                block_size=block_size,
            )

            s = bias_dropout_residual(
                self.tri_att_start,
                s,
                self.tri_att_start(
                    s, attn_mask=tri_start_attn_mask, chunk_size=chunk_size),
                self.row_dropout_share_dim,
                self.dropout,
                self.training,
            )

            s = bias_dropout_residual(
                self.tri_att_end,
                s,
                self.tri_att_end(
                    s, attn_mask=tri_end_attn_mask, chunk_size=chunk_size),
                self.col_dropout_share_dim,
                self.dropout,
                self.training,
            )
        s = residual(s, self.pair_transition(
            s,
            chunk_size=chunk_size,
        ), self.training)
        return s


class TemplatePairStack(nn.Module):

    def __init__(
        self,
        d_template,
        d_hid_tri_att,
        d_hid_tri_mul,
        num_blocks,
        num_heads,
        pair_transition_n,
        dropout_rate,
        tri_attn_first,
        inf=1e9,
        **kwargs,
    ):
        super(TemplatePairStack, self).__init__()

        self.blocks = SimpleModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                TemplatePairStackBlock(
                    d_template=d_template,
                    d_hid_tri_att=d_hid_tri_att,
                    d_hid_tri_mul=d_hid_tri_mul,
                    num_heads=num_heads,
                    pair_transition_n=pair_transition_n,
                    dropout_rate=dropout_rate,
                    inf=inf,
                    tri_attn_first=tri_attn_first,
                ))

        self.layer_norm = LayerNorm(d_template)

    def forward(
        self,
        single_templates: Tuple[torch.Tensor],
        mask: torch.tensor,
        tri_start_attn_mask: torch.Tensor,
        tri_end_attn_mask: torch.Tensor,
        templ_dim: int,
        chunk_size: int,
        block_size: int,
        return_mean: bool,
    ):

        def one_template(i):
            (s, ) = checkpoint_sequential(
                functions=[
                    partial(
                        b,
                        mask=mask,
                        tri_start_attn_mask=tri_start_attn_mask,
                        tri_end_attn_mask=tri_end_attn_mask,
                        chunk_size=chunk_size,
                        block_size=block_size,
                    ) for b in self.blocks
                ],
                input=(single_templates[i], ),
            )
            return s

        n_templ = len(single_templates)
        if n_templ > 0:
            new_single_templates = [one_template(0)]
            if return_mean:
                t = self.layer_norm(new_single_templates[0])
            for i in range(1, n_templ):
                s = one_template(i)
                if return_mean:
                    t = residual(t, self.layer_norm(s), self.training)
                else:
                    new_single_templates.append(s)

        if return_mean:
            if n_templ > 0:
                t /= n_templ
            else:
                t = None
        else:
            t = torch.cat(
                [s.unsqueeze(templ_dim) for s in new_single_templates],
                dim=templ_dim)
            t = self.layer_norm(t)

        return t
