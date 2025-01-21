# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from functools import partialmethod
from typing import List, Optional

import torch
import torch.nn as nn
from unicore.modules import LayerNorm, softmax_dropout
from unicore.utils import permute_final_dims

from .common import Linear, chunk_layer


def gen_attn_mask(mask, neg_inf):
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


class Attention(nn.Module):

    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = Linear(q_dim, total_dim, bias=False, init='glorot')
        self.linear_k = Linear(k_dim, total_dim, bias=False, init='glorot')
        self.linear_v = Linear(v_dim, total_dim, bias=False, init='glorot')
        self.linear_o = Linear(total_dim, q_dim, init='final')
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(q_dim, total_dim, init='gating')
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g = None
        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)

        q = self.linear_q(q)
        q *= self.norm
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(
            -2, -3).contiguous()
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(
            -2, -3).contiguous()
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)

        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k

        attn = softmax_dropout(attn, 0, self.training, mask=mask, bias=bias)
        o = torch.matmul(attn, v)
        del attn, v

        o = o.transpose(-2, -3).contiguous()
        o = o.view(*o.shape[:-2], -1)

        if g is not None:
            o = torch.sigmoid(g) * o

        # merge heads
        o = nn.functional.linear(o, self.linear_o.weight)
        return o

    def get_output_bias(self):
        return self.linear_o.bias


class GlobalAttention(nn.Module):

    def __init__(self, input_dim, head_dim, num_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.num_heads = num_heads
        self.inf = inf
        self.eps = eps
        self.linear_q = Linear(
            input_dim, head_dim * num_heads, bias=False, init='glorot')
        self.linear_k = Linear(input_dim, head_dim, bias=False, init='glorot')
        self.linear_v = Linear(input_dim, head_dim, bias=False, init='glorot')
        self.linear_g = Linear(input_dim, head_dim * num_heads, init='gating')
        self.linear_o = Linear(head_dim * num_heads, input_dim, init='final')
        self.sigmoid = nn.Sigmoid()
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # gating
        g = self.sigmoid(self.linear_g(x))

        k = self.linear_k(x)
        v = self.linear_v(x)

        q = torch.sum(
            x * mask.unsqueeze(-1), dim=-2) / (
                torch.sum(mask, dim=-1, keepdims=True) + self.eps)
        q = self.linear_q(q)
        q *= self.norm
        q = q.view(q.shape[:-1] + (self.num_heads, -1))

        attn = torch.matmul(q, k.transpose(-1, -2))
        del q, k

        attn_mask = gen_attn_mask(mask, -self.inf)[..., :, None, :]
        attn = softmax_dropout(attn, 0, self.training, mask=attn_mask)

        o = torch.matmul(
            attn,
            v,
        )
        del attn, v

        g = g.view(g.shape[:-1] + (self.num_heads, -1))
        o = o.unsqueeze(-3) * g
        del g

        # merge heads
        o = o.reshape(o.shape[:-2] + (-1, ))
        return self.linear_o(o)


def gen_msa_attn_mask(mask, inf, gen_col_mask=True):
    row_mask = gen_attn_mask(mask, -inf)[..., :, None, None, :]
    if gen_col_mask:
        col_mask = gen_attn_mask(mask.transpose(-1, -2), -inf)[..., :, None,
                                                               None, :]
        return row_mask, col_mask
    else:
        return row_mask


class MSAAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_hid,
        num_heads,
        pair_bias=False,
        d_pair=None,
    ):
        super(MSAAttention, self).__init__()

        self.pair_bias = pair_bias
        self.layer_norm_m = LayerNorm(d_in)
        self.layer_norm_z = None
        self.linear_z = None
        if self.pair_bias:
            self.layer_norm_z = LayerNorm(d_pair)
            self.linear_z = Linear(
                d_pair, num_heads, bias=False, init='normal')

        self.mha = Attention(d_in, d_in, d_in, d_hid, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        chunk_size: int = None,
    ) -> torch.Tensor:

        return chunk_layer(
            self._attn_forward,
            {
                'm': m,
                'mask': mask,
                'bias': bias
            },
            chunk_size=chunk_size,
            num_batch_dims=len(m.shape[:-2]),
        )

    @torch.jit.ignore
    def _attn_chunk_forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = 2560,
    ) -> torch.Tensor:
        m = self.layer_norm_m(m)
        num_chunk = (m.shape[-3] + chunk_size - 1) // chunk_size
        outputs = []
        for i in range(num_chunk):
            chunk_start = i * chunk_size
            chunk_end = min(m.shape[-3], chunk_start + chunk_size)
            cur_m = m[..., chunk_start:chunk_end, :, :]
            cur_mask = (
                mask[..., chunk_start:chunk_end, :, :, :]
                if mask is not None else None)
            outputs.append(
                self.mha(q=cur_m, k=cur_m, v=cur_m, mask=cur_mask, bias=bias))
        return torch.cat(outputs, dim=-3)

    def _attn_forward(self, m, mask, bias: Optional[torch.Tensor] = None):
        m = self.layer_norm_m(m)
        return self.mha(q=m, k=m, v=m, mask=mask, bias=bias)

    def forward(
        self,
        m: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:

        bias = None
        if self.pair_bias:
            z = self.layer_norm_z(z)
            bias = (
                permute_final_dims(self.linear_z(z),
                                   (2, 0, 1)).unsqueeze(-4).contiguous())

        if chunk_size is not None:
            m = self._chunk(m, attn_mask, bias, chunk_size)
        else:
            attn_chunk_size = 2560
            if m.shape[-3] <= attn_chunk_size:
                m = self._attn_forward(m, attn_mask, bias)
            else:
                # reduce the peak memory cost in extra_msa_stack
                return self._attn_chunk_forward(
                    m, attn_mask, bias, chunk_size=attn_chunk_size)

        return m

    def get_output_bias(self):
        return self.mha.get_output_bias()


class MSARowAttentionWithPairBias(MSAAttention):

    def __init__(self, d_msa, d_pair, d_hid, num_heads):
        super(MSARowAttentionWithPairBias, self).__init__(
            d_msa,
            d_hid,
            num_heads,
            pair_bias=True,
            d_pair=d_pair,
        )


class MSAColumnAttention(MSAAttention):

    def __init__(self, d_msa, d_hid, num_heads):
        super(MSAColumnAttention, self).__init__(
            d_in=d_msa,
            d_hid=d_hid,
            num_heads=num_heads,
            pair_bias=False,
            d_pair=None,
        )

    def forward(
        self,
        m: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        m = m.transpose(-2, -3)
        m = super().forward(m, attn_mask=attn_mask, chunk_size=chunk_size)
        m = m.transpose(-2, -3)

        return m


class MSAColumnGlobalAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_hid,
        num_heads,
        inf=1e9,
        eps=1e-10,
    ):
        super(MSAColumnGlobalAttention, self).__init__()

        self.layer_norm_m = LayerNorm(d_in)
        self.global_attention = GlobalAttention(
            d_in,
            d_hid,
            num_heads,
            inf=inf,
            eps=eps,
        )

    @torch.jit.ignore
    def _chunk(
        self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._attn_forward,
            {
                'm': m,
                'mask': mask
            },
            chunk_size=chunk_size,
            num_batch_dims=len(m.shape[:-2]),
        )

    def _attn_forward(self, m, mask):
        m = self.layer_norm_m(m)
        return self.global_attention(m, mask=mask)

    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._attn_forward(m, mask=mask)

        m = m.transpose(-2, -3)
        return m


def gen_tri_attn_mask(mask, inf):
    start_mask = gen_attn_mask(mask, -inf)[..., :, None, None, :]
    end_mask = gen_attn_mask(mask.transpose(-1, -2), -inf)[..., :, None,
                                                           None, :]
    return start_mask, end_mask


class TriangleAttention(nn.Module):

    def __init__(
        self,
        d_in,
        d_hid,
        num_heads,
        starting,
    ):
        super(TriangleAttention, self).__init__()
        self.starting = starting
        self.layer_norm = LayerNorm(d_in)
        self.linear = Linear(d_in, num_heads, bias=False, init='normal')
        self.mha = Attention(d_in, d_in, d_in, d_hid, num_heads)

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        chunk_size: int = None,
    ) -> torch.Tensor:
        return chunk_layer(
            self.mha,
            {
                'q': x,
                'k': x,
                'v': x,
                'mask': mask,
                'bias': bias
            },
            chunk_size=chunk_size,
            num_batch_dims=len(x.shape[:-2]),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if not self.starting:
            x = x.transpose(-2, -3)

        x = self.layer_norm(x)
        triangle_bias = (
            permute_final_dims(self.linear(x),
                               (2, 0, 1)).unsqueeze(-4).contiguous())

        if chunk_size is not None:
            x = self._chunk(x, attn_mask, triangle_bias, chunk_size)
        else:
            x = self.mha(q=x, k=x, v=x, mask=attn_mask, bias=triangle_bias)

        if not self.starting:
            x = x.transpose(-2, -3)
        return x

    def get_output_bias(self):
        return self.mha.get_output_bias()


class TriangleAttentionStarting(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=True)


class TriangleAttentionEnding(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
