# The Uni-fold implementation is also open-sourced by the authors under Apache-2.0 license,
# and is publicly available at https://github.com/dptech-corp/Uni-Fold.

from functools import partialmethod
from typing import List, Optional

import torch
import torch.nn as nn
from unicore.modules import LayerNorm
from unicore.utils import permute_final_dims

from .common import Linear


class TriangleMultiplication(nn.Module):

    def __init__(self, d_pair, d_hid, outgoing=True):
        super(TriangleMultiplication, self).__init__()
        self.outgoing = outgoing

        self.linear_ab_p = Linear(d_pair, d_hid * 2)
        self.linear_ab_g = Linear(d_pair, d_hid * 2, init='gating')

        self.linear_g = Linear(d_pair, d_pair, init='gating')
        self.linear_z = Linear(d_hid, d_pair, init='final')

        self.layer_norm_in = LayerNorm(d_pair)
        self.layer_norm_out = LayerNorm(d_hid)

        self._alphafold_original_mode = False

    def _chunk_2d(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        block_size: int = None,
    ) -> torch.Tensor:

        # avoid too small chunk size
        # block_size = max(block_size, 256)
        new_z = z.new_zeros(z.shape)
        dim1 = z.shape[-3]

        def _slice_linear(z, linear: Linear, a=True):
            d_hid = linear.bias.shape[0] // 2
            index = 0 if a else d_hid
            p = (
                nn.functional.linear(z, linear.weight[index:index + d_hid])
                + linear.bias[index:index + d_hid])
            return p

        def _chunk_projection(z, mask, a=True):
            p = _slice_linear(z, self.linear_ab_p, a) * mask
            p *= torch.sigmoid(_slice_linear(z, self.linear_ab_g, a))
            return p

        num_chunk = (dim1 + block_size - 1) // block_size
        for i in range(num_chunk):
            chunk_start = i * block_size
            chunk_end = min(chunk_start + block_size, dim1)
            if self.outgoing:
                a_chunk = _chunk_projection(
                    z[..., chunk_start:chunk_end, :, :],
                    mask[..., chunk_start:chunk_end, :, :],
                    a=True,
                )
                a_chunk = permute_final_dims(a_chunk, (2, 0, 1))
            else:
                a_chunk = _chunk_projection(
                    z[..., :, chunk_start:chunk_end, :],
                    mask[..., :, chunk_start:chunk_end, :],
                    a=True,
                )
                a_chunk = a_chunk.transpose(-1, -3)

            for j in range(num_chunk):
                j_chunk_start = j * block_size
                j_chunk_end = min(j_chunk_start + block_size, dim1)
                if self.outgoing:
                    b_chunk = _chunk_projection(
                        z[..., j_chunk_start:j_chunk_end, :, :],
                        mask[..., j_chunk_start:j_chunk_end, :, :],
                        a=False,
                    )
                    b_chunk = b_chunk.transpose(-1, -3)
                else:
                    b_chunk = _chunk_projection(
                        z[..., :, j_chunk_start:j_chunk_end, :],
                        mask[..., :, j_chunk_start:j_chunk_end, :],
                        a=False,
                    )
                    b_chunk = permute_final_dims(b_chunk, (2, 0, 1))
                x_chunk = torch.matmul(a_chunk, b_chunk)
                del b_chunk
                x_chunk = permute_final_dims(x_chunk, (1, 2, 0))
                x_chunk = self.layer_norm_out(x_chunk)
                x_chunk = self.linear_z(x_chunk)
                x_chunk *= torch.sigmoid(
                    self.linear_g(z[..., chunk_start:chunk_end,
                                    j_chunk_start:j_chunk_end, :]))
                new_z[..., chunk_start:chunk_end,
                      j_chunk_start:j_chunk_end, :] = x_chunk
                del x_chunk
            del a_chunk
        return new_z

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        block_size=None,
    ) -> torch.Tensor:

        mask = mask.unsqueeze(-1)
        if not self._alphafold_original_mode:
            # divided by 1/sqrt(dim) for numerical stability
            mask = mask * (mask.shape[-2]**-0.5)

        z = self.layer_norm_in(z)
        if not self.training and block_size is not None:
            return self._chunk_2d(z, mask, block_size=block_size)

        g = nn.functional.linear(z, self.linear_g.weight)
        if self.training:
            ab = self.linear_ab_p(z) * mask * torch.sigmoid(
                self.linear_ab_g(z))
        else:
            ab = self.linear_ab_p(z)
            ab *= mask
            ab *= torch.sigmoid(self.linear_ab_g(z))
        a, b = torch.chunk(ab, 2, dim=-1)
        del z, ab

        if self.outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = b.transpose(-1, -3)
        else:
            b = permute_final_dims(b, (2, 0, 1))
            a = a.transpose(-1, -3)
        x = torch.matmul(a, b)
        del a, b

        x = permute_final_dims(x, (1, 2, 0))

        x = self.layer_norm_out(x)
        x = nn.functional.linear(x, self.linear_z.weight)
        return x, g

    def get_output_bias(self):
        return self.linear_z.bias, self.linear_g.bias


class TriangleMultiplicationOutgoing(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplication):
    __init__ = partialmethod(TriangleMultiplication.__init__, outgoing=False)
