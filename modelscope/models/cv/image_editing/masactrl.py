# ------------------------------------------------------------------------
# Modified from https://github.com/TencentARC/MasaCtrl/blob/main/masactrl/masactrl.py
# Copyright (c) 2023 TencentARC. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
from einops import rearrange

from .masactrl_utils import AttentionBase


class MutualSelfAttentionControl(AttentionBase):

    def __init__(self,
                 start_step=4,
                 start_layer=10,
                 layer_idx=None,
                 step_idx=None,
                 total_steps=50):
        """
        Mutual self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__()
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(
            range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(
            range(start_step, total_steps))  # denoise index
        print('step_idx: ', self.step_idx)
        print('layer_idx: ', self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet,
                   num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, '(b h) n d -> h (b n) d', h=num_heads)
        k = rearrange(k, '(b h) n d -> h (b n) d', h=num_heads)
        v = rearrange(v, '(b h) n d -> h (b n) d', h=num_heads)

        sim = torch.einsum('h i d, h j d -> h i j', q, k) * kwargs.get('scale')
        attn = sim.softmax(-1)
        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h (b n) d -> b n (h d)', b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads,
                **kwargs):
        """
        Attention forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return super().forward(q, k, v, sim, attn, is_cross, place_in_unet,
                                   num_heads, **kwargs)

        qu, qc = q.chunk(2)  # uncond, cond
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        attnu, attnc = attn.chunk(2)
        # uncond
        # ku[:num_heads], vu[:num_heads] -> source
        # qu -> [source, target]
        out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads],
                                sim[:num_heads], attnu, is_cross,
                                place_in_unet, num_heads, **kwargs)
        out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads],
                                sim[:num_heads], attnc, is_cross,
                                place_in_unet, num_heads, **kwargs)
        out = torch.cat([out_u, out_c], dim=0)

        return out
