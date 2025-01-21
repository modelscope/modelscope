# Part of the implementation is borrowed and modified from Video Swin Transformer,
# publicly available at https://github.com/SwinTransformer/Video-Swin-Transformer

from abc import ABCMeta, abstractmethod
from functools import lru_cache, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import torchvision.transforms as T
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_

from modelscope.models import TorchModel


def normal_init(module, mean=0., std=1., bias=0.):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def window_partition(x, window_size):
    """ window_partition function.
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1],
               window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6,
                        7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ window_reverse function.
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1],
                     W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ This is PyTorch impl of TPS

    Window based multi-head self attention (W-MSA) module with relative position bias.
    The coordinates of patches and patches are shifted together using Pattern C.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        shift (bool, optional): If True, conduct shift operation
        shift_type (str, optional): shift operation type, either using 'psm' or 'tsm'
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 shift=False,
                 shift_type='psm'):

        super().__init__()
        self.dim = dim
        window_size = (16, 7, 7)
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.shift = shift
        self.shift_type = shift_type

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                np.prod([2 * ws - 1 for ws in window_size]),
                num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w,
                           indexing='ij'))  # 3, Wd, Wh, Ww
        # Do the same rotation to coords
        coords_old = coords.clone()

        # pattern patternC - 9
        coords[:, :, 0::3, 0::3] = torch.roll(
            coords[:, :, 0::3, 0::3], shifts=-4, dims=1)
        coords[:, :, 0::3, 1::3] = torch.roll(
            coords[:, :, 0::3, 1::3], shifts=1, dims=1)
        coords[:, :, 0::3, 2::3] = torch.roll(
            coords[:, :, 0::3, 2::3], shifts=2, dims=1)
        coords[:, :, 1::3, 2::3] = torch.roll(
            coords[:, :, 1::3, 2::3], shifts=3, dims=1)
        coords[:, :, 1::3, 0::3] = torch.roll(
            coords[:, :, 1::3, 0::3], shifts=-1, dims=1)
        coords[:, :, 2::3, 0::3] = torch.roll(
            coords[:, :, 2::3, 0::3], shifts=-2, dims=1)
        coords[:, :, 2::3, 1::3] = torch.roll(
            coords[:, :, 2::3, 1::3], shifts=-3, dims=1)
        coords[:, :, 2::3, 2::3] = torch.roll(
            coords[:, :, 2::3, 2::3], shifts=4, dims=1)

        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        coords_old_flatten = torch.flatten(coords_old, 1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords_old = coords_old_flatten[:, :,
                                                 None] - coords_old_flatten[:,
                                                                            None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww

        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords_old = relative_coords_old.permute(
            1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3

        relative_coords[:, :,
                        0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords_old[:, :, 0] += self.window_size[
            0] - 1  # shift to start from 0
        relative_coords_old[:, :, 1] += self.window_size[1] - 1
        relative_coords_old[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1]
                                     - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_coords_old[:, :, 0] *= (2 * self.window_size[1]
                                         - 1) * (2 * self.window_size[2] - 1)
        relative_coords_old[:, :, 1] *= (2 * self.window_size[2] - 1)

        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        relative_position_index_old = relative_coords_old.sum(-1)
        relative_position_index = relative_position_index.view(
            window_size[0], window_size[1] * window_size[2], window_size[0],
            window_size[1] * window_size[2]).permute(0, 2, 1, 3).reshape(
                window_size[0] * window_size[0],
                window_size[1] * window_size[2],
                window_size[1] * window_size[2])[::window_size[0], :, :]

        relative_position_index_old = relative_position_index_old.view(
            window_size[0], window_size[1] * window_size[2], window_size[0],
            window_size[1] * window_size[2]).permute(0, 2, 1, 3).reshape(
                window_size[0] * window_size[0],
                window_size[1] * window_size[2],
                window_size[1] * window_size[2])[::window_size[0], :, :]

        self.register_buffer('relative_position_index',
                             relative_position_index)
        self.register_buffer('relative_position_index_old',
                             relative_position_index_old)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        if self.shift and self.shift_type == 'psm':
            self.shift_op = PatchShift(False, 1)
            self.shift_op_back = PatchShift(True, 1)
        elif self.shift and self.shift_type == 'tsm':
            self.shift_op = TemporalShift(8)

    def forward(self, x, mask=None, batch_size=8, frame_len=8):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        if self.shift:
            x = x.view(B_, N, self.num_heads,
                       C // self.num_heads).permute(0, 2, 1, 3)

            x = self.shift_op(x, batch_size, frame_len)
            x = x.permute(0, 2, 1, 3).reshape(B_, N, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.shift and self.shift_type == 'psm':
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:].reshape(-1), :].reshape(
                    frame_len, N, N, -1)  # 8frames ,Wd*Wh*Ww,Wd*Wh*Ww,nH
        else:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index_old[:].reshape(-1), :].reshape(
                    frame_len, N, N, -1)  # 8frames ,Wd*Wh*Ww,Wd*Wh*Ww,nH

        relative_position_bias = relative_position_bias.permute(
            0, 3, 1, 2).contiguous()  # Frames, nH, Wd*Wh*Ww, Wd*Wh*Ww

        attn = attn.view(
            batch_size, frame_len, -1, self.num_heads, N, N).permute(
                0,
                2, 1, 3, 4, 5) + relative_position_bias.unsqueeze(0).unsqueeze(
                    1)  # B_, nH, N, N
        attn = attn.permute(0, 2, 1, 3, 4, 5).view(-1, self.num_heads, N, N)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # Shift back for psm
        if self.shift and self.shift_type == 'psm':
            x = self.shift_op_back(attn @ v, batch_size,
                                   frame_len).transpose(1,
                                                        2).reshape(B_, N, C)
        else:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PatchShift(nn.Module):
    """ This is PyTorch impl of TPS

    The patches are shifted using Pattern C.

    It supports both of shifted and shift back.

    Args:
        inv (bool): whether using inverse shifted (shift back)
        ratio (float): ratio of channels to be shifted, patch shift using 1.0
    """

    def __init__(self, inv=False, ratio=1):
        super(PatchShift, self).__init__()
        self.inv = inv
        self.ratio = ratio
        # if inv:
        # print('=> Using inverse PatchShift, ratio {}, tps'.format(ratio))
        # else:
        # print('=> Using bayershift, ratio {}, tps'.format(ratio))

    def forward(self, x, batch_size, frame_len):
        x = self.shift(
            x,
            inv=self.inv,
            ratio=self.ratio,
            batch_size=batch_size,
            frame_len=frame_len)
        return x

    @staticmethod
    def shift(x, inv=False, ratio=0.5, batch_size=8, frame_len=8):
        B, num_heads, N, c = x.size()
        fold = int(num_heads * ratio)
        feat = x
        feat = feat.view(batch_size, frame_len, -1, num_heads, 7, 7, c)
        out = feat.clone()
        multiplier = 1
        stride = 1
        if inv:
            multiplier = -1

        # Pattern C
        out[:, :, :, :fold, 0::3, 0::3, :] = torch.roll(
            feat[:, :, :, :fold, 0::3, 0::3, :],
            shifts=-4 * multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 0::3, 1::3, :] = torch.roll(
            feat[:, :, :, :fold, 0::3, 1::3, :],
            shifts=multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 1::3, 0::3, :] = torch.roll(
            feat[:, :, :, :fold, 1::3, 0::3, :],
            shifts=-multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 0::3, 2::3, :] = torch.roll(
            feat[:, :, :, :fold, 0::3, 2::3, :],
            shifts=2 * multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 2::3, 0::3, :] = torch.roll(
            feat[:, :, :, :fold, 2::3, 0::3, :],
            shifts=-2 * multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 1::3, 2::3, :] = torch.roll(
            feat[:, :, :, :fold, 1::3, 2::3, :],
            shifts=3 * multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 2::3, 1::3, :] = torch.roll(
            feat[:, :, :, :fold, 2::3, 1::3, :],
            shifts=-3 * multiplier * stride,
            dims=1)
        out[:, :, :, :fold, 2::3, 2::3, :] = torch.roll(
            feat[:, :, :, :fold, 2::3, 2::3, :],
            shifts=4 * multiplier * stride,
            dims=1)

        out = out.view(B, num_heads, N, c)
        return out


class TemporalShift(nn.Module):
    """ This is PyTorch impl of TPS

    The temporal channel shift.

    The code is adopted from TSM: Temporal Shift Module for Efficient Video Understanding. ICCV19

    https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py

    Args:
        n_div (int): propotion of channel to be shifted.
    """

    def __init__(self, n_div=8):
        super(TemporalShift, self).__init__()
        self.fold_div = n_div

    def forward(self, x, batch_size, frame_len):
        x = self.shift(
            x,
            fold_div=self.fold_div,
            batch_size=batch_size,
            frame_len=frame_len)
        return x

    @staticmethod
    def shift(x, fold_div=8, batch_size=8, frame_len=8):
        B, num_heads, N, c = x.size()
        fold = c // fold_div
        feat = x
        feat = feat.view(batch_size, frame_len, -1, num_heads, N, c)
        out = feat.clone()

        out[:, 1:, :, :, :, :fold] = feat[:, :-1, :, :, :, :fold]  # shift left
        out[:, :-1, :, :, :,
            fold:2 * fold] = feat[:, 1:, :, :, :, fold:2 * fold]  # shift right

        out = out.view(B, num_heads, N, c)

        return out


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block from Video Swin Transformer.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 shift=False,
                 shift_type='psm'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.shift = shift
        self.shift_type = shift_type

        assert 0 <= self.shift_size[0] < self.window_size[
            0], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[1] < self.window_size[
            1], 'shift_size must in 0-window_size'
        assert 0 <= self.shift_size[2] < self.window_size[
            2], 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            shift=self.shift,
            shift_type=self.shift_type)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x,
                                     window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=attn_mask, batch_size=B,
            frame_len=D)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C, )))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp,
                                   Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer from Video Swin Transformer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0],
                                           -shift_size[0]), slice(
                                               -shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1],
                                               -shift_size[1]), slice(
                                                   -shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2],
                                                   -shift_size[2]), slice(
                                                       -shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask,
                                    window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                      float(-100.0)).masked_fill(
                                          attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage from Video Swin Transformer.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 shift_type='psm'):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.shift_type = shift_type

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                shift=True,
                shift_type='tsm' if (i % 2 == 0 and self.shift_type == 'psm')
                or self.shift_type == 'tsm' else 'psm',
            ) for i in range(depth)
        ])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size,
                                                  self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding from Video Swin Transformer.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=(2, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(
                x,
                (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class SwinTransformer2D_TPS(nn.Module):
    """
        Code is adopted from Video Swin Transformer.

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                shift_type='psm')
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers - 1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [
            k for k in state_dict.keys() if 'relative_position_index' in k
        ]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict[
            'patch_embed.proj.weight'].unsqueeze(2).repeat(
                1, 1, self.patch_size[0], 1, 1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [
            k for k in state_dict.keys() if 'relative_position_bias_table' in k
        ]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            # wd = self.window_size[0]
            # to make it match
            wd = 16
            if nH1 != nH2:
                print(f'Error in loading {k}, passing')
            else:
                if L1 != L2:
                    S1 = int(L1**0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(
                            1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1,
                              2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(
                2 * wd - 1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
                self.inflate_weights()
            else:
                # Directly load 3D model.
                torch.load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer2D_TPS, self).train(mode)
        self._freeze_stages()


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score from mmaction.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head from mmaction.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'top1_acc', 'top5_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(), (1, 5))
            losses['top1_acc'] = torch.tensor(
                top_k_acc[0], device=cls_score.device)
            losses['top5_acc'] = torch.tensor(
                top_k_acc[1], device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels
                      + self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses


class I3DHead(BaseHead):
    """Classification head for I3D from mmaction.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


class PatchShiftTransformer(TorchModel):
    """  This is PyTorch impl of PST:
    Spatiotemporal Self-attention Modeling with Temporal Patch Shift for Action Recognition, ECCV22.
    """

    def __init__(self,
                 model_dir=None,
                 num_classes=400,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 embed_dim=96,
                 in_channels=768,
                 pretrained=None):
        super().__init__(model_dir)
        self.backbone = SwinTransformer2D_TPS(
            pretrained=pretrained,
            pretrained2d=True,
            patch_size=(2, 4, 4),
            in_chans=3,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=(1, 7, 7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            frozen_stages=-1,
            use_checkpoint=False)
        self.cls_head = I3DHead(
            num_classes=num_classes, in_channels=in_channels)

    def forward(self, x):
        feature = self.backbone(x)
        output = self.cls_head(feature)
        return output
