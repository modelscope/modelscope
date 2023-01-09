# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


def init_func(m, init_type='xavier', gain=0.02):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                   or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight, gain=gain)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif hasattr(m, 'weight_bar') and (classname.find('Conv') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight_bar, 0.0, gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight_bar, gain=gain)
        elif init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight_bar, gain=1.0)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight_bar, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight_bar, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


def convert_affinematrix_to_homography(A):
    H = torch.nn.functional.pad(A, [0, 0, 0, 1], 'constant', value=0.0)
    H[..., -1, -1] += 1.0
    return H


def normal_transform_pixel(height, width, eps=1e-14):
    tr_mat = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0,
                                                                1.0]])  # 3x3

    # prevent divide by zero bugs
    width_denom = eps if width == 1 else width - 1.0
    height_denom = eps if height == 1 else height - 1.0

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / width_denom
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / height_denom

    return tr_mat.unsqueeze(0)  # 1x3x3


def _torch_inverse_cast(input):
    if not isinstance(input, torch.Tensor):
        raise AssertionError(
            f'Input must be torch.Tensor. Got: {type(input)}.')
    dtype = input.dtype
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float32
    return torch.inverse(input.to(dtype)).to(input.dtype)


def normalize_homography(dst_pix_trans_src_pix, dsize_src, dsize_dst):
    if not isinstance(dst_pix_trans_src_pix, torch.Tensor):
        raise TypeError(
            f'Input type is not a torch.Tensor. Got {type(dst_pix_trans_src_pix)}'
        )

    if not (len(dst_pix_trans_src_pix.shape) == 3
            or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError(
            f'Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {dst_pix_trans_src_pix.shape}'
        )

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: torch.Tensor = normal_transform_pixel(
        src_h, src_w).to(dst_pix_trans_src_pix)

    src_pix_trans_src_norm = _torch_inverse_cast(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix = normal_transform_pixel(
        dst_h, dst_w).to(dst_pix_trans_src_pix)

    # compute chain transformations
    dst_norm_trans_src_norm = dst_norm_trans_dst_pix @ (
        dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    return dst_norm_trans_src_norm


def warp_affine_torch(src,
                      M,
                      dsize,
                      mode='bilinear',
                      padding_mode='zeros',
                      align_corners=True):

    if not isinstance(src, torch.Tensor):
        raise TypeError(
            f'Input src type is not a torch.Tensor. Got {type(src)}')

    if not isinstance(M, torch.Tensor):
        raise TypeError(f'Input M type is not a torch.Tensor. Got {type(M)}')

    if not len(src.shape) == 4:
        raise ValueError(
            f'Input src must be a BxCxHxW tensor. Got {src.shape}')

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError(f'Input M must be a Bx2x3 tensor. Got {M.shape}')

    B, C, H, W = src.size()

    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3 = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm = normalize_homography(M_3x3, (H, W), dsize)
    src_norm_trans_dst_norm = _torch_inverse_cast(dst_norm_trans_src_norm)
    grid = F.affine_grid(
        src_norm_trans_dst_norm[:, :2, :], [B, C, dsize[0], dsize[1]],
        align_corners=align_corners)
    return F.grid_sample(
        src,
        grid,
        align_corners=align_corners,
        mode=mode,
        padding_mode=padding_mode)
