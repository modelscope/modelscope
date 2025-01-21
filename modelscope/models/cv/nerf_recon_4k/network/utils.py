# The implementation is partly adopted from nerfacc, made publicly available under the MIT License
# at https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/ngp.py
import gc
from collections import defaultdict

import mcubes
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class PSNR(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(
                torch.mean(value, dim=tuple(range(value.ndim)[1:])))


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    xx = xx.reshape(-1, 1)
                    yy = yy.reshape(-1, 1)
                    zz = zz.reshape(-1, 1)
                    pts = torch.cat([xx, yy, zz], dim=-1).cuda()
                    val = query_func(pts).reshape(
                        len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys),
                      zi * N:zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (
        b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def chunk_batch(func, chunk_size, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(
            *[
                arg[i:i + chunk_size] if isinstance(arg, torch.Tensor) else arg
                for arg in args
            ], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            exit(1)
        for k, v in out_chunk.items():
            out[k].append(v if torch.is_grad_enabled() else v.detach())

    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


def get_activation(name):
    name = name.lower()
    if name is None or name == 'none':
        return nn.Identity()
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name.lower() == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name.lower() == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def normalize(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


def update_module_step(m, epoch, global_step):
    if hasattr(m, 'update_step'):
        m.update_step(epoch, global_step)
