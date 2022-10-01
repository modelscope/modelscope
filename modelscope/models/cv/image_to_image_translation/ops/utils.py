# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import base64
import binascii
import hashlib
import math
import os
import os.path as osp
import zipfile
from io import BytesIO
from multiprocessing.pool import ThreadPool as Pool

import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .random_color import rand_color

__all__ = [
    'ceil_divide', 'to_device', 'rand_name', 'ema', 'parallel', 'unzip',
    'load_state_dict', 'inverse_indices', 'detect_duplicates', 'md5', 'rope',
    'format_state', 'breakup_grid', 'viz_anno_geometry', 'image_to_base64'
]

TFS_CLIENT = None


def ceil_divide(a, b):
    return int(math.ceil(a / b))


def to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([to_device(u, device, non_blocking) for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([(k, to_device(v, device, non_blocking))
                            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    return batch


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


@torch.no_grad()
def ema(net_ema, net, beta, copy_buffer=False):
    assert 0.0 <= beta <= 1.0
    for p_ema, p in zip(net_ema.parameters(), net.parameters()):
        p_ema.copy_(p.lerp(p_ema, beta))
    if copy_buffer:
        for b_ema, b in zip(net_ema.buffers(), net.buffers()):
            b_ema.copy_(b)


def parallel(func, args_list, num_workers=32, timeout=None):
    assert isinstance(args_list, list)
    if not isinstance(args_list[0], tuple):
        args_list = [(args, ) for args in args_list]
    if num_workers == 0:
        return [func(*args) for args in args_list]
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(func, args) for args in args_list]
        results = [res.get(timeout=timeout) for res in results]
    return results


def unzip(filename, dst_dir=None):
    if dst_dir is None:
        dst_dir = osp.dirname(filename)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dst_dir)


def load_state_dict(module, state_dict, drop_prefix=''):
    # find incompatible key-vals
    src, dst = state_dict, module.state_dict()
    if drop_prefix:
        src = type(src)([
            (k[len(drop_prefix):] if k.startswith(drop_prefix) else k, v)
            for k, v in src.items()
        ])
    missing = [k for k in dst if k not in src]
    unexpected = [k for k in src if k not in dst]
    unmatched = [
        k for k in src.keys() & dst.keys() if src[k].shape != dst[k].shape
    ]

    # keep only compatible key-vals
    incompatible = set(unexpected + unmatched)
    src = type(src)([(k, v) for k, v in src.items() if k not in incompatible])
    module.load_state_dict(src, strict=False)

    # report incompatible key-vals
    if len(missing) != 0:
        print('  Missing: ' + ', '.join(missing), flush=True)
    if len(unexpected) != 0:
        print('  Unexpected: ' + ', '.join(unexpected), flush=True)
    if len(unmatched) != 0:
        print('  Shape unmatched: ' + ', '.join(unmatched), flush=True)


def inverse_indices(indices):
    r"""Inverse map of indices.
        E.g., if A[indices] == B, then B[inv_indices] == A.
    """
    inv_indices = torch.empty_like(indices)
    inv_indices[indices] = torch.arange(len(indices)).to(indices)
    return inv_indices


def detect_duplicates(feats, thr=0.9):
    assert feats.ndim == 2

    # compute simmat
    feats = F.normalize(feats, p=2, dim=1)
    simmat = torch.mm(feats, feats.T)
    simmat.triu_(1)
    torch.cuda.synchronize()

    # detect duplicates
    mask = ~simmat.gt(thr).any(dim=0)
    return torch.where(mask)[0]


def md5(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def rope(x):
    r"""Apply rotary position embedding on x of shape [B, *(spatial dimensions), C].
    """
    # reshape
    shape = x.shape
    x = x.view(x.size(0), -1, x.size(-1))
    l, c = x.shape[-2:]
    assert c % 2 == 0
    half = c // 2

    # apply rotary position embedding on x
    sinusoid = torch.outer(
        torch.arange(l).to(x),
        torch.pow(10000, -torch.arange(half).to(x).div(half)))
    sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
    x1, x2 = x.chunk(2, dim=-1)
    x = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    # reshape back
    return x.view(shape)


def format_state(state, filename=None):
    r"""For comparing/aligning state_dict.
    """
    content = '\n'.join([f'{k}\t{tuple(v.shape)}' for k, v in state.items()])
    if filename:
        with open(filename, 'w') as f:
            f.write(content)


def breakup_grid(img, grid_size):
    r"""The inverse operator of ``torchvision.utils.make_grid``.
    """
    # params
    nrow = img.height // grid_size
    ncol = img.width // grid_size
    wrow = wcol = 2  # NOTE: use default values here

    # collect grids
    grids = []
    for i in range(nrow):
        for j in range(ncol):
            x1 = j * grid_size + (j + 1) * wcol
            y1 = i * grid_size + (i + 1) * wrow
            grids.append(img.crop((x1, y1, x1 + grid_size, y1 + grid_size)))
    return grids


def viz_anno_geometry(item):
    r"""Visualize an annotation item from SmartLabel.
    """
    if isinstance(item, str):
        item = json.loads(item)
    assert isinstance(item, dict)

    # read image
    orig_img = read_image(item['image_url'], retry=100)
    img = cv2.cvtColor(np.asarray(orig_img), cv2.COLOR_BGR2RGB)

    # loop over geometries
    for geometry in item['sd_result']['items']:
        # params
        poly_img = img.copy()
        color = rand_color()
        points = np.array(geometry['meta']['geometry']).round().astype(int)
        line_color = tuple([int(u * 0.55) for u in color])

        # draw polygons
        poly_img = cv2.fillPoly(poly_img, pts=[points], color=color)
        poly_img = cv2.polylines(
            poly_img,
            pts=[points],
            isClosed=True,
            color=line_color,
            thickness=2)

        # mixing
        img = np.clip(0.25 * img + 0.75 * poly_img, 0, 255).astype(np.uint8)
    return orig_img, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def image_to_base64(img, format='JPEG'):
    buffer = BytesIO()
    img.save(buffer, format=format)
    code = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return code
