# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
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
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io

import math

import matplotlib
import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize

# Search table for suggested max. inference batch size
bs_search_table = [
    # tested on A100-PCIE-80GB
    {
        'res': 768,
        'total_vram': 79,
        'bs': 35,
        'dtype': torch.float32
    },
    {
        'res': 1024,
        'total_vram': 79,
        'bs': 20,
        'dtype': torch.float32
    },
    # tested on A100-PCIE-40GB
    {
        'res': 768,
        'total_vram': 39,
        'bs': 15,
        'dtype': torch.float32
    },
    {
        'res': 1024,
        'total_vram': 39,
        'bs': 8,
        'dtype': torch.float32
    },
    {
        'res': 768,
        'total_vram': 39,
        'bs': 30,
        'dtype': torch.float16
    },
    {
        'res': 1024,
        'total_vram': 39,
        'bs': 15,
        'dtype': torch.float16
    },
    # tested on RTX3090, RTX4090
    {
        'res': 512,
        'total_vram': 23,
        'bs': 20,
        'dtype': torch.float32
    },
    {
        'res': 768,
        'total_vram': 23,
        'bs': 7,
        'dtype': torch.float32
    },
    {
        'res': 1024,
        'total_vram': 23,
        'bs': 3,
        'dtype': torch.float32
    },
    {
        'res': 512,
        'total_vram': 23,
        'bs': 40,
        'dtype': torch.float16
    },
    {
        'res': 768,
        'total_vram': 23,
        'bs': 18,
        'dtype': torch.float16
    },
    {
        'res': 1024,
        'total_vram': 23,
        'bs': 10,
        'dtype': torch.float16
    },
    # tested on GTX1080Ti
    {
        'res': 512,
        'total_vram': 10,
        'bs': 5,
        'dtype': torch.float32
    },
    {
        'res': 768,
        'total_vram': 10,
        'bs': 2,
        'dtype': torch.float32
    },
    {
        'res': 512,
        'total_vram': 10,
        'bs': 10,
        'dtype': torch.float16
    },
    {
        'res': 768,
        'total_vram': 10,
        'bs': 5,
        'dtype': torch.float16
    },
    {
        'res': 1024,
        'total_vram': 10,
        'bs': 3,
        'dtype': torch.float16
    },
]


def find_batch_size(ensemble_size: int, input_res: int,
                    dtype: torch.dtype) -> int:
    """
    Automatically search for suitable operating batch size.

    Args:
        ensemble_size (`int`):
            Number of predictions to be ensembled.
        input_res (`int`):
            Operating resolution of the input image.

    Returns:
        `int`: Operating batch size.
    """
    if not torch.cuda.is_available():
        return 1

    total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
    filtered_bs_search_table = [
        s for s in bs_search_table if s['dtype'] == dtype
    ]
    for settings in sorted(
            filtered_bs_search_table,
            key=lambda k: (k['res'], -k['total_vram']),
    ):
        if input_res <= settings['res'] and total_vram >= settings[
                'total_vram']:
            bs = settings['bs']
            if bs > ensemble_size:
                bs = ensemble_size
            elif bs > math.ceil(ensemble_size / 2) and bs < ensemble_size:
                bs = math.ceil(ensemble_size / 2)
            return bs

    return 1


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i:i + 1]
        arr2 = tensors[j:j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist


def ensemble_depths(
    input_images: torch.Tensor,
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    reduction: str = 'median',
    max_res: int = None,
):
    """
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the scale and shift
    """
    device = input_images.device
    dtype = input_images.dtype
    np_dtype = np.float32

    original_input = input_images.clone()
    n_img = input_images.shape[0]
    ori_shape = input_images.shape

    if max_res is not None:
        scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(
                scale_factor=scale_factor, mode='nearest')
            input_images = downscaler(torch.from_numpy(input_images)).numpy()

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype)

    input_images = input_images.to(device)

    # objective function
    def closure(x):
        length = len(x)
        s = x[:int(length / 2)]
        t = x[int(length / 2):]
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)

        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view(
            (-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = torch.sqrt(torch.mean(dists**2))

        if 'mean' == reduction:
            pred = torch.mean(transformed_arrays, dim=0)
        elif 'median' == reduction:
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError

        near_err = torch.sqrt((0 - torch.min(pred))**2)
        far_err = torch.sqrt((1 - torch.max(pred))**2)

        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy().astype(np_dtype)
        return err

    res = minimize(
        closure,
        x,
        method='BFGS',
        tol=tol,
        options={
            'maxiter': max_iter,
            'disp': False
        })
    x = res.x
    length = len(x)
    s = x[:int(length / 2)]
    t = x[int(length / 2):]

    # Prediction
    s = torch.from_numpy(s).to(dtype=dtype).to(device)
    t = torch.from_numpy(t).to(dtype=dtype).to(device)
    transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1)
    if 'mean' == reduction:
        aligned_images = torch.mean(transformed_arrays, dim=0)
        std = torch.std(transformed_arrays, dim=0)
        uncertainty = std
    elif 'median' == reduction:
        aligned_images = torch.median(transformed_arrays, dim=0).values
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = torch.abs(transformed_arrays - aligned_images)
        mad = torch.median(abs_dev, dim=0).values
        uncertainty = mad
    else:
        raise ValueError(f'Unknown reduction method: {reduction}')

    # Scale and shift to [0, 1]
    _min = torch.min(aligned_images)
    _max = torch.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= _max - _min

    return aligned_images, uncertainty


def colorize_depth_maps(depth_map,
                        min_depth,
                        max_depth,
                        cmap='Spectral',
                        valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, 'Invalid dimension'

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).

    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min(max_edge_resolution / original_width,
                           max_edge_resolution / original_height)

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = img.resize((new_width, new_height))
    return resized_img
