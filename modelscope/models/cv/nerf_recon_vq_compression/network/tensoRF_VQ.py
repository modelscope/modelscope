import os
import random
from typing import Callable, Iterator, List, Optional, Union

import torch.nn as nn
from tqdm import tqdm

from .tensorBase import *
from .tensoRF import TensorVMSplit
from .weighted_vq import VectorQuantize


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name, debug=False):
        self.name = name
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return

        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        if not self.debug:
            return

        self.end.record()
        torch.cuda.synchronize()
        print(self.name, 'elapsed', self.start.elapsed_time(self.end), 'ms')


def dec2bin(x, bits):
    mask = 2**torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2**torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


class TensorVMSplitVQ(TensorVMSplit):

    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplitVQ, self).__init__(aabb, gridSize, device, **kargs)
        self.codebook_size = kargs['codebook_size']
        print('codebook size: ' + str(self.codebook_size))
        self.use_cosine_sim = kargs['use_cosine_sim'] == 1
        self.codebook_dim = None if kargs['codebook_dim'] == 0 else kargs[
            'codebook_dim']
        self.vq = nn.ModuleList([
            VectorQuantize(
                dim=self.app_n_comp[0],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.app_n_comp[1],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.app_n_comp[2],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device)
        ])
        self.den_vq = nn.ModuleList([
            VectorQuantize(
                dim=self.density_n_comp[0],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.density_n_comp[1],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device),
            VectorQuantize(
                dim=self.density_n_comp[2],
                codebook_size=self.codebook_size,  # codebook size
                decay=0.8,  # specify number of quantizer
                commitment_weight=1.0,
                use_cosine_sim=self.use_cosine_sim,
                codebook_dim=self.codebook_dim,
                threshold_ema_dead_code=2.0,
            ).to(self.device)
        ])
        self.importance = kargs.get('importance', None)
        self.plane_mask = kargs.get('plane_mask', None)
        self.all_indices = kargs.get('all_indices', None)

    def extreme_load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(
                np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(
                    ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(
                self.device, ckpt['alphaMask.aabb'].to(self.device),
                alpha_volume.float().to(self.device))

        # 1. load non-vq part
        self.density_line.load_state_dict(ckpt['density_line'])
        self.app_line.load_state_dict(ckpt['app_line'])
        self.basis_mat.load_state_dict(ckpt['basis_mat'])
        self.renderModule.load_state_dict(ckpt['mlp'])

        # 2. load vq part
        # load vq_mask, keep_mask
        self.plane_mask = []
        for i in range(3):
            mask_shape = self.app_plane[i].shape[-2:]
            vq_mask = np.unpackbits(
                ckpt[f'vq_mask_{i}'],
                count=np.prod(mask_shape)).reshape(mask_shape).astype(bool)
            keep_mask = np.unpackbits(
                ckpt[f'keep_mask_{i}'],
                count=np.prod(mask_shape)).reshape(mask_shape).astype(bool)
            self.plane_mask.append((vq_mask, keep_mask))

        # recover app_plane, density_plane
        import math
        bits = int(math.log2(self.codebook_size))
        for idx_plane in range(3):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            # load appearance keep data from quantized data
            int_repr = ckpt[f'quant_keep_data_{idx_plane}.int_repr']
            scale = ckpt[f'quant_keep_data_{idx_plane}.scale']
            zero_points = ckpt[f'quant_keep_data_{idx_plane}.zero_points']
            dequant = (int_repr - zero_points) * scale
            keep_data = dequant.T.reshape(
                *self.app_plane[idx_plane][:, :, keep_mask].shape)
            self.app_plane[idx_plane].data[:, :, keep_mask] = keep_data

            # load appearance vq data from codebook
            codebook = ckpt[f'codebook_{idx_plane}'].float()  #
            vq_count = int(vq_mask.sum())
            unpack1 = np.unpackbits(
                ckpt[f'vq_indice_{idx_plane}'], count=vq_count * bits)
            unpack2 = bin2dec(
                torch.from_numpy(unpack1).reshape(vq_count, bits).long(),
                bits=bits)
            vq_data = codebook[0, unpack2, :]  # N*len
            vq_data = vq_data.T.reshape(
                *(self.app_plane[idx_plane][:, :, vq_mask].shape))
            self.app_plane[idx_plane].data[:, :, vq_mask] = vq_data

        for idx_plane in range(3):
            (vq_mask, keep_mask) = self.plane_mask[idx_plane]
            # load density keep data from quantized data
            int_repr = ckpt[f'quant_den_data_{idx_plane}.int_repr']
            scale = ckpt[f'quant_den_data_{idx_plane}.scale']
            zero_points = ckpt[f'quant_den_data_{idx_plane}.zero_points']
            dequant = (int_repr - zero_points) * scale
            keep_data = dequant.T.reshape(
                *self.density_plane[idx_plane][:, :, keep_mask].shape)
            self.density_plane[idx_plane].data[:, :, keep_mask] = keep_data

            # load density vq data from codebook
            codebook = ckpt[f'codebook_den_{idx_plane}'].float()  #
            vq_count = int(vq_mask.sum())
            unpack1 = np.unpackbits(
                ckpt[f'den_vq_indice_{idx_plane}'], count=vq_count * bits)
            unpack2 = bin2dec(
                torch.from_numpy(unpack1).reshape(vq_count, bits).long(),
                bits=bits)
            vq_data = codebook[0, unpack2, :]  # N*len
            vq_data = vq_data.T.reshape(
                *(self.density_plane[idx_plane][:, :, vq_mask].shape))
            self.density_plane[idx_plane].data[:, :, vq_mask] = vq_data

    def forward(self,
                rays_chunk,
                white_bg=True,
                is_train=False,
                ndc_ray=False,
                N_samples=-1,
                isvq=False):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3],
                viewdirs,
                is_train=is_train,
                N_samples=N_samples)
            dists = torch.cat(
                (z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(
                    z_vals[:, :1])),
                dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3),
                          device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma,
                                             dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask],
                                           viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1, )) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map


def getsize(compressed_file, tag='MB'):
    size = os.path.getsize(compressed_file)
    if tag == 'B':
        pass
    elif tag == 'KB':
        size = size / 1024
    elif tag == 'MB':
        size = size / 1024 / 1024
    elif tag == 'GB':
        size = size / 1024 / 1024 / 1024
    return f'{size} {tag}'
