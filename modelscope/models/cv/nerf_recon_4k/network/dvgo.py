import functools
import math
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo

parent_dir = os.path.dirname(os.path.abspath(__file__))
parent_list = parent_dir.split('/')
del parent_list[-4:]
parent_dir = '/'.join(parent_list)
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path) for path in
        ['ops/4knerf/render_utils.cpp', 'ops/4knerf/render_utils_kernel.cu']
    ],
    verbose=True)


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    else:
        raise NotImplementedError


class DenseGrid(nn.Module):

    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(
            (-1, )) * 2 - 1
        out = F.grid_sample(
            self.grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = nn.Parameter(
                torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(
                    self.grid.data,
                    size=tuple(new_world_size),
                    mode='trilinear',
                    align_corners=True))

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Mask grid
It supports query for the known free space and unknown space.
'''


class MaskGrid(nn.Module):

    def __init__(self,
                 path=None,
                 mask_cache_thres=None,
                 mask=None,
                 xyz_min=None,
                 xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(
                st['model_state_dict']['density.grid'],
                kernel_size=3,
                padding=1,
                stride=1)
            alpha = 1 - torch.exp(
                -F.softplus(density + st['model_state_dict']['act_shift'])
                * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale',
                             (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz,
                                                  self.xyz2ijk_scale,
                                                  self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return 'mask.shape=list(self.mask.shape)'


'''Model'''


class DirectVoxGO(torch.nn.Module):

    def __init__(self,
                 xyz_min,
                 xyz_max,
                 num_voxels=0,
                 num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None,
                 mask_cache_thres=1e-3,
                 mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid',
                 k0_type='DenseGrid',
                 density_config={},
                 k0_config={},
                 rgbnet_dim=0,
                 rgbnet_direct=False,
                 rgbnet_full_implicit=False,
                 rgbnet_depth=3,
                 rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        self.num_voxels_base = num_voxels_base
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod()
                                / self.num_voxels_base).pow(1 / 3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer(
            'act_shift', torch.FloatTensor([np.log(1 / (1 - alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = create_grid(
            density_type,
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth,
            'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        self.dim_rend = 3  # kwargs['dim_rend']
        self.act_type = 'mlp'  # kwargs['act_type']
        self.mode_type = 'mlp'

        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct
            self.register_buffer(
                'viewfreq',
                torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3 + 3 * viewbase_pe * 2)

            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim - 3

            self.dim0 = dim0
            if self.dim_rend > 3:
                self.rgbnet = nn.Sequential(
                    nn.Linear(self.dim0, rgbnet_width),
                    nn.LeakyReLU(),
                    nn.Linear(rgbnet_width, self.dim_rend),
                    nn.LeakyReLU(),
                )
                self.rend_layer = nn.Sequential(nn.Linear(self.dim_rend, 3), )
                nn.init.constant_(self.rend_layer[-1].bias, 0)
            else:
                self.rgbnet = nn.Sequential(
                    nn.Linear(dim0, rgbnet_width),
                    nn.ReLU(inplace=True),
                    *[
                        nn.Sequential(
                            nn.Linear(rgbnet_width, rgbnet_width),
                            nn.ReLU(inplace=True))
                        for _ in range(rgbnet_depth - 2)
                    ],
                    nn.Linear(rgbnet_width, 3),
                )
                nn.init.constant_(self.rgbnet[-1].bias, 0)

            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskGrid(
                path=mask_cache_path,
                mask_cache_thres=mask_cache_thres).to('cuda')
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0],
                                   mask_cache_world_size[0]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1],
                                   mask_cache_world_size[1]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2],
                                   mask_cache_world_size[2]),
                ), -1).to('cuda')
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = MaskGrid(
            path=None, mask=mask, xyz_min=self.xyz_min,
            xyz_max=self.xyz_max).to(self.xyz_min.device)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod()
                           / num_voxels).pow(1 / 3)
        self.world_size = ((self.xyz_max - self.xyz_min)
                           / self.voxel_size).long()
        self.max_world_size = self.world_size.max()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'mode_type': self.mode_type,
            'act_type': self.act_type,
            'dim_rend': self.dim_rend,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0],
                               self.world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1],
                               self.world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2],
                               self.world_size[2]),
            ), -1)
        nearest_dist = torch.stack([
            (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in cam_o.split(100)  # for memory saving
        ]).amin(0)
        self.density.grid[nearest_dist[None, None] <= near_clip] = -100

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from',
              ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0],
                                   self.world_size[0]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1],
                                   self.world_size[1]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2],
                                   self.world_size[2]),
                ), -1)
            self_alpha = F.max_pool3d(
                self.activate_density(self.density.get_dense_grid()),
                kernel_size=3,
                padding=1,
                stride=1)[0, 0]
            self.mask_cache = MaskGrid(
                path=None,
                mask=self.mask_cache(self_grid_xyz)
                & (self_alpha > self.fast_color_thres),
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0],
                               self.mask_cache.mask.shape[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1],
                               self.mask_cache.mask.shape[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2],
                               self.mask_cache.mask.shape[2]),
            ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(
            cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def voxel_count_views(self,
                          rays_o_tr,
                          rays_d_tr,
                          imsz,
                          near,
                          far,
                          stepsize,
                          downrate=1,
                          irregular_shape=False):
        print('dvgo: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()
        N_samples = int(
            np.linalg.norm(np.array(self.world_size.cpu()) + 1) / stepsize) + 1
        rng = torch.arange(N_samples)[None].float()
        count = torch.zeros_like(self.density.get_dense_grid())
        device = rng.device
        for rays_o_, rays_d_ in zip(
                rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(
                    0, -2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(
                    0, -2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6),
                                  rays_d)
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(
                    min=near, max=far)
                # t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(
                #     min=near, max=far)
                step = stepsize * self.voxel_size * rng
                interpx = (
                    t_min[..., None]
                    + step / rays_d.norm(dim=-1, keepdim=True))
                rays_pts = rays_o[
                    ..., None, :] + rays_d[..., None, :] * interpx[..., None]
                ones(rays_pts).sum().backward()
            with torch.no_grad():
                count += (ones.grid.grad > 1)
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift,
                               interval).reshape(shape)

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize,
                       **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        rays_o = rays_o.reshape(-1, 3).contiguous()
        rays_d = rays_d.reshape(-1, 3).contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far,
            stepdist)[:3]
        mask_inbbox = ~mask_outbbox
        hit = torch.zeros([len(rays_o)], dtype=torch.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        N_samples = int((self.max_world_size - 1) / stepsize) + 1

        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox

        # mask_ori = torch.zeros((len(N_steps), N_steps.max())).bool()
        # for idx, item in enumerate(N_steps):
        #     mask_ori[idx, :item] = mask_inbbox[ray_id == idx]
        mask_ori = None

        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id, mask_ori, N_samples

    def forward(self,
                rays_o,
                rays_d,
                viewdirs,
                global_step=None,
                **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape) == 2 and rays_o.shape[
            -1] == 3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, mask_inbbox, N_samples = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask1 = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask1]
            ray_id = ray_id[mask1]
            step_id = step_id[mask1]

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask2 = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask2]
            ray_id = ray_id[mask2]
            step_id = step_id[mask2]
            density = density[mask2]
            alpha = alpha[mask2]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask3 = (weights > self.fast_color_thres)
            weights = weights[mask3]
            alpha = alpha[mask3]
            ray_pts = ray_pts[mask3]
            ray_id = ray_id[mask3]
            step_id = step_id[mask3]

        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            k0 = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb_raw = torch.sigmoid(k0)
        else:
            # view-dependent color emission
            if self.rgbnet_direct:
                k0_view = k0
            else:
                k0_view = k0[:, 3:]
                k0_diffuse = k0[:, :3]
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat(
                [viewdirs, viewdirs_emb.sin(),
                 viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0, -2)[ray_id]
            rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)

            if self.mode_type == 'TRANS':
                rgb_feat_pre3 = torch.zeros((mask3.shape[0], self.dim0))
                rgb_feat_pre3[mask3] = rgb_feat
                rgb_feat_pre2 = torch.zeros((mask2.shape[0], self.dim0))
                rgb_feat_pre2[mask2] = rgb_feat_pre3
                rgb_feat_pre1 = torch.zeros((mask1.shape[0], self.dim0))
                rgb_feat_pre1[mask1] = rgb_feat_pre2
                rgb_feat_ori = torch.zeros(
                    (mask_inbbox.shape[0], mask_inbbox.shape[1], self.dim0))
                rgb_feat_ori[mask_inbbox] = rgb_feat_pre1
                # rgb_logit_tran = self.reformer(rgb_feat_ori)
                rgb_logit = self.trans_nn(rgb_feat_ori)
                rgb_logit = rgb_logit[mask_inbbox.view(
                    -1)][mask1][mask2][mask3]
            elif self.mode_type == 'adain':
                rgb_logit = self.adainet(rgb_feat, rgb_feat)
            else:
                rgb_logit = self.rgbnet(rgb_feat)

            if self.rgbnet_direct:
                rgb_raw = torch.sigmoid(rgb_logit)
            else:
                rgb_raw = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        rgb_feature = segment_coo(
            src=(weights.unsqueeze(-1) * rgb_raw),
            index=ray_id,
            out=torch.zeros([N, self.dim_rend], device='cuda'),
            reduce='sum')
        if self.dim_rend > 3:
            rgb_raw = torch.sigmoid(self.rend_layer(rgb_raw))
            rgb_marched = self.rend_layer(rgb_feature)
            # rgb_marched = torch.sigmoid(rgb_marched)
        else:
            rgb_marched = rgb_feature

        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id + 0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'rgb_feature': rgb_feature,
            'raw_alpha': alpha,
            'raw_rgb': rgb_raw,
            'ray_id': ray_id,
        })

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * s),  # step_id
                    index=ray_id,
                    out=torch.zeros([N], device='cuda'),
                    reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


class DirectMPIGO(torch.nn.Module):

    def __init__(self,
                 xyz_min,
                 xyz_max,
                 num_voxels=0,
                 mpi_depth=0,
                 mask_cache_path=None,
                 mask_cache_thres=1e-3,
                 mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid',
                 k0_type='DenseGrid',
                 density_config={},
                 k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3,
                 rgbnet_width=128,
                 viewbase_pe=0,
                 spatial_pe=0,
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = create_grid(
            density_type,
            channels=1,
            world_size=self.world_size,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            config=self.density_config)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = DenseGrid(
            channels=1,
            world_size=[1, 1, mpi_depth],
            xyz_min=xyz_min,
            xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1. / mpi_depth - 1e-6)
            p = [1 - g[0]]
            for i in range(1, len(g)):
                p.append((1 - g[:i + 1].sum()) / (1 - g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(
                    np.log(p[i]**(-1 / self.voxel_size_ratio) - 1))

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth,
            'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
            'spatial_pe': spatial_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = create_grid(
                k0_type,
                channels=self.k0_dim,
                world_size=self.world_size,
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max,
                config=self.k0_config)
            self.register_buffer(
                'viewfreq',
                torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            self.register_buffer(
                'posfreq',
                torch.FloatTensor([(2**i) for i in range(spatial_pe)]))
            self.dim0 = (3 + 3 * viewbase_pe * 2 + 3
                         + 3 * spatial_pe * 2) + self.k0_dim
            self.pe_dim = 3 + 3 * viewbase_pe * 2 + 3 + 3 * spatial_pe * 2

            self.dim_rend = 3  # kwargs['dim_rend']
            self.act_type = kwargs['act_type']
            act = nn.ReLU(inplace=True)

            if self.dim_rend > 3:
                self.rgbnet = nn.Sequential(
                    nn.Linear(self.dim0, rgbnet_width),
                    nn.LeakyReLU(),
                    nn.Linear(rgbnet_width, self.dim_rend),
                    nn.LeakyReLU(),
                )
                self.rend_layer = nn.Sequential(nn.Linear(self.dim_rend, 3), )
                # self.rgbper_layer = nn.Sequential(
                #     nn.Linear(self.dim_rend, 3)
                # )
                nn.init.constant_(self.rend_layer[-1].bias, 0)
            else:
                self.rgbnet = nn.Sequential(
                    nn.Linear(self.dim0, rgbnet_width),
                    act,
                    *[
                        nn.Sequential(
                            nn.Linear(rgbnet_width, rgbnet_width), act)
                        for _ in range(rgbnet_depth - 2)
                    ],
                    nn.Linear(rgbnet_width, self.dim_rend),
                )
                nn.init.constant_(self.rgbnet[-1].bias, 0)

            print('dmpigo: densitye grid', self.density)
            print('dmpigo: feature grid', self.k0)
            self.mode_type = kwargs['mode_type']
            print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = MaskGrid(
                path=mask_cache_path,
                mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0],
                                   mask_cache_world_size[0]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1],
                                   mask_cache_world_size[1]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2],
                                   mask_cache_world_size[2]),
                ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = MaskGrid(
            path=None, mask=mask, xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = num_voxels / self.mpi_depth
        r = (r / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)

    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min.cpu().numpy(),
            'xyz_max': self.xyz_max.cpu().numpy(),
            'num_voxels': self.num_voxels,
            'mpi_depth': self.mpi_depth,
            'voxel_size_ratio': self.voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            'mode_type': self.mode_type,
            'act_type': self.act_type,
            'dim_rend': self.dim_rend,
            **self.rgbnet_kwargs,
        }

    @torch.no_grad()
    def scale_volume_grid(self, num_voxels, mpi_depth):
        print('dmpigo: scale_volume_grid start')
        ori_world_size = self.world_size
        self._set_grid_resolution(num_voxels, mpi_depth)
        print('dmpigo: scale_volume_grid scale world_size from',
              ori_world_size.tolist(), 'to', self.world_size.tolist())

        self.density.scale_volume_grid(self.world_size)
        self.k0.scale_volume_grid(self.world_size)

        if np.prod(self.world_size.tolist()) <= 256**3:
            self_grid_xyz = torch.stack(
                torch.meshgrid(
                    torch.linspace(self.xyz_min[0], self.xyz_max[0],
                                   self.world_size[0]),
                    torch.linspace(self.xyz_min[1], self.xyz_max[1],
                                   self.world_size[1]),
                    torch.linspace(self.xyz_min[2], self.xyz_max[2],
                                   self.world_size[2]),
                ), -1)
            dens = self.density.get_dense_grid() + self.act_shift.grid
            self_alpha = F.max_pool3d(
                self.activate_density(dens),
                kernel_size=3,
                padding=1,
                stride=1)[0, 0]
            self.mask_cache = MaskGrid(
                path=None,
                mask=self.mask_cache(self_grid_xyz)
                & (self_alpha > self.fast_color_thres),
                xyz_min=self.xyz_min,
                xyz_max=self.xyz_max)

        print('dmpigo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        ori_p = self.mask_cache.mask.float().mean().item()
        cache_grid_xyz = torch.stack(
            torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0],
                               self.mask_cache.mask.shape[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1],
                               self.mask_cache.mask.shape[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2],
                               self.mask_cache.mask.shape[2]),
            ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None, None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(
            cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0, 0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')

    def update_occupancy_cache_lt_nviews(self, rays_o_tr, rays_d_tr, imsz,
                                         render_kwargs, maskout_lt_nviews):
        print('dmpigo: update mask_cache lt_nviews start')
        eps_time = time.time()
        count = torch.zeros_like(self.density.get_dense_grid()).long()
        device = count.device
        for rays_o_, rays_d_ in zip(
                rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = DenseGrid(1, self.world_size, self.xyz_min, self.xyz_max)
            for rays_o, rays_d in zip(
                    rays_o_.split(8192), rays_d_.split(8192)):
                ray_pts, ray_id, step_id, N_samples, _ = self.sample_ray(
                    rays_o=rays_o.to(device),
                    rays_d=rays_d.to(device),
                    **render_kwargs)
                ones(ray_pts).sum().backward()
            count.data += (ones.grid.grad > 1)
        ori_p = self.mask_cache.mask.float().mean().item()
        self.mask_cache.mask &= (count >= maskout_lt_nviews)[0, 0]
        new_p = self.mask_cache.mask.float().mean().item()
        print(f'dmpigo: update mask_cache {ori_p:.4f} => {new_p:.4f}')
        torch.cuda.empty_cache()
        eps_time = time.time() - eps_time
        print('dmpigo: update mask_cache lt_nviews finish (eps time:',
              eps_time, 'sec)')

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), 0, interval).reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        assert near == 0 and far == 1
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        N_samples = int((self.mpi_depth - 1) / stepsize) + 1
        ray_pts, mask_outbbox = render_utils_cuda.sample_ndc_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, N_samples)
        mask_inbbox = ~mask_outbbox

        ray_pts = ray_pts.view(-1, 3)
        ray_pts = ray_pts[mask_inbbox.view(-1)]
        if mask_inbbox.all():
            ray_id, step_id = create_full_step_id(mask_inbbox.shape)
        else:
            ray_id = torch.arange(mask_inbbox.shape[0]).to('cuda').view(
                -1, 1).expand_as(mask_inbbox)[mask_inbbox].to('cuda')
            step_id = torch.arange(mask_inbbox.shape[1]).to('cuda').view(
                1, -1).expand_as(mask_inbbox)[mask_inbbox].to('cuda')
        return ray_pts, ray_id, step_id, N_samples, mask_inbbox

    def forward(self,
                rays_o,
                rays_d,
                viewdirs,
                global_step=None,
                **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape) == 2 and rays_o.shape[
            -1] == 3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}
        N = len(rays_o)

        # sample points on rays
        ray_pts, ray_id, step_id, N_samples, mask_inbbox = self.sample_ray(
            rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        ray_id = ray_id.to('cuda')
        step_id = step_id.to('cuda')
        interval = render_kwargs['stepsize'] * self.voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask1 = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask1]
            ray_id = ray_id[mask1]
            step_id = step_id[mask1]

        # query for alpha w/ post-activation
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask2 = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask2]
            ray_id = ray_id[mask2]
            step_id = step_id[mask2]
            alpha = alpha[mask2]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask3 = (weights > self.fast_color_thres)
            ray_pts = ray_pts[mask3]
            ray_id = ray_id[mask3]
            step_id = step_id[mask3]
            alpha = alpha[mask3]
            weights = weights[mask3]

        # query for color
        vox_emb = self.k0(ray_pts)

        pe_spa = ((ray_pts - self.xyz_min) / (self.xyz_max - self.xyz_min))
        pe_spa = pe_spa.flip((-1, )) * 2 - 1
        # B_gau = torch.normal(mean=0, std=1, size=(128, 3)).to(rays_o.device) * 10
        # pe_emb = input_mapping(pe_spa.detach().clone(), B_gau)

        if self.rgbnet is None:
            # no view-depend effect
            rgb_raw = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat(
                [viewdirs, viewdirs_emb.sin(),
                 viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            pe_emb = (pe_spa.unsqueeze(-1) * self.posfreq).flatten(-2)
            pe_emb = torch.cat([pe_spa, pe_emb.sin(), pe_emb.cos()], -1)

            rgb_feat = torch.cat([vox_emb, pe_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            if self.dim_rend == 3:
                rgb_raw = torch.sigmoid(rgb_logit)
            else:
                rgb_raw = torch.sigmoid(rgb_logit)

        # Ray marching
        rgb_feature = segment_coo(
            src=(weights.unsqueeze(-1) * rgb_raw),
            index=ray_id,
            out=torch.zeros([N, self.dim_rend], device='cuda'),
            reduce='sum')
        if self.dim_rend > 3:
            rgb_raw = torch.sigmoid(self.rend_layer(rgb_raw))
            rgb_marched = self.rend_layer(rgb_feature)
            # rgb_marched = torch.sigmoid(rgb_marched)
        else:
            rgb_marched = rgb_feature

        if render_kwargs.get('rand_bkgd', False) and global_step is not None:
            rgb_marched = rgb_marched + (
                alphainv_last.unsqueeze(-1) * torch.rand_like(rgb_marched))
        else:
            rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        s = (step_id + 0.5) / N_samples
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'rgb_feature': rgb_feature,
            'raw_alpha': alpha,
            'raw_rgb': rgb_raw,
            'ray_id': ray_id,
            'n_max': N_samples,
            's': s,
        })
        # print('alphainv_last shape:', alphainv_last.shape)
        # print('weights shape:', weights.shape)
        # print('raw_alpha shape:', alpha.shape)
        # print('raw_rgb shape:', rgb.shape)
        # print('ray_id shape:', ray_id.shape)
        # print('rgb_marched shape:', rgb_marched.shape)
        # print('s shape:', s.shape, '\n')

        if render_kwargs.get('render_depth', False):
            with torch.no_grad():
                depth = segment_coo(
                    src=(weights * s),
                    index=ray_id,
                    out=torch.zeros([N], device='cuda'),
                    reduce='sum')
            ret_dict.update({'depth': depth})

        return ret_dict


@functools.lru_cache(maxsize=128)
def create_full_step_id(shape):
    ray_id = torch.arange(shape[0]).view(-1, 1).expand(shape).flatten()
    step_id = torch.arange(shape[1]).view(1, -1).expand(shape).flatten()
    return ray_id, step_id


''' Misc
'''


class Raw2Alpha(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_backward(exp,
                                                    grad_back.contiguous(),
                                                    interval), None, None


class Raw2Alpha_nonuni(torch.autograd.Function):

    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(
            density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(
            exp, grad_back.contiguous(), interval), None, None


class Alphas2Weights(torch.autograd.Function):

    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(
            alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start,
                                  i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(alpha, weights, T,
                                                       alphainv_last, i_start,
                                                       i_end, ctx.n_rays,
                                                       grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(
            0, H - 1, H,
            device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1, ))
    if flip_y:
        j = j.flip((0, ))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1],
                            torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1],
                            -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3],
        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy')
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)],
        -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3],
        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (
        rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (
        rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H,
                       W,
                       K,
                       c2w,
                       ndc,
                       inverse_y,
                       flip_x,
                       flip_y,
                       mode='center'):
    rays_o, rays_d = get_rays(
        H,
        W,
        K,
        c2w,
        inverse_y=inverse_y,
        flip_x=flip_x,
        flip_y=flip_y,
        mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x,
                      flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks), -1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(
        rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            ndc=ndc,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y,
                              flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(
        Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            ndc=ndc,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y)
        n = H * W
        rgb_tr[top:top + n].copy_(img.flatten(0, 1))
        rays_o_tr[top:top + n].copy_(rays_o.flatten(0, 1).to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d.flatten(0, 1).to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs.flatten(0, 1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks,
                                            ndc, inverse_y, flip_x, flip_y,
                                            model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(
        Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N, 3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            ndc=ndc,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i + CHUNK] = model.hit_coarse_geo(
                rays_o=rays_o[i:i + CHUNK],
                rays_d=rays_d[i:i + CHUNK],
                **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top + n].copy_(img[mask])
        rays_o_tr[top:top + n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top + n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top + n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:',
          eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling_sr(rgb_tr_ori, train_poses, HW, Ks,
                                               ndc, inverse_y, flip_x, flip_y,
                                               model, render_kwargs, cfgs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(
        Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    H, W = HW[0]
    # rgb_tr = torch.zeros([N,3], device=DEVICE)
    rgb_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr_ori.device)
    rays_o_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr_ori), H, W, 3], device=rgb_tr.device)

    def mask_patch_generator(arr_all, index_all):
        # arr_all_sr = arr_all
        list_bp = list(range(len(arr_all)))
        num_total = len(list_bp)
        idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0

        while True:
            if top >= num_total:
                idx_im, top = torch.LongTensor(
                    np.random.permutation(list_bp)), 0
            bp_chioce = idx_im[top]
            image_chioce = index_all[bp_chioce]
            patch_chioce = arr_all[bp_chioce]
            patch_4x_chioce = patch_chioce
            # patch_chioce = arr_all[patch_ind]
            # patch_4x_chioce = arr_all_sr[patch_ind]
            top += 1
            pr, pc = patch_chioce.shape[0], patch_chioce.shape[1]
            patch_chioce = patch_chioce.reshape(-1, 2)
            patch_4x_chioce = patch_4x_chioce.reshape(-1, 2)
            patch_chioce = np.moveaxis(patch_chioce, -1, 0)
            patch_4x_chioce = np.moveaxis(patch_4x_chioce, -1, 0)
            yield image_chioce, list(patch_chioce[0]), list(
                patch_chioce[1]), list(patch_4x_chioce[0]), list(
                    patch_4x_chioce[1]), [pr, pc]

    imsz = []
    top = 0
    idx_b = 0
    patch_all = []
    index_all = []
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
            H=H,
            W=W,
            K=K,
            c2w=c2w,
            ndc=ndc,
            inverse_y=inverse_y,
            flip_x=flip_x,
            flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i + CHUNK] = model.hit_coarse_geo(
                rays_o=rays_o[i:i + CHUNK],
                rays_d=rays_d[i:i + CHUNK],
                **render_kwargs).to(DEVICE)
        n = mask.sum()
        arr_all = patch_gen(imsz=[H, W], num_im=100, BS=4096, sz_patch=CHUNK)
        masks_arr = [mask[arr[:, :, 0], arr[:, :, 1]] for arr in arr_all]
        for idx, patch_mask in enumerate(masks_arr):
            if patch_mask.sum() > 2048:
                index_all.append(idx_b)
                patch_all.append(arr_all[idx])

        rgb_tr[idx_b].copy_(img)
        rays_o_tr[idx_b].copy_(rays_o.to(DEVICE))
        rays_d_tr[idx_b].copy_(rays_d.to(DEVICE))
        viewdirs_tr[idx_b].copy_(viewdirs.to(DEVICE))
        imsz.append(n)
        top += n
        idx_b += 1
        print(f'patches of image {idx_b} has generated.')

    patch_all = np.stack(patch_all, axis=0)
    index_all = np.stack(index_all, axis=0)
    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    patch_generator = mask_patch_generator(patch_all, index_all)
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:',
          eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, patch_generator


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top + BS]
        top += BS


def batch_images_generator(N, imsz, BS):
    idx, top = range(imsz), 0
    n_im = 0
    while True:
        if top + BS >= imsz:
            yield idx[top:imsz], n_im, True
            idx, top = range(imsz), 0
            n_im += 1
            if n_im >= N:
                n_im = 0
        else:
            yield idx[top:top + BS], n_im, False
            top += BS


def simg_patch_indices_generator(imsz, BS):
    BS = BS // 64
    H, W = imsz[0], imsz[1]
    num_x, num_y = H // BS, W // BS
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(x, y)
    arr_index = np.stack((yy, xx), axis=-1).astype(np.int64)

    slice_x = np.linspace(1, num_x, num_x).astype(np.int64) * BS
    slcie_y = np.linspace(1, num_y, num_y).astype(np.int64) * BS
    arr_yp = np.split(arr_index, slice_x, axis=0)
    arr_yp_last = arr_yp.pop(-1)
    arr_yp = np.stack(arr_yp, axis=-1)
    arr_xyp = np.split(arr_yp, slcie_y, axis=1)
    arr_yp_last = np.split(arr_yp_last, slcie_y, axis=1)
    arr_xp_last = arr_xyp.pop(-1)
    arr_xp_last = list(np.moveaxis(arr_xp_last, -1, 0))
    arr_xyp = np.concatenate(arr_xyp, axis=-1)
    arr_xyp = list(np.moveaxis(arr_xyp, -1, 0))

    arr_all = []
    arr_all.extend(arr_xyp)
    arr_all.extend(arr_xp_last)
    arr_all.extend(arr_yp_last)
    num_p = len(arr_all)
    idx, top = torch.LongTensor(np.random.permutation(num_p)), 0
    while True:
        if top >= num_p:
            idx, top = torch.LongTensor(np.random.permutation(num_p)), 0
        patch_chioce = arr_all[idx[top]]
        top += 1
        patch_chioce = patch_chioce.reshape(-1, 2)
        yield list(np.moveaxis(patch_chioce, -1, 0))


def patch_gen(imsz, num_im, BS, sz_patch):
    BS = BS // sz_patch
    H, W = imsz[0], imsz[1]
    num_x, num_y = H // BS, W // BS
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xx, yy = np.meshgrid(x, y)
    arr_index = np.stack((yy, xx), axis=-1).astype(np.int64)

    slice_x = np.linspace(1, num_x, num_x).astype(np.int64) * BS
    slcie_y = np.linspace(1, num_y, num_y).astype(np.int64) * BS
    arr_yp = np.split(arr_index, slice_x, axis=0)
    arr_yp_last = arr_yp.pop(-1)
    arr_yp = np.stack(arr_yp, axis=-1)
    arr_xyp = np.split(arr_yp, slcie_y, axis=1)
    arr_yp_last = np.split(arr_yp_last, slcie_y, axis=1)
    arr_xp_last = arr_xyp.pop(-1)
    arr_xp_last = list(np.moveaxis(arr_xp_last, -1, 0))
    arr_xyp = np.concatenate(arr_xyp, axis=-1)
    arr_xyp = list(np.moveaxis(arr_xyp, -1, 0))

    arr_all = []
    arr_all.extend(arr_xyp)
    arr_all.extend(arr_xp_last)
    arr_all.extend(arr_yp_last)

    return arr_all


def mimg_patch_indices_generator(imsz, num_im, BS, sz_patch, sr_ratio):

    arr_all = patch_gen(imsz, num_im, BS, sz_patch)
    arr_all_sr = patch_gen(imsz * sr_ratio, num_im, BS * sr_ratio, sz_patch)

    num_p = len(arr_all)
    list_p = np.ones(num_p)
    list_b = [list_p * i for i in range(num_im)]
    list_b = np.concatenate(list_b, axis=0)
    list_p = np.array(range(num_p))
    list_p = np.tile(list_p, num_im)
    list_bp = np.stack((list_b, list_p), axis=1)
    num_total = num_p * num_im
    idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0

    while True:
        if top >= num_total:
            idx_im, top = torch.LongTensor(np.random.permutation(list_bp)), 0
        bp_chioce = idx_im[top]
        image_chioce, patch_ind = bp_chioce[0], bp_chioce[1]
        patch_chioce = arr_all[patch_ind]
        patch_4x_chioce = arr_all_sr[patch_ind]
        top += 1
        pr, pc = patch_chioce.shape[0], patch_chioce.shape[1]
        patch_chioce = patch_chioce.reshape(-1, 2)
        patch_4x_chioce = patch_4x_chioce.reshape(-1, 2)
        patch_chioce = np.moveaxis(patch_chioce, -1, 0)
        patch_4x_chioce = np.moveaxis(patch_4x_chioce, -1, 0)
        yield image_chioce, list(patch_chioce[0]), list(patch_chioce[1]), list(
            patch_4x_chioce[0]), list(patch_4x_chioce[1]), [pr, pc]


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class SFTLayer(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(num_grow_ch, num_grow_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(num_grow_ch, num_feat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(num_grow_ch, num_grow_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(num_grow_ch, num_feat, 1)

    def forward(self, x, cond):
        scale = self.SFT_scale_conv1(
            F.leaky_relu(self.SFT_scale_conv0(cond), 0.2, inplace=True))
        shift = self.SFT_shift_conv1(
            F.leaky_relu(self.SFT_shift_conv0(cond), 0.2, inplace=True))
        return x * (scale + 1) + shift


class ResidualDenseBlock_SFT(nn.Module):
    """Residual Dense Block.
    Used in RRDB block in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock_SFT, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1,
                               1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.sft0 = SFTLayer(num_feat, num_grow_ch)
        self.sft1 = SFTLayer(num_grow_ch, num_grow_ch)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        xc0 = self.sft0(x[0], x[1])
        x1 = self.lrelu(self.conv1(xc0))
        x2 = self.lrelu(self.conv2(torch.cat((xc0, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((xc0, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((xc0, x1, x2, x3), 1)))
        xc1 = self.sft1(x4, x[1])
        x5 = self.conv5(torch.cat((xc0, x1, x2, x3, xc1), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return (x5 * 0.2 + x[0], x[1])


class RRDB_SFT(nn.Module):
    """Residual in Residual Dense Block.
    Used in RRDB-Net in ESRGAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB_SFT, self).__init__()
        self.rdb1 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock_SFT(num_feat, num_grow_ch)
        self.sft0 = SFTLayer(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.sft0(out[0], x[1])
        # Emperically, we use 0.2 to scale the residual for better performance
        return (out * 0.2 + x[0], x[1])


class SFTNet(nn.Module):
    """
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self,
                 n_in_colors,
                 scale,
                 num_feat=64,
                 num_block=5,
                 num_grow_ch=32,
                 num_cond=1,
                 dswise=False):
        super(SFTNet, self).__init__()
        self.scale = scale
        self.dswise = dswise
        # if scale == 2:
        #     num_in_ch = num_in_ch * 4
        # elif scale == 1:
        #     num_in_ch = num_in_ch * 16
        if dswise:
            self.conv_first = nn.Conv2d(n_in_colors, num_feat, 1)
        else:
            self.conv_first = nn.Conv2d(n_in_colors, num_feat, 3, 1, 1)
        self.body = make_layer(
            RRDB_SFT, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if n_in_colors > 3:
            self.conv_fea = nn.Conv2d(n_in_colors, num_feat, 3, 1, 1)
            self.conv_prefea = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1)

        # upsample
        if self.scale > 1:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.scale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sftbody = SFTLayer(num_feat, num_grow_ch)
        self.CondNet = nn.Sequential(
            nn.Conv2d(num_cond, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2,
                                               True), nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(0.2, True), nn.Conv2d(64, 32, 1))

    def forward(self, x, cond, fea=None):
        if fea is None:
            feat = self.conv_first(x)
        else:
            feat_rgb = self.conv_first(x)
            # feat += torch.sigmoid(self.conv_fea(fea))
            feat = torch.cat((feat_rgb, fea), dim=1)
            feat = self.conv_prefea(feat)
        cond = self.CondNet(cond)
        body_feat = self.body((feat, cond))
        body_feat = self.sftbody(body_feat[0], body_feat[1])
        body_feat = self.conv_body(body_feat)
        body_feat += feat
        # upsample
        if self.scale > 1:
            body_feat = self.lrelu(
                self.conv_up1(
                    F.interpolate(body_feat, scale_factor=2, mode='nearest')))
            if self.scale == 4:
                body_feat = self.lrelu(
                    self.conv_up2(
                        F.interpolate(
                            body_feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(body_feat)))
        return out

    def tile_process(self, img, cond, tile_size, tile_pad=10):
        """Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)
        cond = cond.unsqueeze(0)

        # start with black image
        output = img.new_zeros(output_shape).to('cpu')
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad,
                                 input_start_x_pad:input_end_x_pad]
                cond_tile = cond[:, :, input_start_y_pad:input_end_y_pad,
                                 input_start_x_pad:input_end_x_pad]
                # upscale tile
                # try:
                with torch.no_grad():
                    out_t = self(input_tile, cond_tile)
                # except Exception as error:
                #     print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                out_s_x = input_start_x * self.scale
                out_e_x = input_end_x * self.scale
                out_s_y = input_start_y * self.scale
                out_e_y = input_end_y * self.scale

                # output tile area without padding
                out_s_x_t = (input_start_x - input_start_x_pad) * self.scale
                out_e_x_t = out_s_x_t + input_tile_width * self.scale
                out_s_y_t = (input_start_y - input_start_y_pad) * self.scale
                out_e_y_t = out_s_y_t + input_tile_height * self.scale

                # put tile into output image
                output[:, :, out_s_y:out_e_y,
                       out_s_x:out_e_x] = out_t[:, :, out_s_y_t:out_e_y_t,
                                                out_s_x_t:out_e_x_t]
        return output.detach().to('cpu')

    def load_network(self,
                     load_path,
                     device,
                     strict=True,
                     param_key='params_ema'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        # net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=device)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                print('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        print(
            f'Loading {self.__class__.__name__} model from {load_path}, with param key: [{param_key}].'
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(load_net, strict)
        self.load_state_dict(load_net, strict=strict)

    def _print_different_keys_loading(self, load_net, strict=True):
        """Print keys with differnet name or different size when loading models.

        1. Print keys with differnet names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                          f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def save_network(self,
                     save_root,
                     net_label,
                     current_iter,
                     param_key='params'):

        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(save_root, save_filename)

        net = self if isinstance(self, list) else [self]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(
            param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                print(
                    f'Save model error: {e}, remaining retry times: {retry - 1}'
                )
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            print(f'Still cannot save {save_path}. Just ignore it.')
