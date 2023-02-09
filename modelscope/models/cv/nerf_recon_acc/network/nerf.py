# The implementation is modified from nerfacc, made publicly available under the MIT License
# at https://github.com/KAIR-BAIR/nerfacc/blob/master/examples/radiance_fields/ngp.py
import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from nerfacc import ContractionType, OccupancyGrid, ray_marching, rendering
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from .utils import chunk_batch, cleanup, get_activation, normalize


class VanillaMLP(nn.Module):

    def __init__(self, dim_in, dim_out, n_neurons, n_hidden_layers,
                 activation):
        super().__init__()
        self.layers = [
            self.make_linear(dim_in, n_neurons),
            self.make_activation()
        ]
        for i in range(n_hidden_layers - 1):
            self.layers += [
                self.make_linear(n_neurons, n_neurons),
                self.make_activation()
            ]
        self.layers += [self.make_linear(n_neurons, dim_out)]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(activation)

    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out):
        layer = nn.Linear(dim_in, dim_out, bias=True)
        torch.nn.init.constant_(layer.bias, 0.0)
        torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        return layer

    def make_activation(self):
        return nn.ReLU(inplace=True)


class MarchingCubeHelper(nn.Module):

    def __init__(self, resolution, use_torch=True):
        super().__init__()
        self.resolution = resolution
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self):
        if self.verts is None:
            x, y, z = torch.linspace(*self.points_range,
                                     self.resolution), torch.linspace(
                                         *self.points_range,
                                         self.resolution), torch.linspace(
                                             *self.points_range,
                                             self.resolution)
            x, y, z = torch.meshgrid(x, y, z)
            verts = torch.cat(
                [x.reshape(-1, 1),
                 y.reshape(-1, 1),
                 z.reshape(-1, 1)], dim=-1).reshape(-1, 3)
            self.verts = verts.cuda()
        return self.verts

    def forward(self, level, threshold=0.):
        level = level.float().view(self.resolution, self.resolution,
                                   self.resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.cuda(), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(),
                                        threshold)  # transform to numpy
            verts, faces = torch.from_numpy(
                verts.astype(np.float32)), torch.from_numpy(
                    faces.astype(np.int64))  # transform back to pytorch
        verts = verts / (self.resolution - 1.)
        return {'v_pos': verts, 't_pos_idx': faces}


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = _TruncExp.apply


class VolumeDensity(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.radius = self.config.radius
        self.n_input_dims = 3
        self.n_output_dims = self.config.geometry_feature_dim + 1
        point_encoding = tcnn.Encoding(
            n_input_dims=self.n_input_dims,
            encoding_config={
                'otype': 'HashGrid',
                'n_levels': self.config.n_levels,
                'n_features_per_level': self.config.n_features_per_level,
                'log2_hashmap_size': self.config.log2_hashmap_size,
                'base_resolution': self.config.base_resolution,
                'per_level_scale': self.config.per_level_scale,
            },
        )
        point_network = VanillaMLP(point_encoding.n_output_dims,
                                   self.n_output_dims, 64, 1, 'none')
        self.encoding_with_network = torch.nn.Sequential(
            point_encoding, point_network)

        self.density_activation = trunc_exp
        self.helper = MarchingCubeHelper(
            self.config.isosurface_resolution, use_torch=False)

    def forward(self, points):
        points = normalize(points, (-self.radius, self.radius), (0, 1))
        out = self.encoding_with_network(points.view(
            -1, self.n_input_dims)).view(*points.shape[:-1],
                                         self.n_output_dims).float()
        density, feature = out[..., 0], out[..., 1:]
        density = self.density_activation(density - 1)
        return density, feature

    def forward_level(self, points):
        points = normalize(points, (-self.radius, self.radius), (0, 1))
        out = self.encoding_with_network(points.view(
            -1, self.n_input_dims)).view(*points.shape[:-1],
                                         self.n_output_dims).float()
        density, _ = out[..., 0], out[..., 1:]
        density = self.density_activation(density - 1)
        return -density

    def isosurface_(self, vmin, vmax):
        grid_verts = self.helper.grid_vertices()
        grid_verts_1 = normalize(grid_verts[..., 0], (0, 1),
                                 (vmin[0], vmax[0]))
        grid_verts_2 = normalize(grid_verts[..., 1], (0, 1),
                                 (vmin[1], vmax[1]))
        grid_verts_3 = normalize(grid_verts[..., 2], (0, 1),
                                 (vmin[2], vmax[2]))
        grid_verts = torch.stack([grid_verts_1, grid_verts_2, grid_verts_3],
                                 dim=-1)

        def batch_func(x):
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv

        level = chunk_batch(batch_func, self.config.isosurface_chunk,
                            grid_verts)
        mesh = self.helper(level, threshold=self.config.isosurface_threshold)
        mesh_1 = normalize(mesh['v_pos'][..., 0], (0, 1), (vmin[0], vmax[0]))
        mesh_2 = normalize(mesh['v_pos'][..., 1], (0, 1), (vmin[1], vmax[1]))
        mesh_3 = normalize(mesh['v_pos'][..., 2], (0, 1), (vmin[2], vmax[2]))
        mesh['v_pos'] = torch.stack([mesh_1, mesh_2, mesh_3], dim=-1)
        return mesh

    @torch.no_grad()
    def isosurface(self):
        mesh_coarse = self.isosurface_(
            (-self.radius, -self.radius, -self.radius),
            (self.radius, self.radius, self.radius))
        vmin, vmax = mesh_coarse['v_pos'].amin(
            dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        mesh_fine = self.isosurface_(vmin_, vmax_)
        return mesh_fine


class VolumeRadiance(nn.Module):

    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = 3
        self.n_output_dims = 3
        self.direction_encoding = tcnn.Encoding(
            n_input_dims=self.n_dir_dims,
            encoding_config={
                'otype':
                'Composite',
                'nested': [
                    {
                        'n_dims_to_encode': self.n_dir_dims,
                        'otype': 'SphericalHarmonics',
                        'degree': self.config.degree,
                    },
                ],
            },
        )
        self.n_input_dims = self.config.geometry_feature_dim + self.direction_encoding.n_output_dims
        self.network = VanillaMLP(self.n_input_dims, self.n_output_dims, 64, 2,
                                  'sigmoid')

    def forward(self, features, dirs):
        dirs = (dirs + 1.) / 2.  # (-1, 1) => (0, 1)
        dirs_embd = self.direction_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat(
            [dirs_embd,
             features.view(-1, self.config.geometry_feature_dim)],
            dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1],
                                               self.n_output_dims).float()
        return color


class NeRFModel(nn.Module):

    def __init__(self, network_cfg, **kwargs):
        super().__init__()
        self.config = network_cfg
        self.num_samples_per_ray = kwargs['num_samples_per_ray']
        self.test_ray_chunk = kwargs['test_ray_chunk']
        self.background = self.config.background
        self.geometry = VolumeDensity(self.config)
        self.texture = VolumeRadiance(self.config)
        radius_list = [
            -self.config.radius, -self.config.radius, -self.config.radius,
            self.config.radius, self.config.radius, self.config.radius
        ]
        radius_tensor = torch.as_tensor(radius_list, dtype=torch.float32)
        self.register_buffer('scene_aabb', radius_tensor)
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=128,
            contraction_type=ContractionType.AABB)
        self.render_step_size = 1.732 * 2 * self.config.radius / self.num_samples_per_ray

    def update_step(self, global_step):
        # progressive viewdir PE frequencies

        def occ_eval_fn(x):
            density, _ = self.geometry(x)
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
            return density[..., None] * self.render_step_size

        self.occupancy_grid.every_n_step(
            step=global_step, occ_eval_fn=occ_eval_fn)

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def forward(self, rays):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        if self.training:
            if self.background == 'random':
                background_color = torch.rand(
                    3, dtype=torch.float32, device=rays_o.device)
            elif self.background == 'white':
                background_color = torch.ones(
                    3, dtype=torch.float32, device=rays_o.device)
            elif self.background == 'black':
                background_color = torch.zeros(
                    3, dtype=torch.float32, device=rays_o.device)
        else:
            background_color = torch.ones(
                3, dtype=torch.float32, device=rays_o.device)

        def sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, _ = self.geometry(positions)

            return density[..., None]

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.geometry(positions)
            rgb = self.texture(feature, t_dirs)
            return rgb, density[..., None]

        with torch.no_grad():
            packed_info, t_starts, t_ends = ray_marching(
                rays_o,
                rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid,
                sigma_fn=sigma_fn,
                near_plane=None,
                far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.training,
                cone_angle=0.0,
                alpha_thre=0.0)
        rgb, opacity, depth = rendering(
            packed_info,
            t_starts,
            t_ends,
            rgb_sigma_fn=rgb_sigma_fn,
            render_bkgd=background_color)

        opacity, depth = opacity.squeeze(-1), depth.squeeze(-1)

        return {
            'comp_rgb':
            rgb,
            'opacity':
            opacity,
            'depth':
            depth,
            'rays_valid':
            opacity > 0,
            'num_samples':
            torch.as_tensor([len(t_starts)],
                            dtype=torch.int32,
                            device=rays.device)
        }

    def inference(self, rays):
        out = chunk_batch(self.forward, self.test_ray_chunk, rays)
        return {**out}
