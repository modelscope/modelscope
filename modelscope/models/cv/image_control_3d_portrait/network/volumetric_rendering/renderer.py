# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""

import math

import torch
import torch.nn as nn

from . import math_utils
from .ray_marcher import MipRayMarcher2


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor(
        [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
         [[0, 0, 1], [1, 0, 0], [0, 1, 0]]],
        dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1,
                                                  -1).reshape(
                                                      N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(
        N, -1, -1, -1).reshape(N * n_planes, 3, 3).to(coordinates.device)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(plane_axes,
                       plane_features,
                       coordinates,
                       mode='bilinear',
                       padding_mode='zeros',
                       box_warp=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)

    coordinates = (2 / box_warp) * coordinates  # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes,
                                                coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(
        plane_features,
        projected_coordinates.float(),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features


def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(
        grid.expand(batch_size, -1, -1, -1, -1),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2,
                                                1).reshape(N, H * W * D, C)
    return sampled_features


class ImportanceRenderer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions,
                rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options[
                'ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(
                ray_origins,
                ray_directions,
                box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(
                ray_origins, ray_start, ray_end,
                rendering_options['depth_resolution'],
                rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(
                ray_origins, rendering_options['ray_start'],
                rendering_options['ray_end'],
                rendering_options['depth_resolution'],
                rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates = (
            ray_origins.unsqueeze(-2)
            + depths_coarse * ray_directions.unsqueeze(-2)).reshape(
                batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(
            -1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates,
                             sample_directions, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays,
                                              samples_per_ray,
                                              colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays,
                                                    samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse,
                                             depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights,
                                                 N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(
                -1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (
                ray_origins.unsqueeze(-2)
                + depths_fine * ray_directions.unsqueeze(-2)).reshape(
                    batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates,
                                 sample_directions, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays,
                                              N_importance,
                                              colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays,
                                                    N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, depths_fine,
                colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(
                all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(
                colors_coarse, densities_coarse, depths_coarse,
                rendering_options)

        return rgb_final, depth_final, weights.sum(2)

    def run_model(self, planes, decoder, sample_coordinates, sample_directions,
                  options):
        sampled_features = sample_from_planes(
            self.plane_axes,
            planes,
            sample_coordinates,
            padding_mode='zeros',
            box_warp=options['box_warp'])

        out = decoder(sampled_features, sample_directions)
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(
                out['sigma']) * options['density_noise']
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(
            all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2,
                                     indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2,
                      densities2):
        all_depths = torch.cat([depths1, depths2], dim=-2)
        all_colors = torch.cat([colors1, colors2], dim=-2)
        all_densities = torch.cat([densities1, densities2], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(
            all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2,
                                     indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self,
                          ray_origins,
                          ray_start,
                          ray_end,
                          depth_resolution,
                          disparity_space_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = torch.linspace(
                0, 1, depth_resolution,
                device=ray_origins.device).reshape(1, 1, depth_resolution,
                                                   1).repeat(N, M, 1, 1)
            depth_delta = 1 / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1. / (1. / ray_start * (1. - depths_coarse)
                                  + 1. / ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                depths_coarse = math_utils.linspace(ray_start, ray_end,
                                                    depth_resolution).permute(
                                                        1, 2, 0, 3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[
                    ..., None]
            else:
                depths_coarse = torch.linspace(
                    ray_start,
                    ray_end,
                    depth_resolution,
                    device=ray_origins.device).reshape(1, 1, depth_resolution,
                                                       1).repeat(N, M, 1, 1)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(
                batch_size * num_rays,
                -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(
                weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1],
                                                N_importance).detach().reshape(
                                                    batch_size, num_rays,
                                                    N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(
            weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(
            pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf],
                        -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above],
                                   -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1,
                             inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1,
                              inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (
            bins_g[..., 1] - bins_g[..., 0])
        return samples
