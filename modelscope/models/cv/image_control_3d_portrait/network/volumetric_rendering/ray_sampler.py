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
The ray sampler is a module that takes in camera matrices and resolution and batches of rays.
Expects cam2world matrices that use the OpenCV camera coordinate system conventions.
"""

import torch


class RaySampler(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.ray_origins_h, self.ray_directions, self.depths, self.image_coords, self.rendering_options = \
            None, None, None, None, None

    def forward(self, cam2world_matrix, intrinsics, resolution):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        resolution: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        N, M = cam2world_matrix.shape[0], resolution**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(
            torch.meshgrid(
                torch.arange(
                    resolution,
                    dtype=torch.float32,
                    device=cam2world_matrix.device),
                torch.arange(
                    resolution,
                    dtype=torch.float32,
                    device=cam2world_matrix.device))) * (1. / resolution) + (
                        0.5 / resolution)
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1)
        y_cam = uv[:, :, 1].view(N, -1)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (x_cam - cx.unsqueeze(-1) + cy.unsqueeze(-1)
                  * sk.unsqueeze(-1) / fy.unsqueeze(-1) - sk.unsqueeze(-1)
                  * y_cam / fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z_cam
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack(
            (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1)

        world_rel_points = torch.bmm(cam2world_matrix,
                                     cam_rel_points.permute(0, 2, 1)).permute(
                                         0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(
            1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs
