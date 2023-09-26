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
The ray marcher takes the raw output of the implicit representation and
uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MipRayMarcher2(nn.Module):

    def __init__(self):
        super().__init__()

    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(
                densities_mid
                - 1)  # activation bias of -1 makes things initialize better
        else:
            assert False, 'MipRayMarcher only supports `clamp_mode`=`softplus`!'

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat(
            [torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths),
                                      torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1  # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights

    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(
            colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights
