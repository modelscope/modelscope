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
Utils for extracting 3D shapes using marching cubes. Based on code from DeepSDF (Park et al.)

Takes as input an .mrc file and extracts a mesh.

Ex.
    python shape_utils.py my_shape.mrc
Ex.
    python shape_utils.py myshapes_directory --level=12
"""

import numpy as np
import plyfile
import skimage.measure


def convert_sdf_samples_to_ply(numpy_3d_sdf_tensor,
                               voxel_grid_origin,
                               voxel_size,
                               ply_filename_out,
                               offset=None,
                               scale=None,
                               level=0.0):

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3)
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts, ),
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(), )))
    faces_tuple = np.array(
        faces_building, dtype=[('vertex_indices', 'i4', (3, ))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, 'vertex')
    el_faces = plyfile.PlyElement.describe(faces_tuple, 'face')

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)
