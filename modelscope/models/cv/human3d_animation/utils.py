# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F


def read_obj(obj_path, print_shape=False):
    with open(obj_path, 'r') as f:
        bfm_lines = f.readlines()

    vertices = []
    faces = []
    uvs = []
    vns = []
    faces_uv = []
    faces_normal = []
    max_face_length = 0
    for line in bfm_lines:
        if line[:2] == 'v ':
            vertex = [
                float(a) for a in line.strip().split(' ')[1:] if len(a) > 0
            ]
            vertices.append(vertex)

        if line[:2] == 'f ':
            items = line.strip().split(' ')[1:]
            face = [int(a.split('/')[0]) for a in items if len(a) > 0]
            max_face_length = max(max_face_length, len(face))
            faces.append(face)

            if '/' in items[0] and len(items[0].split('/')[1]) > 0:
                face_uv = [int(a.split('/')[1]) for a in items if len(a) > 0]
                faces_uv.append(face_uv)

            if '/' in items[0] and len(items[0].split('/')) >= 3 and len(
                    items[0].split('/')[2]) > 0:
                face_normal = [
                    int(a.split('/')[2]) for a in items if len(a) > 0
                ]
                faces_normal.append(face_normal)

        if line[:3] == 'vt ':
            items = line.strip().split(' ')[1:]
            uv = [float(a) for a in items if len(a) > 0]
            uvs.append(uv)

        if line[:3] == 'vn ':
            items = line.strip().split(' ')[1:]
            vn = [float(a) for a in items if len(a) > 0]
            vns.append(vn)

    vertices = np.array(vertices).astype(np.float32)
    if max_face_length <= 3:
        faces = np.array(faces).astype(np.int32)
    else:
        print('not a triangle face mesh!')

    if vertices.shape[1] == 3:
        mesh = {
            'vertices': vertices,
            'faces': faces,
        }
    else:
        mesh = {
            'vertices': vertices[:, :3],
            'colors': vertices[:, 3:],
            'faces': faces,
        }

    if len(uvs) > 0:
        uvs = np.array(uvs).astype(np.float32)
        mesh['uvs'] = uvs

    if len(vns) > 0:
        vns = np.array(vns).astype(np.float32)
        mesh['normals'] = vns

    if len(faces_uv) > 0:
        if max_face_length <= 3:
            faces_uv = np.array(faces_uv).astype(np.int32)
        mesh['faces_uv'] = faces_uv

    if len(faces_normal) > 0:
        if max_face_length <= 3:
            faces_normal = np.array(faces_normal).astype(np.int32)
        mesh['faces_normal'] = faces_normal

    if print_shape:
        print('num of vertices', len(vertices))
        print('num of faces', len(faces))
    return mesh


def write_obj(save_path, mesh):
    save_dir = os.path.dirname(save_path)
    save_name = os.path.splitext(os.path.basename(save_path))[0]

    if 'texture_map' in mesh:
        cv2.imwrite(
            os.path.join(save_dir, save_name + '.png'), mesh['texture_map'])

        with open(os.path.join(save_dir, save_name + '.mtl'), 'w') as wf:
            wf.write('newmtl material_0\n')
            wf.write('Ka 1.000000 0.000000 0.000000\n')
            wf.write('Kd 1.000000 1.000000 1.000000\n')
            wf.write('Ks 0.000000 0.000000 0.000000\n')
            wf.write('Tr 0.000000\n')
            wf.write('illum 0\n')
            wf.write('Ns 0.000000\n')
            wf.write('map_Kd {}\n'.format(save_name + '.png'))

    with open(save_path, 'w') as wf:
        if 'texture_map' in mesh:
            wf.write('# Create by ModelScope\n')
            wf.write('mtllib ./{}.mtl\n'.format(save_name))

        if 'colors' in mesh:
            for i, v in enumerate(mesh['vertices']):
                wf.write('v {} {} {} {} {} {}\n'.format(
                    v[0], v[1], v[2], mesh['colors'][i][0],
                    mesh['colors'][i][1], mesh['colors'][i][2]))
        else:
            for v in mesh['vertices']:
                wf.write('v {} {} {}\n'.format(v[0], v[1], v[2]))

        if 'uvs' in mesh:
            for uv in mesh['uvs']:
                wf.write('vt {} {}\n'.format(uv[0], uv[1]))

        if 'normals' in mesh:
            for vn in mesh['normals']:
                wf.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))

        if 'faces' in mesh:
            for ind, face in enumerate(mesh['faces']):
                if 'faces_uv' in mesh or 'faces_normal' in mesh:
                    if 'faces_uv' in mesh:
                        face_uv = mesh['faces_uv'][ind]
                    else:
                        face_uv = face
                    if 'faces_normal' in mesh:
                        face_normal = mesh['faces_normal'][ind]
                    else:
                        face_normal = face
                    row = 'f ' + ' '.join([
                        '{}/{}/{}'.format(face[i], face_uv[i], face_normal[i])
                        for i in range(len(face))
                    ]) + '\n'
                else:
                    row = 'f ' + ' '.join(
                        ['{}'.format(face[i])
                         for i in range(len(face))]) + '\n'
                wf.write(row)


def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n / x, 0, 0, 0], [0, n / x, 0, 0],
                     [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                     [0, 0, -1, 0]]).astype(np.float32)


def translate(x, y, z):
    return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0],
                     [0, 0, 0, 1]]).astype(np.float32)


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0],
                     [0, 0, 0, 1]]).astype(np.float32)


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(
        dot(x, x),
        min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx,
                                                       np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip,
           max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(
        glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(
            uv[None, ...],
            rast_out,
            uv_idx,
            rast_db=rast_out_db,
            diff_attrs='all')
        color = dr.texture(
            tex[None, ...],
            texc,
            texd,
            filter_mode='linear-mipmap-linear',
            max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    pos_idx = pos_idx.type(torch.long)
    v0 = pos[pos_idx[:, 0], :]
    v1 = pos[pos_idx[:, 1], :]
    v2 = pos[pos_idx[:, 2], :]
    face_normals = safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(
        0, face_normals.shape[0], dtype=torch.int64,
        device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = dr.interpolate(face_normals[None, ...], rast_out,
                                            face_normal_indices.int())
    normal = (gb_geometric_normal + 1) * 0.5
    mask = torch.clamp(rast_out[..., -1:], 0, 1)
    color = color * mask + (1 - mask) * torch.ones_like(color)
    normal = normal * mask + (1 - mask) * torch.ones_like(normal)

    return color, mask, normal


# The following code is based on https://github.com/Mathux/ACTOR.git
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Check PYTORCH3D_LICENCE before use


def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix  shape f{matrix.shape}.')
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles])
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48)
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
