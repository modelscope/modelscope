# ------------------------------------------------------------------------
# Modified from https://github.com/facebookresearch/pytorch3d
# All Rights Reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def batch_mm(matrix, matrix_batch):
    """
    https://github.com/pytorch/pytorch/issues/14489#issuecomment-607730242
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product,
    size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector
    # product of the columns.
    # And then reverse the reshaping.
    # (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size,
                                      -1).transpose(1, 0)


def aa2quat(rots, form='wxyz', unified_orient=True):
    """
    Convert angle-axis representation to wxyz quaternion
    and to the half plan (w >= 0)
    @param rots: angle-axis rotations, (*, 3)
    @param form: quaternion format, either 'wxyz' or 'xyzw'
    @param unified_orient: Use unified orientation for quaternion
    (quaternion is dual cover of SO3)
    :return:
    """
    angles = rots.norm(dim=-1, keepdim=True)
    norm = angles.clone()
    norm[norm < 1e-8] = 1
    axis = rots / norm
    quats = torch.empty(
        rots.shape[:-1] + (4, ), device=rots.device, dtype=rots.dtype)
    angles = angles * 0.5
    if form == 'wxyz':
        quats[..., 0] = torch.cos(angles.squeeze(-1))
        quats[..., 1:] = torch.sin(angles) * axis
    elif form == 'xyzw':
        quats[..., :3] = torch.sin(angles) * axis
        quats[..., 3] = torch.cos(angles.squeeze(-1))

    if unified_orient:
        idx = quats[..., 0] < 0
        quats[idx, :] *= -1

    return quats


def quat2aa(quats):
    """
    Convert wxyz quaternions to angle-axis representation
    :param quats:
    :return:
    """
    _cos = quats[..., 0]
    xyz = quats[..., 1:]
    _sin = xyz.norm(dim=-1)
    norm = _sin.clone()
    norm[norm < 1e-7] = 1
    axis = xyz / norm.unsqueeze(-1)
    angle = torch.atan2(_sin, _cos) * 2
    return axis * angle.unsqueeze(-1)


def quat2mat(quats: torch.Tensor):
    """
    Convert (w, x, y, z) quaternions to 3x3 rotation matrix
    :param quats: quaternions of shape (..., 4)
    :return:  rotation matrices of shape (..., 3, 3)
    """
    qw = quats[..., 0]
    qx = quats[..., 1]
    qy = quats[..., 2]
    qz = quats[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(
        quats.shape[:-1] + (3, 3), device=quats.device, dtype=quats.dtype)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def quat2euler(q, order='xyz', degrees=True):
    """
    Convert (w, x, y, z) quaternions to xyz euler angles.
    This is  used for bvh output.
    """
    q0 = q[..., 0]
    q1 = q[..., 1]
    q2 = q[..., 2]
    q3 = q[..., 3]
    es = torch.empty(q0.shape + (3, ), device=q.device, dtype=q.dtype)

    if order == 'xyz':
        es[..., 2] = torch.atan2(2 * (q0 * q3 - q1 * q2),
                                 q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)
        es[..., 1] = torch.asin((2 * (q1 * q3 + q0 * q2)).clip(-1, 1))
        es[..., 0] = torch.atan2(2 * (q0 * q1 - q2 * q3),
                                 q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3)
    else:
        raise NotImplementedError('Cannot convert to ordering %s' % order)

    if degrees:
        es = es * 180 / np.pi

    return es


def aa2mat(rots):
    """
    Convert angle-axis representation to rotation matrix
    :param rots: angle-axis representation
    :return:
    """
    quat = aa2quat(rots)
    mat = quat2mat(quat)
    return mat


def inv_affine(mat):
    """
    Calculate the inverse of any affine transformation
    """
    affine = torch.zeros((mat.shape[:2] + (1, 4)))
    affine[..., 3] = 1
    vert_mat = torch.cat((mat, affine), dim=2)
    vert_mat_inv = torch.inverse(vert_mat)
    return vert_mat_inv[..., :3, :]


def inv_rigid_affine(mat):
    """
    Calculate the inverse of a rigid affine transformation
    """
    res = mat.clone()
    res[..., :3] = mat[..., :3].transpose(-2, -1)
    res[...,
        3] = -torch.matmul(res[..., :3], mat[..., 3].unsqueeze(-1)).squeeze(-1)
    return res


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f'Invalid rotation matrix shape {matrix.shape}.')

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9, )), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ))

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01],
                        dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20],
                        dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21],
                        dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2],
                        dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[F.one_hot(q_abs.argmax(
        dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4, ))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
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


def mat2aa(matrix: torch.Tensor) -> torch.Tensor:
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


def batch_rodrigues(rot_vecs: Tensor, epsilon: float = 1e-8) -> Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
