# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
from torch._C import dtype


def mat2euler(mat):
    euler = torch.ones(mat.shape[0], 3, dtype=mat.dtype, device=mat.device)
    cy_thresh = 1e-6
    # try:
    #     cy_thresh = np.finfo(mat.dtype).eps * 4
    # except ValueError:
    #     cy_thresh = np.finfo(np.float).eps * 4.0
    # print("cy_thresh", cy_thresh)
    r11, r12, r13, r21, r22, r23, _, _, r33 = mat[:, 0, 0], mat[:, 0, 1], mat[:, 0, 2], \
        mat[:, 1, 0], mat[:, 1, 1], mat[:, 1, 2], \
        mat[:, 2, 0], mat[:, 2, 1], mat[:, 2, 2]
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33 * r33 + r23 * r23)

    mask = cy > cy_thresh

    if torch.sum(mask) > 1:
        euler[mask, 0] = torch.atan2(-r23, r33)[mask]
        euler[mask, 1] = torch.atan2(r13, cy)[mask]
        euler[mask, 2] = torch.atan2(-r12, r11)[mask]

    mask = cy <= cy_thresh
    if torch.sum(mask) > 1:
        print('mat2euler!!!!!!')
        euler[mask, 0] = 0.0
        euler[mask, 1] = torch.atan2(r13, cy)  # atan2(sin(y), cy)
        euler[mask, 2] = torch.atan2(r21, r22)

    return euler


########################################################################################################################


def euler2mat(angle):
    """Convert euler angles to rotation matrix"""
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack(
        [cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones],
        dim=1).view(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack(
        [cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy],
        dim=1).view(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack(
        [ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx],
        dim=1).view(B, 3, 3)

    rot_mat = xmat.bmm(ymat).bmm(zmat)
    return rot_mat


########################################################################################################################


def pose_vec2mat(vec, mode='euler'):
    """Convert Euler parameters to transformation matrix."""
    if mode is None:
        return vec
    trans, rot = vec[:, :3].unsqueeze(-1), vec[:, 3:]
    if mode == 'euler':
        rot_mat = euler2mat(rot)
    elif mode == 'axis_angle':
        from modelscope.models.cv.video_depth_estimation.geometry.pose_trans import axis_angle_to_matrix
        rot_mat = axis_angle_to_matrix(rot)
    else:
        raise ValueError('Rotation mode not supported {}'.format(mode))
    mat = torch.cat([rot_mat, trans], dim=2)  # [B,3,4]
    return mat


########################################################################################################################


def invert_pose(T):
    """Inverts a [B,4,4] torch.tensor pose"""
    Tinv = torch.eye(4, device=T.device, dtype=T.dtype).repeat([len(T), 1, 1])
    Tinv[:, :3, :3] = torch.transpose(T[:, :3, :3], -2, -1)
    Tinv[:, :3, -1] = torch.bmm(-1. * Tinv[:, :3, :3],
                                T[:, :3, -1].unsqueeze(-1)).squeeze(-1)
    return Tinv


########################################################################################################################


def invert_pose_numpy(T):
    """Inverts a [4,4] np.array pose"""
    Tinv = np.copy(T)
    R, t = Tinv[:3, :3], Tinv[:3, 3]
    Tinv[:3, :3], Tinv[:3, 3] = R.T, -np.matmul(R.T, t)
    return Tinv


########################################################################################################################
