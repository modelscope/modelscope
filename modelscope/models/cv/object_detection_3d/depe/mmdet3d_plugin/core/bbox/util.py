"""
The implementation here is modified based on PETR, originally Apache-2.0 license and publicly available at
https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/core/bbox
"""
import mmdet3d
import numpy as np
import torch


def normalize_bbox(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    _l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, _l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, _l, cz, h, rot.sin(), rot.cos()), dim=-1)
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    nan = normalized_bboxes.isnan().sum()
    if nan > 0:
        print(f'found {nan} nan!')
        normalized_bboxes = torch.nan_to_num(normalized_bboxes, 0.1)
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    _l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    _l = _l.exp()
    h = h.exp()
    # check mmdet3d version
    if int(mmdet3d.__version__.split('.')[0]) > 0:
        tmp = w
        w = _l
        _l = tmp
        rot = -rot - 0.5 * np.pi
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, _l, h, rot, vx, vy],
                                        dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, _l, h, rot], dim=-1)
    return denormalized_bboxes
