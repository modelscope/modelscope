# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..DUT.config import cfg
from .ProjectionUtils import HomoCalc, HomoProj


def mesh_warp_frame(frame, x_motion, y_motion, cap_width, cap_height):
    """
    @param frame current frame [N, 1, H, W]
    @param x_motion [N, 1, G_H, G_W]
    @param y_motion [N, 1, G_H, G_W]

    @return mesh warping according to given motion
    """

    target_device = frame.device

    src_grids = torch.stack(
        torch.meshgrid(
            torch.arange(
                0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS, device=target_device),
            torch.arange(
                0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS, device=target_device)),
        0).permute(0, 2, 1).unsqueeze(0).float()  # 2, G_H, G_W

    des_grids = src_grids + torch.cat([x_motion, y_motion], 1)

    projection = []

    for i in range(des_grids.shape[0]):
        homo = HomoCalc(src_grids[0], des_grids[i])

        origin_kp = torch.stack(
            torch.meshgrid(
                torch.arange(0, cfg.MODEL.WIDTH, device=target_device),
                torch.arange(0, cfg.MODEL.HEIGHT, device=target_device)),
            0).permute(0, 2, 1).float()  # 2, H, W

        projected_kp = HomoProj(
            homo,
            origin_kp.contiguous().view(2, -1).permute(1, 0)).permute(1, 0)

        projection.append(projected_kp.contiguous().view(
            *origin_kp.shape).permute(1, 2, 0))  # 2, H, W --> H, W, 2
    projection = torch.stack(projection, 0)

    projection[:, :, :, 0] = projection[:, :, :, 0] / cfg.MODEL.WIDTH * 2. - 1.
    projection[:, :, :, 1] = projection[:, :, :, 1] / \
        cfg.MODEL.HEIGHT * 2. - 1.
    # warp with original images
    projection = projection.permute(0, 3, 1, 2)
    projection = F.interpolate(
        projection, (cap_height, cap_width),
        mode='bilinear',
        align_corners=True)
    projection = projection.permute(0, 2, 3, 1)

    generated_frame = F.grid_sample(frame, projection, align_corners=True)

    return generated_frame


def warpListImage(images, x_motion, y_motion, cap_width, cap_height):
    """
    @param images List(image [1, 1, H, W])
    @param x_motion [G_H, G_W, N]
    @param y_motion [G_H, G_W, N]
    """

    frames = np.concatenate(images, 0)
    x_motion = np.expand_dims(np.transpose(x_motion, (2, 0, 1)), 1)
    y_motion = np.expand_dims(np.transpose(y_motion, (2, 0, 1)), 1)
    frames = torch.from_numpy(frames.astype(np.float32))
    x_motion = torch.from_numpy(x_motion.astype(np.float32))
    y_motion = torch.from_numpy(y_motion.astype(np.float32))
    return mesh_warp_frame(frames, x_motion, y_motion, cap_width, cap_height)
