# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple

from modelscope.models.cv.video_stabilization.utils.ProjectionUtils import (
    HomoCalc, HomoProj, MotionDistanceMeasure)
from ..DUT.config import cfg


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0],
                     self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1, )).median(dim=-1)[0]
        return x


def SingleMotionPropagate(x_flow, y_flow, pts):
    """
    Traditional median filter for motion propagation
    @param: x_flow [B, 1, H, W]
    @param: y_flow [B, 1, H, W]
    @param: pts    [B*topk, 4]
    """

    pts = pts.float()

    medfilt = MedianPool2d(same=True)

    _, _, H, W = x_flow.shape
    grids = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)),
                        0).to(x_flow.device).permute(0, 2,
                                                     1)  # 2, W, H --> 2, H, W
    grids = grids.unsqueeze(0)  # 1, 2, H, W
    grids = grids.float()
    new_points = grids + torch.cat([x_flow, y_flow], 1)  # B, 2, H, W
    new_points_S = new_points.clone()
    new_points = new_points[0, :, pts[:, 2].long(),
                            pts[:, 3].long()].permute(1, 0)  # B*topK, 2
    old_points = grids[0, :, pts[:, 2].long(),
                       pts[:, 3].long()].permute(1, 0)  # B*topK, 2

    old_points_numpy = old_points.detach().cpu().numpy()
    new_points_numpy = new_points.detach().cpu().numpy()

    # pre-warping with global homography
    Homo, state = cv2.findHomography(old_points_numpy, new_points_numpy,
                                     cv2.RANSAC)

    if Homo is None:
        Homo = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    Homo = torch.from_numpy(Homo.astype(np.float32)).to(old_points.device)

    meshes_x, meshes_y = torch.meshgrid(
        torch.arange(0, W, cfg.MODEL.PIXELS),
        torch.arange(0, H, cfg.MODEL.PIXELS))
    meshes_x = meshes_x.float().permute(1, 0)
    meshes_y = meshes_y.float().permute(1, 0)
    meshes_x = meshes_x.to(old_points.device)
    meshes_y = meshes_y.to(old_points.device)
    meshes_z = torch.ones_like(meshes_x).to(old_points.device)
    meshes = torch.stack([meshes_x, meshes_y, meshes_z], 0)

    meshes_projected = torch.mm(Homo, meshes.view(3, -1)).view(*meshes.shape)
    x_motions = meshes[0, :, :] - meshes_projected[0, :, :] / (
        meshes_projected[2, :, :] + 1e-5)
    y_motions = meshes[1, :, :] - meshes_projected[1, :, :] / (
        meshes_projected[2, :, :] + 1e-5)

    temp_x_motion = torch.zeros_like(x_motions)
    temp_y_motion = torch.zeros_like(x_motions)

    for i in range(x_motions.shape[0]):
        for j in range(x_motions.shape[1]):
            distance = torch.sqrt((pts[:, 2] - i * cfg.MODEL.PIXELS)**2
                                  + (pts[:, 3] - j * cfg.MODEL.PIXELS)**2)
            distance = distance < cfg.MODEL.RADIUS
            index = distance.nonzero(
            )  # the indexes whose distances are smaller than RADIUS
            if index.shape[0] == 0:
                continue
            old_points_median = pts[index[:, 0].long(), :]  # N', 4(B, C, H, W)
            dominator = old_points_median[:, 3:4] * Homo[2, 0] + \
                old_points_median[:, 2:3] * Homo[2, 1] + \
                Homo[2, 2] + 1e-5  # N', 1
            x_nominator = old_points_median[:, 3:4] * Homo[0, 0] + \
                old_points_median[:, 2:3] * Homo[0, 1] + Homo[0, 2]
            y_nominator = old_points_median[:, 3:4] * Homo[1, 0] + \
                old_points_median[:, 2:3] * Homo[1, 1] + Homo[1, 2]
            new_points_homo = torch.cat(
                [x_nominator / dominator, y_nominator / dominator],
                -1)  # N', 2
            new_points_flow = new_points_S[
                0, :, old_points_median[:, 2].long(),
                old_points_median[:, 3].long()].permute(1, 0)  # N', 2
            temp_motion = new_points_flow - new_points_homo
            temp_x_motion[i, j] = temp_motion[:, 0].median()
            temp_y_motion[i, j] = temp_motion[:, 1].median()

    x_motions = x_motions + temp_x_motion
    y_motions = y_motions + temp_y_motion

    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motions.unsqueeze(0).unsqueeze(0))
    y_motion_mesh = medfilt(y_motions.unsqueeze(0).unsqueeze(0))

    return torch.cat([x_motion_mesh, y_motion_mesh], 1)


def MultiMotionPropagate(x_flow, y_flow, pts):
    """
    Median filter for propagation with multi homography
    @param: x_flow B, 1, H, W
    @param: y_flow B, 1, H, W
    @param: pts    B*topk, 4
    """

    medfilt = MedianPool2d(same=True)

    # spreads motion over the mesh for the old_frame
    from sklearn.cluster import KMeans
    pts = pts.float()

    B, C, H, W = x_flow.shape
    grids = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H)),
                        0).to(x_flow.device).permute(0, 2,
                                                     1)  # 2, W, H --> 2, H, W
    grids = grids.unsqueeze(0)  # 1, 2, H, W
    grids = grids.float()
    new_points = grids + torch.cat([x_flow, y_flow], 1)  # B, 2, H, W
    new_points = new_points[0, :, pts[:, 2].long(),
                            pts[:, 3].long()].permute(1, 0)  # B*topK, 2
    old_points = grids[0, :, pts[:, 2].long(),
                       pts[:, 3].long()].permute(1, 0)  # B*topK, 2

    old_points_numpy = old_points.detach().cpu().numpy()
    new_points_numpy = new_points.detach().cpu().numpy()
    motion_numpy = new_points_numpy - old_points_numpy
    pred_Y = KMeans(n_clusters=2, random_state=2).fit_predict(motion_numpy)
    if np.sum(pred_Y) > cfg.TRAIN.TOPK / 2:
        pred_Y = 1 - pred_Y
    cluster1_old_points = old_points_numpy[(pred_Y == 0).nonzero()[0], :]
    cluster1_new_points = new_points_numpy[(pred_Y == 0).nonzero()[0], :]

    # pre-warping with global homography
    Homo, _ = cv2.findHomography(cluster1_old_points, cluster1_new_points,
                                 cv2.RANSAC)

    if Homo is None:
        Homo = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    dominator = (
        Homo[2, 0] * old_points_numpy[:, 0]
        + Homo[2, 1] * old_points_numpy[:, 1] + Homo[2, 2])
    new_points_projected = np.stack(
        [(Homo[0, 0] * old_points_numpy[:, 0]
          + Homo[0, 1] * old_points_numpy[:, 1] + Homo[0, 2]) / dominator,
         (Homo[1, 0] * old_points_numpy[:, 0]
          + Homo[1, 1] * old_points_numpy[:, 1] + Homo[1, 2]) / dominator], 1)

    index = (pred_Y == 1).nonzero()[0]
    attribute = np.zeros_like(new_points_numpy[:, 0:1])  # N', 1
    old_points_numpy_chosen = old_points_numpy[index, :]
    new_points_numpy_chosen = new_points_numpy[index, :]

    cluster1_motion = cluster1_new_points - cluster1_old_points
    clsuter2_motion = new_points_numpy_chosen - old_points_numpy_chosen
    cluster1_meanMotion = np.mean(cluster1_motion, 0)
    cluster2_meanMotion = np.mean(clsuter2_motion, 0)
    distanceMeasure = MotionDistanceMeasure(cluster1_meanMotion,
                                            cluster2_meanMotion)

    if np.sum(pred_Y) > cfg.MODEL.THRESHOLDPOINT and distanceMeasure:

        attribute[index, :] = np.expand_dims(np.ones_like(index), 1)

        Homo_2, _ = cv2.findHomography(old_points_numpy_chosen,
                                       new_points_numpy_chosen, cv2.RANSAC)
        if Homo_2 is None:
            Homo_2 = Homo

        meshes_x, meshes_y = np.meshgrid(
            np.arange(0, W, cfg.MODEL.PIXELS),
            np.arange(0, H, cfg.MODEL.PIXELS))

        x_dominator = Homo[0, 0] * meshes_x + \
            Homo[0, 1] * meshes_y + Homo[0, 2]
        y_dominator = Homo[1, 0] * meshes_x + \
            Homo[1, 1] * meshes_y + Homo[1, 2]
        noiminator = Homo[2, 0] * meshes_x + Homo[2, 1] * meshes_y + Homo[2, 2]

        projected_1 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2),
            (-1, 2))

        x_dominator = Homo_2[0, 0] * meshes_x + \
            Homo_2[0, 1] * meshes_y + Homo_2[0, 2]
        y_dominator = Homo_2[1, 0] * meshes_x + \
            Homo_2[1, 1] * meshes_y + Homo_2[1, 2]
        noiminator = Homo_2[2, 0] * meshes_x + \
            Homo_2[2, 1] * meshes_y + Homo_2[2, 2]

        projected_2 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2),
            (-1, 2))

        distance_x = np.expand_dims(new_points_numpy[:, 0], 0) - np.reshape(
            meshes_x, (-1, 1))
        distance_y = np.expand_dims(new_points_numpy[:, 1], 0) - np.reshape(
            meshes_y, (-1, 1))
        distance = distance_x**2 + distance_y**2  # N, N'
        distance_mask = (distance < (cfg.MODEL.RADIUS**2))  # N, N'
        distance_mask_value = (distance_mask.astype(np.float32)
                               * attribute.transpose(1, 0))  # N, N'
        distance = np.sum(distance_mask_value, 1) / \
            (np.sum(distance_mask, 1) + 1e-9)  # N

        project_pos = np.reshape(
            np.expand_dims(distance, 1) * projected_2 + np.expand_dims(
                (1 - distance), 1) * projected_1,
            (cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS,
             cfg.MODEL.WIDTH // cfg.MODEL.PIXELS, 2))

        meshes_projected = torch.from_numpy(project_pos.astype(np.float32)).to(
            new_points.device).permute(2, 0, 1)

        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, W, cfg.MODEL.PIXELS),
            torch.arange(0, H, cfg.MODEL.PIXELS))
        meshes_x = meshes_x.float().permute(1, 0)
        meshes_y = meshes_y.float().permute(1, 0)
        meshes_x = meshes_x.to(old_points.device)
        meshes_y = meshes_y.to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y], 0)

        x_motions = meshes[0, :, :] - meshes_projected[0, :, :]
        y_motions = meshes[1, :, :] - meshes_projected[1, :, :]

        homo_cal = HomoCalc(meshes, meshes_projected)
        project_pts = HomoProj(homo_cal, old_points)
        new_points_projected = project_pts

        Homo = torch.from_numpy(Homo.astype(np.float32)).to(old_points.device)

    else:

        Homo = torch.from_numpy(Homo.astype(np.float32)).to(old_points.device)
        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, W, cfg.MODEL.PIXELS),
            torch.arange(0, H, cfg.MODEL.PIXELS))
        meshes_x = meshes_x.float().permute(1, 0)
        meshes_y = meshes_y.float().permute(1, 0)
        meshes_x = meshes_x.to(old_points.device)
        meshes_y = meshes_y.to(old_points.device)
        meshes_z = torch.ones_like(meshes_x).to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y, meshes_z], 0)

        meshes_projected = torch.mm(Homo, meshes.view(3,
                                                      -1)).view(*meshes.shape)

        x_motions = meshes[0, :, :] - meshes_projected[0, :, :] / (
            meshes_projected[2, :, :])
        y_motions = meshes[1, :, :] - meshes_projected[1, :, :] / (
            meshes_projected[2, :, :])
        new_points_projected = torch.from_numpy(new_points_projected).to(
            old_points.device)

    temp_x_motion = torch.zeros_like(x_motions)
    temp_y_motion = torch.zeros_like(x_motions)

    for i in range(x_motions.shape[0]):
        for j in range(x_motions.shape[1]):
            distance = torch.sqrt((old_points[:, 0] - i * cfg.MODEL.PIXELS)**2
                                  + (old_points[:, 1]
                                     - j * cfg.MODEL.PIXELS)**2)
            distance = distance < cfg.MODEL.RADIUS  # B * topK
            index = distance.nonzero()
            if index.shape[0] == 0:
                continue

            new_points_homo = new_points_projected[index[:, 0].long(), :]

            new_points_flow = new_points[index[:, 0].long(), :]
            temp_motion = -(new_points_homo - new_points_flow)
            temp_x_motion[i, j] = temp_motion[:, 0].median()
            temp_y_motion[i, j] = temp_motion[:, 1].median()

    x_motions = x_motions + temp_x_motion
    y_motions = y_motions + temp_y_motion

    # apply second median filter (f-2) over the motion mesh for outliers
    x_motion_mesh = medfilt(x_motions.unsqueeze(0).unsqueeze(0))
    y_motion_mesh = medfilt(y_motions.unsqueeze(0).unsqueeze(0))

    return torch.cat([x_motion_mesh, y_motion_mesh], 1)
