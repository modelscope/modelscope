# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import math

import cv2
import numpy as np
import torch

from ..DUT.config import cfg


def HomoCalc(grids, new_grids_loc):
    """
    @param: grids the location of origin grid vertices [2, H, W]
    @param: new_grids_loc the location of desired grid vertices [2, H, W]

    @return: homo_t homograph projection matrix for each grid [3, 3, H-1, W-1]
    """

    _, H, W = grids.shape

    new_grids = new_grids_loc.unsqueeze(0)

    Homo = torch.zeros(1, 3, 3, H - 1, W - 1).to(grids.device)

    grids = grids.unsqueeze(0)

    try:
        # for common cases if all the homograph can be calculated
        one = torch.ones_like(grids[:, 0:1, :-1, :-1], device=grids.device)
        zero = torch.zeros_like(grids[:, 1:2, :-1, :-1], device=grids.device)

        A = torch.cat(
            [
                torch.stack([
                    grids[:, 0:1, :-1, :-1], grids[:, 1:2, :-1, :-1], one,
                    zero, zero, zero,
                    -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 0:1, :-1, :-1],
                    -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 0:1, :-1, :-1]
                ], 2),  # 1, 1, 8, h-1, w-1
                torch.stack([
                    grids[:, 0:1, 1:, :-1], grids[:, 1:2, 1:, :-1], one, zero,
                    zero, zero,
                    -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 0:1, 1:, :-1],
                    -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 0:1, 1:, :-1]
                ], 2),
                torch.stack([
                    grids[:, 0:1, :-1, 1:], grids[:, 1:2, :-1,
                                                  1:], one, zero, zero, zero,
                    -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 0:1, :-1, 1:],
                    -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 0:1, :-1, 1:]
                ], 2),
                torch.stack([
                    grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:,
                                                 1:], one, zero, zero, zero,
                    -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 0:1, 1:, 1:],
                    -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 0:1, 1:, 1:]
                ], 2),
                torch.stack([
                    zero, zero, zero, grids[:, 0:1, :-1, :-1],
                    grids[:, 1:2, :-1, :-1], one,
                    -1 * grids[:, 0:1, :-1, :-1] * new_grids[:, 1:2, :-1, :-1],
                    -1 * grids[:, 1:2, :-1, :-1] * new_grids[:, 1:2, :-1, :-1]
                ], 2),
                torch.stack([
                    zero, zero, zero, grids[:, 0:1, 1:, :-1],
                    grids[:, 1:2, 1:, :-1], one,
                    -1 * grids[:, 0:1, 1:, :-1] * new_grids[:, 1:2, 1:, :-1],
                    -1 * grids[:, 1:2, 1:, :-1] * new_grids[:, 1:2, 1:, :-1]
                ], 2),
                torch.stack([
                    zero, zero, zero, grids[:, 0:1, :-1,
                                            1:], grids[:, 1:2, :-1, 1:], one,
                    -1 * grids[:, 0:1, :-1, 1:] * new_grids[:, 1:2, :-1, 1:],
                    -1 * grids[:, 1:2, :-1, 1:] * new_grids[:, 1:2, :-1, 1:]
                ], 2),
                torch.stack([
                    zero, zero, zero, grids[:, 0:1, 1:, 1:], grids[:, 1:2, 1:,
                                                                   1:], one,
                    -1 * grids[:, 0:1, 1:, 1:] * new_grids[:, 1:2, 1:, 1:],
                    -1 * grids[:, 1:2, 1:, 1:] * new_grids[:, 1:2, 1:, 1:]
                ], 2),
            ],
            1).view(8, 8, -1).permute(2, 0, 1)  # 1, 8, 8, h-1, w-1
        B_ = torch.stack([
            new_grids[:, 0, :-1, :-1],
            new_grids[:, 0, 1:, :-1],
            new_grids[:, 0, :-1, 1:],
            new_grids[:, 0, 1:, 1:],
            new_grids[:, 1, :-1, :-1],
            new_grids[:, 1, 1:, :-1],
            new_grids[:, 1, :-1, 1:],
            new_grids[:, 1, 1:, 1:],
        ], 1).view(8, -1).permute(
            1, 0)  # B, 8, h-1, w-1 ==> A @ H = B ==> H = A^-1 @ B
        A_inverse = torch.inverse(A)
        # B, 8, 8 @ B, 8, 1 --> B, 8, 1
        H_recovered = torch.bmm(A_inverse, B_.unsqueeze(2))

        H_ = torch.cat([
            H_recovered,
            torch.ones_like(H_recovered[:, 0:1, :], device=H_recovered.device)
        ], 1).view(H_recovered.shape[0], 3, 3)

        H_ = H_.permute(1, 2, 0)
        H_ = H_.view(Homo.shape)
        Homo = H_
    except Exception:
        # if some of the homography can not be calculated
        one = torch.ones_like(grids[:, 0:1, 0, 0], device=grids.device)
        zero = torch.zeros_like(grids[:, 1:2, 0, 0], device=grids.device)
        H_ = torch.eye(3, device=grids.device)
        for i in range(H - 1):
            for j in range(W - 1):
                A = torch.cat([
                    torch.stack([
                        grids[:, 0:1, i, j], grids[:, 1:2, i,
                                                   j], one, zero, zero, zero,
                        -1 * grids[:, 0:1, i, j] * new_grids[:, 0:1, i, j],
                        -1 * grids[:, 1:2, i, j] * new_grids[:, 0:1, i, j]
                    ], 2),
                    torch.stack([
                        grids[:, 0:1, i + 1, j], grids[:, 1:2, i + 1, j], one,
                        zero, zero, zero, -1 * grids[:, 0:1, i + 1, j]
                        * new_grids[:, 0:1, i + 1, j], -1
                        * grids[:, 1:2, i + 1, j] * new_grids[:, 0:1, i + 1, j]
                    ], 2),
                    torch.stack([
                        grids[:, 0:1, i, j + 1], grids[:, 1:2, i, j + 1], one,
                        zero, zero, zero, -1 * grids[:, 0:1, i, j + 1]
                        * new_grids[:, 0:1, i, j + 1], -1
                        * grids[:, 1:2, i, j + 1] * new_grids[:, 0:1, i, j + 1]
                    ], 2),
                    torch.stack([
                        grids[:, 0:1, i + 1, j + 1], grids[:, 1:2, i + 1,
                                                           j + 1], one, zero,
                        zero, zero, -1 * grids[:, 0:1, i + 1, j + 1]
                        * new_grids[:, 0:1, i + 1, j + 1],
                        -1 * grids[:, 1:2, i + 1, j + 1]
                        * new_grids[:, 0:1, i + 1, j + 1]
                    ], 2),
                    torch.stack([
                        zero, zero, zero, grids[:, 0:1, i, j], grids[:, 1:2, i,
                                                                     j], one,
                        -1 * grids[:, 0:1, i, j] * new_grids[:, 1:2, i, j],
                        -1 * grids[:, 1:2, i, j] * new_grids[:, 1:2, i, j]
                    ], 2),
                    torch.stack([
                        zero, zero, zero, grids[:, 0:1, i + 1,
                                                j], grids[:, 1:2, i + 1, j],
                        one, -1 * grids[:, 0:1, i + 1, j]
                        * new_grids[:, 1:2, i + 1, j], -1
                        * grids[:, 1:2, i + 1, j] * new_grids[:, 1:2, i + 1, j]
                    ], 2),
                    torch.stack([
                        zero, zero, zero, grids[:, 0:1, i, j + 1],
                        grids[:, 1:2, i,
                              j + 1], one, -1 * grids[:, 0:1, i, j + 1]
                        * new_grids[:, 1:2, i, j + 1], -1
                        * grids[:, 1:2, i, j + 1] * new_grids[:, 1:2, i, j + 1]
                    ], 2),
                    torch.stack([
                        zero, zero, zero, grids[:, 0:1, i + 1, j + 1],
                        grids[:, 1:2, i + 1,
                              j + 1], one, -1 * grids[:, 0:1, i + 1, j + 1]
                        * new_grids[:, 1:2, i + 1, j + 1],
                        -1 * grids[:, 1:2, i + 1, j + 1]
                        * new_grids[:, 1:2, i + 1, j + 1]
                    ], 2),
                ], 1)  # B, 8, 8
                B_ = torch.stack([
                    new_grids[:, 0, i, j],
                    new_grids[:, 0, i + 1, j],
                    new_grids[:, 0, i, j + 1],
                    new_grids[:, 0, i + 1, j + 1],
                    new_grids[:, 1, i, j],
                    new_grids[:, 1, i + 1, j],
                    new_grids[:, 1, i, j + 1],
                    new_grids[:, 1, i + 1, j + 1],
                ], 1)  # B, 8 ==> A @ H = B ==> H = A^-1 @ B
                try:
                    A_inverse = torch.inverse(A)

                    # B, 8, 8 @ B, 8, 1 --> B, 8, 1
                    H_recovered = torch.bmm(A_inverse, B_.unsqueeze(2))

                    H_ = torch.cat([
                        H_recovered,
                        torch.ones_like(H_recovered[:, 0:1, :]).to(
                            H_recovered.device)
                    ], 1).view(H_recovered.shape[0], 3, 3)
                except Exception:
                    pass
                Homo[:, :, :, i, j] = H_

    homo_t = Homo.view(3, 3, H - 1, W - 1)

    return homo_t


def HomoProj(homo, pts):
    """
    @param: homo [3, 3, G_H-1, G_W-1]
    @param: pts  [N, 2(W, H)] - [:, 0] for width and [:, 1] for height

    @return: projected pts [N, 2(W, H)] - [:, 0] for width and [:, 1] for height
    """

    # pts_location_x = (pts[:, 0:1] // cfg.MODEL.PIXELS).long()
    # pts_location_y = (pts[:, 1:2] // cfg.MODEL.PIXELS).long()
    pts_location_x = torch.div(
        pts[:, 0:1], cfg.MODEL.PIXELS, rounding_mode='floor').long()
    pts_location_y = torch.div(
        pts[:, 1:2], cfg.MODEL.PIXELS, rounding_mode='floor').long()

    # if the grid is outside of the image
    maxWidth = cfg.MODEL.WIDTH // cfg.MODEL.PIXELS - 1
    maxHeight = cfg.MODEL.HEIGHT // cfg.MODEL.PIXELS - 1
    index = (pts_location_x[:, 0] >= maxWidth).nonzero().long()
    pts_location_x[index, :] = maxWidth - 1
    index = (pts_location_y[:, 0] >= maxHeight).nonzero().long()
    pts_location_y[index, :] = maxHeight - 1

    homo = homo.to(pts.device)

    # calculate the projection
    x_dominator = pts[:, 0] * homo[0, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[0, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[0, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    y_dominator = pts[:, 0] * homo[1, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[1, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[1, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    noiminator = pts[:, 0] * homo[2, 0, pts_location_y[:, 0], pts_location_x[:, 0]] + pts[:, 1] * \
        homo[2, 1, pts_location_y[:, 0], pts_location_x[:, 0]] + homo[2, 2, pts_location_y[:, 0], pts_location_x[:, 0]]
    noiminator = noiminator

    new_kp_x = x_dominator / noiminator
    new_kp_y = y_dominator / noiminator

    return torch.stack([new_kp_x, new_kp_y], 1)


def multiHomoEstimate(motion, kp):
    """
    @param: motion [4, N]
    @param: kp     [2, N]
    """

    from sklearn.cluster import KMeans

    new_kp = torch.cat([kp[1:2, :], kp[0:1, :]], 0) + motion[2:, :]
    new_points_numpy = new_kp.cpu().detach().numpy().transpose(1, 0)
    old_points = torch.stack([kp[1, :], kp[0, :]], 1).to(motion.device)
    old_points_numpy = torch.cat([kp[1:2, :], kp[0:1, :]],
                                 0).cpu().detach().numpy().transpose(1, 0)
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
    new_points_projected = torch.from_numpy(
        np.stack(
            [(Homo[0, 0] * old_points_numpy[:, 0]
              + Homo[0, 1] * old_points_numpy[:, 1] + Homo[0, 2]) / dominator,
             (Homo[1, 0] * old_points_numpy[:, 0]
              + Homo[1, 1] * old_points_numpy[:, 1] + Homo[1, 2]) / dominator],
            1).astype(np.float32)).to(old_points.device).permute(1, 0)

    index = (pred_Y == 1).nonzero()[0]
    attribute = np.zeros_like(new_points_numpy[:, 0:1])  # N', 1
    cluster2_old_points = old_points_numpy[index, :]
    cluster2_new_points = new_points_numpy[index, :]
    attribute[index, :] = np.expand_dims(np.ones_like(index), 1)

    cluster1_motion = cluster1_new_points - cluster1_old_points
    clsuter2_motion = cluster2_new_points - cluster2_old_points
    cluster1_meanMotion = np.mean(cluster1_motion, 0)
    cluster2_meanMotion = np.mean(clsuter2_motion, 0)
    distanceMeasure = MotionDistanceMeasure(cluster1_meanMotion,
                                            cluster2_meanMotion)

    threhold = (np.sum(pred_Y) > cfg.MODEL.THRESHOLDPOINT) and distanceMeasure

    if threhold:

        Homo_2, _ = cv2.findHomography(cluster2_old_points,
                                       cluster2_new_points, cv2.RANSAC)
        if Homo_2 is None:
            Homo_2 = Homo

        meshes_x, meshes_y = np.meshgrid(
            np.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
            np.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS))

        # Use first cluster to do projection
        x_dominator = Homo[0, 0] * meshes_x + \
            Homo[0, 1] * meshes_y + Homo[0, 2]
        y_dominator = Homo[1, 0] * meshes_x + \
            Homo[1, 1] * meshes_y + Homo[1, 2]
        noiminator = Homo[2, 0] * meshes_x + Homo[2, 1] * meshes_y + Homo[2, 2]

        projected_1 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2),
            (-1, 2))

        # Use second cluster to do projection
        x_dominator = Homo_2[0, 0] * meshes_x + \
            Homo_2[0, 1] * meshes_y + Homo_2[0, 2]
        y_dominator = Homo_2[1, 0] * meshes_x + \
            Homo_2[1, 1] * meshes_y + Homo_2[1, 2]
        noiminator = Homo_2[2, 0] * meshes_x + \
            Homo_2[2, 1] * meshes_y + Homo_2[2, 2]

        projected_2 = np.reshape(
            np.stack([x_dominator / noiminator, y_dominator / noiminator], 2),
            (-1, 2))

        # Determine use which projected position
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
            old_points.device).permute(2, 0, 1)

        # calculate reference location for each keypoint
        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
            torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS))
        meshes_x = meshes_x.float().permute(1, 0)
        meshes_y = meshes_y.float().permute(1, 0)
        meshes_x = meshes_x.to(old_points.device)
        meshes_y = meshes_y.to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y], 0)

        x_motions = meshes[0, :, :] - \
            meshes_projected[0, :, :]
        y_motions = meshes[1, :, :] - meshes_projected[1, :, :]

        homo_cal = HomoCalc(meshes, meshes_projected)
        project_pts = HomoProj(homo_cal, old_points)
        new_points_projected = project_pts.to(old_points.device).permute(1, 0)

    else:
        Homo = torch.from_numpy(Homo.astype(np.float32)).to(old_points.device)
        meshes_x, meshes_y = torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
            torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS))
        meshes_x = meshes_x.float().permute(1, 0)
        meshes_y = meshes_y.float().permute(1, 0)
        meshes_x = meshes_x.to(old_points.device)
        meshes_y = meshes_y.to(old_points.device)
        meshes_z = torch.ones_like(meshes_x).to(old_points.device)
        meshes = torch.stack([meshes_x, meshes_y, meshes_z], 0)
        meshes_projected = torch.mm(Homo, meshes.view(3,
                                                      -1)).view(*meshes.shape)

        x_motions = meshes[0, :, :] - meshes_projected[0, :, :] / \
            (meshes_projected[2, :, :])
        y_motions = meshes[1, :, :] - meshes_projected[1, :, :] / \
            (meshes_projected[2, :, :])

    grids = torch.stack(
        torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
            torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)),
        0).to(motion.device).permute(0, 2, 1).reshape(2, -1).permute(1, 0)

    grids = grids.unsqueeze(2).float()  # N', 2, 1
    projected_motion = torch.stack([x_motions, y_motions],
                                   2).view(-1, 2,
                                           1).to(motion.device)  # G_H, G_W, 2

    redisual_kp_motion = new_points_projected - torch.cat(
        [kp[1:2, :], kp[0:1, :]], 0)

    motion[:2, :] = motion[:2, :] + motion[2:, :]
    motion = motion.unsqueeze(0).repeat(grids.shape[0], 1, 1)  # N', 4, N
    motion[:, :2, :] = (motion[:, :2, :] - grids) / cfg.MODEL.WIDTH
    origin_motion = motion[:, 2:, :] / cfg.MODEL.FLOWC
    motion[:, 2:, :] = (redisual_kp_motion.unsqueeze(0)
                        - motion[:, 2:, :]) / cfg.MODEL.FLOWC

    return motion, projected_motion / cfg.MODEL.FLOWC, origin_motion


def singleHomoEstimate(motion, kp):
    """
    @param: motion [4, N]
    @param: kp     [2, N]
    """
    new_kp = torch.cat([kp[1:2, :], kp[0:1, :]], 0) + motion[2:, :]
    new_points_numpy = new_kp.cpu().detach().numpy().transpose(1, 0)
    old_points = torch.stack([kp[1, :], kp[0, :]], 1).to(motion.device)
    old_points_numpy = torch.cat([kp[1:2, :], kp[0:1, :]],
                                 0).cpu().detach().numpy().transpose(1, 0)

    cluster1_old_points = old_points_numpy
    cluster1_new_points = new_points_numpy

    # pre-warping with global homography
    Homo, _ = cv2.findHomography(cluster1_old_points, cluster1_new_points,
                                 cv2.RANSAC)

    if Homo is None:
        Homo = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    dominator = (
        Homo[2, 0] * old_points_numpy[:, 0]
        + Homo[2, 1] * old_points_numpy[:, 1] + Homo[2, 2])
    new_points_projected = torch.from_numpy(
        np.stack(
            [(Homo[0, 0] * old_points_numpy[:, 0]
              + Homo[0, 1] * old_points_numpy[:, 1] + Homo[0, 2]) / dominator,
             (Homo[1, 0] * old_points_numpy[:, 0]
              + Homo[1, 1] * old_points_numpy[:, 1] + Homo[1, 2]) / dominator],
            1).astype(np.float32)).to(old_points.device).permute(1, 0)

    Homo = torch.from_numpy(Homo.astype(np.float32)).to(
        old_points.device)  # 3, 3
    meshes_x, meshes_y = torch.meshgrid(
        torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
        torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS))
    meshes_x = meshes_x.float().permute(1, 0)
    meshes_y = meshes_y.float().permute(1, 0)
    meshes_x = meshes_x.to(old_points.device)
    meshes_y = meshes_y.to(old_points.device)
    meshes_z = torch.ones_like(meshes_x).to(old_points.device)
    meshes = torch.stack([meshes_x, meshes_y, meshes_z],
                         0)  # 3, H // PIXELS, W // PIXELS
    meshes_projected = torch.mm(Homo, meshes.view(3, -1)).view(*meshes.shape)
    x_motions = meshes[0, :, :] - meshes_projected[0, :, :] / \
        (meshes_projected[2, :, :])  # H//PIXELS, W//PIXELS
    y_motions = meshes[1, :, :] - \
        meshes_projected[1, :, :] / (meshes_projected[2, :, :])

    grids = torch.stack(
        torch.meshgrid(
            torch.arange(0, cfg.MODEL.WIDTH, cfg.MODEL.PIXELS),
            torch.arange(0, cfg.MODEL.HEIGHT, cfg.MODEL.PIXELS)),
        0).to(motion.device).permute(0, 2, 1).reshape(2, -1).permute(
            1, 0)  # 2, W, H --> 2, H, W --> 2, N'

    grids = grids.unsqueeze(2).float()  # N', 2, 1
    projected_motion = torch.stack([x_motions, y_motions],
                                   2).view(-1, 2,
                                           1).to(motion.device)  # G_H, G_W, 2

    redisual_kp_motion = new_points_projected - torch.cat(
        [kp[1:2, :], kp[0:1, :]], 0)

    # to kp_flow (kp(t)) location
    motion[:2, :] = motion[:2, :] + motion[2:, :]
    motion = motion.unsqueeze(0).repeat(grids.shape[0], 1, 1)  # N', 4, N
    motion[:, :2, :] = (motion[:, :2, :] - grids) / cfg.MODEL.WIDTH
    origin_motion = motion[:, 2:, :] / cfg.MODEL.FLOWC
    motion[:, 2:, :] = (redisual_kp_motion.unsqueeze(0)
                        - motion[:, 2:, :]) / cfg.MODEL.FLOWC

    return motion, projected_motion / cfg.MODEL.FLOWC, origin_motion


def f_rot(x):
    return math.atan2(x[1], x[0]) / math.pi * 180


def MotionDistanceMeasure(motion1, motion2):
    """
    MotionDistanceMeasure
    @params motion1 np.array(2) (w, h)
    @params motion2 np.array(2) (w, h)

    @return bool describe whether the two motion are close or not, True for far and False for close
    """

    mangnitue_motion1 = np.sqrt(np.sum(motion1**2))
    mangnitue_motion2 = np.sqrt(np.sum(motion2**2))
    diff_mangnitude = np.abs(mangnitue_motion1 - mangnitue_motion2)

    rot_motion1 = f_rot(motion1)
    rot_motion2 = f_rot(motion2)
    diff_rot = np.abs(rot_motion1 - rot_motion2)
    if diff_rot > 180:
        diff_rot = 360 - diff_rot

    temp_value_12 = (diff_mangnitude >= cfg.Threshold.MANG)
    temp_value_13 = (diff_rot >= cfg.Threshold.ROT)

    return temp_value_12 or temp_value_13
