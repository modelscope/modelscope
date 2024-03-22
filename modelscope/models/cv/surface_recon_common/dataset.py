# ------------------------------------------------------------------------
# Modified from https://github.com/Totoro97/NeuS/blob/main/models/dataset.py
# Copyright (c) 2021 Peng Wang. All Rights Reserved.
# ------------------------------------------------------------------------

import os
from glob import glob

import cv2 as cv
import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(' ') for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:

    def __init__(self, data_dir, device):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = device
        self.data_dir = data_dir
        print('data_dir: ', self.data_dir)

        camera_dict = np.load(
            os.path.join(self.data_dir, 'cameras_sphere.npz'))
        self.camera_dict = camera_dict
        self.images_lis = sorted(
            glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        print('found %d images' % self.n_images)

        self.world_mats_np = [
            camera_dict['world_mat_%d' % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]
        self.scale_mats_np = [
            camera_dict['scale_mat_%d' % idx].astype(np.float32)
            for idx in range(self.n_images)
        ]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np,
                                        self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(
            self.device)  # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(
            self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(
            self.device)  # [n_images, 4, 4]

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([1.01, 1.01, 1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(
            os.path.join(self.data_dir, 'cameras_sphere.npz'))['scale_mat_0']
        object_bbox_min = np.linalg.inv(
            self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:,
                                                                        None]
        object_bbox_max = np.linalg.inv(
            self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:,
                                                                        None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        level = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // level)
        ty = torch.linspace(0, self.H - 1, self.H // level)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3],
                              rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3,
                               3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_o_at(self, img_idx):
        """
        Generate rays_o at world space from one camera.
        """
        rays_o = self.pose_all[img_idx, :3, 3]
        return rays_o

    # add
    def gen_rays_at_camera(self, pose, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        level = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // level)
        ty = torch.linspace(0, self.H - 1, self.H // level)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[:3, :3], rays_v[:, :, :,
                                                   None]).squeeze()  # W, H, 3
        rays_o = pose[:3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])  # bs
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])  # bs
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3

        depth = self.depths[img_idx][(pixels_y, pixels_x)]  # batch_size, 1

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)],
            dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3],
                         p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3],
                              rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3,
                               3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat(
            [rays_o.cpu(),
             rays_v.cpu(), color, mask[:, :1], depth[:, None]],
            dim=-1).cuda()  # batch_size, 10

    def gen_random_rays_at_mask(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])  # bs
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])  # bs
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3

        depth = self.depths[img_idx][(pixels_y, pixels_x)]  # batch_size, 1

        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)],
            dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3],
                         p[:, :, None]).squeeze()  # batch_size, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3],
                              rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3,
                               3].expand(rays_v.shape)  # batch_size, 3
        return torch.cat(
            [rays_o.cpu(),
             rays_v.cpu(), color, mask[:, :1], depth[:, None]],
            dim=-1).cuda()  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        level = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // level)
        ty = torch.linspace(0, self.H - 1, self.H // level)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (
            1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3],
                              rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), pose

    def gen_rays_across(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        level = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // level)
        ty = torch.linspace(0, self.H - 1, self.H // level)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3],
                         p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(
            p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (
            1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3],
                              rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), pose

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level,
                                self.H // resolution_level))).clip(0, 255)
