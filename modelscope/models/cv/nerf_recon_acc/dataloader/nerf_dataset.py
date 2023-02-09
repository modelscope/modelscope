# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os

import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .read_write_model import (read_cameras_binary, read_images_binary,
                               read_points3D_binary)


def get_rays(directions, c2w, keepdim=False):
    assert directions.shape[-1] == 3

    if directions.ndim == 2:
        assert c2w.ndim == 3
        rays_d = (directions[:, None, :] * c2w[:, :3, :3]).sum(-1)
        rays_o = c2w[:, :, 3].expand(rays_d.shape)
    elif directions.ndim == 3:
        if c2w.ndim == 2:
            rays_d = (directions[:, :, None, :]
                      * c2w[None, None, :3, :3]).sum(-1)
            rays_o = c2w[None, None, :, 3].expand(rays_d.shape)
        elif c2w.ndim == 3:
            rays_d = (directions[None, :, :, None, :]
                      * c2w[:, None, None, :3, :3]).sum(-1)
            rays_o = c2w[:, None, None, :, 3].expand(rays_d.shape)

    if not keepdim:
        rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

    return rays_o, rays_d


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy')
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    directions = torch.stack(
        [(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (
        dis > mean - (q75 - q25) * 1.5) & (
            dis < mean + (q75 - q25) * 1.5)
    pts = pts[valid]
    center = pts.mean(0)
    return center, pts


def normalize_poses(poses, pts):
    center, pts = get_center(pts)

    z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    Rc = torch.stack([x, y, z], dim=1)
    tc = center.reshape(3, 1)

    R, t = Rc.T, -Rc.T @ tc

    pose_last = torch.as_tensor([[[0., 0., 0.,
                                   1.]]]).expand(poses.shape[0], -1, -1)
    poses_homo = torch.cat([poses, pose_last], dim=1)
    inv_trans = torch.cat(
        [torch.cat([R, t], dim=1),
         torch.as_tensor([[0., 0., 0., 1.]])], dim=0)

    poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 4, 4)
    scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
    poses_norm[..., 3] /= scale

    pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:, 0:1])],
                                 dim=-1)[..., None])[:, :3, 0]
    pts = pts / scale

    return poses_norm, pts


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.],
                             dtype=cameras.dtype,
                             device=cameras.device)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.],
                         dtype=center.dtype,
                         device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        h = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(h.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(h), p=2, dim=0)
        concat = torch.stack([s, u, -h], dim=1)
        c2w = torch.cat([concat, cam_pos[:, None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w


def to4x4(pose):
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def get_spiral_path(cameras,
                    fx,
                    fy,
                    n_steps=120,
                    radius=0.1,
                    rots=2,
                    zrate=0.5):
    up = cameras[0, :3, 2]
    fx = torch.tensor(fx, dtype=torch.float32)
    fy = torch.tensor(fy, dtype=torch.float32)
    focal = torch.min(fx, fy)
    target = torch.tensor(
        [0, 0, -focal],
        device=cameras.device)  # camera looking in -z direction
    rad = torch.tensor([radius] * 3, device=cameras.device)
    c2w = cameras[0]
    c2wh_global = to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, n_steps + 1)[:-1]:
        theta_list = [
            torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)
        ]
        center = (torch.tensor(theta_list, device=cameras.device) * rad)
        lookat = center - target
        vec2 = F.normalize(lookat, p=2, dim=0)
        vec1_avg = F.normalize(up, p=2, dim=0)
        vec0 = F.normalize(torch.cross(vec1_avg, vec2), p=2, dim=0)
        vec1 = F.normalize(torch.cross(vec2, vec0), p=2, dim=0)
        c2w = torch.stack([vec0, vec1, vec2, center], 1)
        c2wh = to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)
    return new_c2ws


class BlenderDataset(Dataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
        self,
        root_fp,
        split,
        img_wh=(800, 800),
        max_size=None,
        num_rays=None,
        color_bkgd_aug='white',
        near=2.0,
        far=6.0,
        batch_over_images=True,
    ):
        super().__init__()
        self.root_fp = root_fp
        self.split = split
        self.max_size = max_size
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (split == 'train')
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images

        with open(
                os.path.join(self.root_fp, f'transforms_{self.split}.json'),
                'r') as f:
            meta = json.load(f)

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = img_wh

        self.w, self.h = W, H
        self.image_wh = (self.w, self.h)
        self.focal = 0.5 * self.w / math.tan(0.5 * meta['camera_angle_x'])
        self.directions = get_ray_directions(self.w, self.h, self.focal,
                                             self.focal, self.w // 2,
                                             self.h // 2).cuda()

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []

        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4])
            self.all_c2w.append(c2w)

            img_path = os.path.join(self.root_fp, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(img[..., -1])  # (h, w)
            self.all_images.append(img[..., :3])

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().cuda(), \
            torch.stack(self.all_images, dim=0).float().cuda(), \
            torch.stack(self.all_fg_masks, dim=0).float().cuda()

    def __len__(self):
        return len(self.all_images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        return data

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays
        if self.training:
            if self.batch_over_images:
                index = torch.randint(
                    0,
                    len(self.all_images),
                    size=(num_rays, ),
                    device=self.all_images.device)

            else:
                index = torch.randint(
                    0,
                    len(self.all_images),
                    size=(1, ),
                    device=self.all_images.device)

            x = torch.randint(
                0, self.w, size=(num_rays, ), device=self.all_images.device)
            y = torch.randint(
                0, self.h, size=(num_rays, ), device=self.all_images.device)
            c2w = self.all_c2w[index]
            directions = self.directions[y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index, y, x].view(-1,
                                                    self.all_images.shape[-1])
            fg_mask = self.all_fg_masks[index, y, x].view(-1)

        else:
            c2w = self.all_c2w[index]
            directions = self.directions
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index].view(-1, self.all_images.shape[-1])
            fg_mask = self.all_fg_masks[index].view(-1)

        rays = torch.cat([rays_o, rays_d], dim=-1)

        if self.training:
            if self.color_bkgd_aug == 'random':
                color_bkgd = torch.rand(3, device=self.all_images.device)
            elif self.color_bkgd_aug == 'white':
                color_bkgd = torch.ones(3, device=self.all_images.device)
            elif self.color_bkgd_aug == 'black':
                color_bkgd = torch.zeros(3, device=self.all_images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.all_images.device)

        rgb = rgb * fg_mask[..., None] + color_bkgd * (1 - fg_mask[..., None])

        return {
            'pixels': rgb,  # [h*w, 4] or [num_rays, 4]
            'rays': rays,  # [h*w, 6] or [num_rays, 6]
            'fg_mask': fg_mask,
            'image_wh': self.image_wh,
        }


class ColmapDataset(Dataset):
    """data loader for training and evaluation."""

    def __init__(
        self,
        root_fp,
        split,
        img_wh,
        max_size=1200,
        num_rays=None,
        use_mask=True,
        color_bkgd_aug='random',
        batch_over_images=True,
        n_test_traj_steps=120,
    ):
        super().__init__()
        if os.path.exists(os.path.join(root_fp, 'preprocess')):
            self.root_fp = os.path.join(root_fp, 'preprocess')
            self.distort = True
        else:
            self.root_fp = root_fp
            self.distort = False
        self.split = split
        self.num_rays = num_rays
        self.use_mask = use_mask
        self.training = (num_rays is not None) and (split == 'train')
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.n_test_traj_steps = n_test_traj_steps

        if self.distort:
            camdata = read_cameras_binary(
                os.path.join(self.root_fp, 'sparse/cameras.bin'))
        else:
            camdata = read_cameras_binary(
                os.path.join(self.root_fp, 'sparse/0/cameras.bin'))
        H, W = int(camdata[1].height), int(camdata[1].width)

        if img_wh is not None:
            w, h = img_wh
            self.width, self.height = w, h
            self.factor = w / W
        else:
            if H <= max_size and W <= max_size:
                self.height = H
                self.width = W
                self.factor = 1
            else:
                if H > W:
                    self.height = max_size
                    self.width = round(max_size * W / H)
                    self.factor = max_size / H
                else:
                    self.width = max_size
                    self.height = round(max_size * H / W)
                    self.factor = max_size / W
        self.image_wh = (self.width, self.height)
        print('process image width and height: {}'.format(self.image_wh))

        print(camdata[1].model)
        if camdata[1].model == 'SIMPLE_RADIAL':
            fx = fy = camdata[1].params[0] * self.factor
            cx = camdata[1].params[1] * self.factor
            cy = camdata[1].params[2] * self.factor
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0] * self.factor
            fy = camdata[1].params[1] * self.factor
            cx = camdata[1].params[2] * self.factor
            cy = camdata[1].params[3] * self.factor
        else:
            raise ValueError(
                f'Please parse the intrinsics for camera model {camdata[1].model}!'
            )

        self.directions = get_ray_directions(self.width, self.height, fx, fy,
                                             cx, cy).cuda()
        if self.distort:
            imdata = read_images_binary(
                os.path.join(self.root_fp, 'sparse/images.bin'))
        else:
            imdata = read_images_binary(
                os.path.join(self.root_fp, 'sparse/0/images.bin'))

        mask_dir = os.path.join(self.root_fp, 'masks')
        self.use_mask = os.path.exists(mask_dir) and self.use_mask

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        for i, d in enumerate(imdata.values()):
            R = d.qvec2rotmat()
            t = d.tvec.reshape(3, 1)
            c2w = torch.from_numpy(np.concatenate([R.T, -R.T @ t],
                                                  axis=1)).float()
            c2w[:, 1:3] *= -1.
            self.all_c2w.append(c2w)
            if self.split in ['train', 'val']:
                img_path = os.path.join(self.root_fp, 'images', d.name)
                img = Image.open(img_path)
                img = img.resize(self.image_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]
                if self.use_mask:
                    mask_path = os.path.join(mask_dir, d.name)
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize(self.image_wh, Image.BICUBIC)
                    mask = TF.to_tensor(mask)[0]
                else:
                    mask = torch.ones_like(img[..., 0])
                self.all_fg_masks.append(mask)
                self.all_images.append(img)

        self.all_c2w = torch.stack(self.all_c2w, dim=0)

        if self.distort:
            pts3d = read_points3D_binary(
                os.path.join(self.root_fp, 'sparse/points3D.bin'))
        else:
            pts3d = read_points3D_binary(
                os.path.join(self.root_fp, 'sparse/0/points3D.bin'))
        pts3d = torch.from_numpy(np.array([pts3d[k].xyz
                                           for k in pts3d])).float()

        self.all_c2w, pts3d = normalize_poses(self.all_c2w, pts3d)

        if self.split == 'test':
            # self.all_c2w = get_spiral_path(
            #     self.all_c2w, fx, fy, n_steps=self.n_test_traj_steps)
            self.all_c2w = create_spheric_poses(
                self.all_c2w[:, :, 3], n_steps=self.n_test_traj_steps)
            self.all_images = torch.zeros(
                (self.n_test_traj_steps, self.height, self.width, 3),
                dtype=torch.float32)
            self.all_fg_masks = torch.zeros(
                (self.n_test_traj_steps, self.height, self.width),
                dtype=torch.float32)
        else:
            self.all_images, self.all_fg_masks = torch.stack(
                self.all_images, dim=0), torch.stack(
                    self.all_fg_masks, dim=0)

        self.all_c2w, self.all_images, self.all_fg_masks = \
            self.all_c2w.float().cuda(), \
            self.all_images.float().cuda(), \
            self.all_fg_masks.float().cuda()

    def __len__(self):
        return len(self.all_images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        return data

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays
        if self.training:
            if self.batch_over_images:
                index = torch.randint(
                    0,
                    len(self.all_images),
                    size=(num_rays, ),
                    device=self.all_images.device)

            else:
                index = torch.randint(
                    0,
                    len(self.all_images),
                    size=(1, ),
                    device=self.all_images.device)

            x = torch.randint(
                0,
                self.width,
                size=(num_rays, ),
                device=self.all_images.device)
            y = torch.randint(
                0,
                self.height,
                size=(num_rays, ),
                device=self.all_images.device)
            c2w = self.all_c2w[index]
            directions = self.directions[y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index, y, x].view(-1,
                                                    self.all_images.shape[-1])
            fg_mask = self.all_fg_masks[index, y, x].view(-1)

        else:
            c2w = self.all_c2w[index]
            directions = self.directions
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.all_images[index].view(-1, self.all_images.shape[-1])
            fg_mask = self.all_fg_masks[index].view(-1)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if self.training:
            if self.color_bkgd_aug == 'random':
                color_bkgd = torch.rand(3, device=self.all_images.device)
            elif self.color_bkgd_aug == 'white':
                color_bkgd = torch.ones(3, device=self.all_images.device)
            elif self.color_bkgd_aug == 'black':
                color_bkgd = torch.zeros(3, device=self.all_images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.all_images.device)

        rgb = rgb * fg_mask[..., None] + color_bkgd * (1 - fg_mask[..., None])

        return {
            'pixels': rgb,  # [h*w, 4] or [num_rays, 4]
            'rays': rays,  # [h*w, 6] or [num_rays, 6]
            'fg_mask': fg_mask,
            'image_wh': self.image_wh,
        }
