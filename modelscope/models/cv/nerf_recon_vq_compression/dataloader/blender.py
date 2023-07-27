import os

import cv2
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from .ray_utils import *


def trans_t(t):
    return torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t],
                         [0, 0, 0, 1]]).float()


def rot_phi(phi):
    return torch.Tensor([[1, 0, 0, 0], [0, np.cos(phi), -np.sin(phi), 0],
                         [0, np.sin(phi), np.cos(phi), 0], [0, 0, 0,
                                                            1]]).float()


def rot_theta(th):
    return torch.Tensor([[np.cos(th), 0, -np.sin(th), 0], [0, 1, 0, 0],
                         [np.sin(th), 0, np.cos(th), 0], [0, 0, 0,
                                                          1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]
                  ])) @ c2w
    return c2w


class BlenderDataset(Dataset):

    def __init__(self,
                 datadir,
                 split='train',
                 downsample=1.0,
                 is_stack=False,
                 N_vis=-1):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800 / downsample), int(800 / downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                        [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0, 6.0]

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):

        with open(
                os.path.join(self.root_dir, f'transforms_{self.split}.json'),
                'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])
        self.focal *= self.img_wh[0] / 800

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(
            h, w, [self.focal, self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal, 0, w / 2],
                                        [0, self.focal, h / 2], [0, 0,
                                                                 1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample = 1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(
            self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir,
                                      f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])
            self.all_rgbs += [img]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            self.all_rgbs = torch.stack(self.all_rgbs,
                                        0).reshape(-1, *self.img_wh[::-1], 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(
            self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            mask = self.all_masks[idx]  # for quantity evaluation

            sample = {'rays': rays, 'rgbs': img, 'mask': mask}
        return sample

    def get_render_pose(self, N_cameras=120):
        render_poses = torch.stack([
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, N_cameras + 1)[:-1]
        ], 0)
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                                   [0, 0, 0, 1]])
        return render_poses @ torch.Tensor(blender2opencv).float()
