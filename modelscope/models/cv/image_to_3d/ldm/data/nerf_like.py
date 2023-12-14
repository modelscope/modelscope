from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
import imageio
import math
import cv2
from torchvision import transforms

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])


def get_T(T_target, T_cond):
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T

def get_spherical(T_target, T_cond):
    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([math.degrees(d_theta.item()), math.degrees(d_azimuth.item()), d_z.item()])
    return d_T

class RTMV(Dataset):
    def __init__(self, root_dir='datasets/RTMV/google_scanned',\
                 first_K=64, resolution=256, load_target=False):
        self.root_dir = root_dir
        self.scene_list = sorted(next(os.walk(root_dir))[1])
        self.resolution = resolution
        self.first_K = first_K
        self.load_target = load_target

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.root_dir, self.scene_list[idx])
        with open(os.path.join(scene_dir, 'transforms.json'), "r") as f:
            meta = json.load(f)
        imgs = []
        poses = []
        for i_img in range(self.first_K):
            meta_img = meta['frames'][i_img]

            if i_img == 0 or self.load_target:
                img_path = os.path.join(scene_dir, meta_img['file_path'])
                img = imageio.imread(img_path)
                img = cv2.resize(img, (self.resolution, self.resolution), interpolation = cv2.INTER_LINEAR)
                imgs.append(img)
            
            c2w = meta_img['transform_matrix']
            poses.append(c2w)
            
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
        imgs = torch.tensor(self.blend_rgba(imgs)).permute(0, 3, 1, 2)
        imgs = imgs * 2 - 1. # convert to stable diffusion range
        poses = torch.tensor(np.array(poses).astype(np.float32))
        return imgs, poses
                
    def blend_rgba(self, img):
        img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
        return img
            

class GSO(Dataset):
    def __init__(self, root_dir='datasets/GoogleScannedObjects',\
                 split='val', first_K=5, resolution=256, load_target=False, name='render_mvs'):
        self.root_dir = root_dir
        with open(os.path.join(root_dir, '%s.json' % split), "r") as f:
            self.scene_list = json.load(f)
        self.resolution = resolution
        self.first_K = first_K
        self.load_target = load_target
        self.name = name

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.root_dir, self.scene_list[idx])
        with open(os.path.join(scene_dir, 'transforms_%s.json' % self.name), "r") as f:
            meta = json.load(f)
        imgs = []
        poses = []
        for i_img in range(self.first_K):
            meta_img = meta['frames'][i_img]

            if i_img == 0 or self.load_target:
                img_path = os.path.join(scene_dir, meta_img['file_path'])
                img = imageio.imread(img_path)
                img = cv2.resize(img, (self.resolution, self.resolution), interpolation = cv2.INTER_LINEAR)
                imgs.append(img)
            
            c2w = meta_img['transform_matrix']
            poses.append(c2w)
            
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
        mask = imgs[:, :, :, -1]
        imgs = torch.tensor(self.blend_rgba(imgs)).permute(0, 3, 1, 2)
        imgs = imgs * 2 - 1. # convert to stable diffusion range
        poses = torch.tensor(np.array(poses).astype(np.float32))
        return imgs, poses
                
    def blend_rgba(self, img):
        img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
        return img
             
class WILD(Dataset):
    def __init__(self, root_dir='data/nerf_wild',\
                 first_K=33, resolution=256, load_target=False):
        self.root_dir = root_dir
        self.scene_list = sorted(next(os.walk(root_dir))[1])
        self.resolution = resolution
        self.first_K = first_K
        self.load_target = load_target

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, idx):
        scene_dir = os.path.join(self.root_dir, self.scene_list[idx])
        with open(os.path.join(scene_dir, 'transforms_train.json'), "r") as f:
            meta = json.load(f)
        imgs = []
        poses = []
        for i_img in range(self.first_K):
            meta_img = meta['frames'][i_img]

            if i_img == 0 or self.load_target:
                img_path = os.path.join(scene_dir, meta_img['file_path'])
                img = imageio.imread(img_path + '.png')
                img = cv2.resize(img, (self.resolution, self.resolution), interpolation = cv2.INTER_LINEAR)
                imgs.append(img)
            
            c2w = meta_img['transform_matrix']
            poses.append(c2w)
            
        imgs = (np.array(imgs) / 255.).astype(np.float32)  # (RGBA) imgs
        imgs = torch.tensor(self.blend_rgba(imgs)).permute(0, 3, 1, 2)
        imgs = imgs * 2 - 1. # convert to stable diffusion range
        poses = torch.tensor(np.array(poses).astype(np.float32))
        return imgs, poses
                
    def blend_rgba(self, img):
        img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])  # blend A to RGB
        return img