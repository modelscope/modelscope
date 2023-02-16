# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import random
import re
import time
from collections import OrderedDict
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .dataloader.nerf_dataset import BlenderDataset, ColmapDataset
from .network.nerf import NeRFModel
from .network.segmenter import ObjectSegmenter
from .network.utils import PSNR

logger = get_logger()

__all__ = ['NeRFReconAcc']


@MODELS.register_module(
    Tasks.nerf_recon_acc, module_name=Models.nerf_recon_acc)
class NeRFReconAcc(TorchModel):

    def __init__(self, model_dir, data_type, use_mask, network_cfg, **kwargs):
        """initialize the acceleration version of nerf reconstruction model for object.

        Args:
            model_dir (str): the model path.
            data_type (str): default is 'colmap'
            use_mask (bool): whether use mask of objects, default True
            max_step (int): max train steps, default 30000
            train_num_rays (int): init number of rays in training, default 256
            num_samples_per_ray (int): sampling numbers for each ray, default 1024
            max_train_num_rays (int): max number of rays in training, default 8192
            test_ray_chunk (int): chunk size for rendering, default 1024
            dynamic_ray_sampling (bool): whether use dynamic ray sampling when training, default True
            max_size (int): max size of (width, height) when training, default 800
            n_test_traj_steps (int): number of testing images, default 120
            log_every_n_steps (int): print log info every n steps, default 1000
            save_mesh (bool): whether to save the reconstructed mesh of object, default False
            save_ckpt (bool): whether to save the checkpoints in data_dir, default False
            network_cfg (dict): args of network config
        """
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')

        self.data_type = data_type
        self.use_mask = use_mask
        self.max_step = kwargs['max_step']
        self.train_num_rays = kwargs['train_num_rays']
        self.num_samples_per_ray = kwargs['num_samples_per_ray']
        self.train_num_samples = self.train_num_rays * self.num_samples_per_ray
        self.max_train_num_rays = kwargs['max_train_num_rays']
        self.dynamic_ray_sampling = kwargs['dynamic_ray_sampling']

        self.log_every_n_steps = kwargs['log_every_n_steps']
        self.save_mesh = kwargs['save_mesh']
        self.save_ckpt = kwargs['save_ckpt']

        if self.use_mask:
            segment_path = os.path.join(model_dir, 'matting.pb')
            self.segmenter = ObjectSegmenter(segment_path)

        if self.data_type == 'blender':
            self.img_wh = (800, 800)
            network_cfg['radius'] = 1.5
            self.background = 'white'
            network_cfg['background'] = 'white'
        elif self.data_type == 'colmap':
            self.img_wh = None
            self.max_size = kwargs['max_size']
            self.n_test_traj_steps = kwargs['n_test_traj_steps']
            network_cfg['radius'] = 0.5
            if self.use_mask:
                self.background = 'white'
                network_cfg['background'] = 'white'
                logger.info('run nerf with mask data')
            else:
                self.background = 'random'
                network_cfg['background'] = 'random'
                logger.info('run nerf without mask data')
        logger.info(network_cfg)

        self.model = NeRFModel(network_cfg, **kwargs).cuda()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, eps=1e-15)
        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[
                self.max_step // 2, self.max_step * 3 // 4,
                self.max_step * 9 // 10
            ],
            gamma=0.33,
        )
        self.criterions = PSNR()
        self.set_random_seed(42)

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @torch.enable_grad()
    def nerf_reconstruction(self, data_dir):
        if os.path.exists(os.path.join(data_dir, 'preprocess')):
            use_distortion = True
        else:
            use_distortion = False
        if self.use_mask:
            if use_distortion:
                image_dir = os.path.join(data_dir, 'preprocess/images')
                save_mask_dir = os.path.join(data_dir, 'preprocess/masks')
            else:
                image_dir = os.path.join(data_dir, 'images')
                save_mask_dir = os.path.join(data_dir, 'masks')
            os.makedirs(save_mask_dir, exist_ok=True)
            img_list = glob.glob('{}/*.*g'.format(image_dir)) + glob.glob(
                '{}/*.*G'.format(image_dir))
            for img_path in img_list:
                img = cv2.imread(img_path)
                mask = self.segmenter.run_mask(img)
                outpath = os.path.join(save_mask_dir,
                                       os.path.basename(img_path))
                cv2.imwrite(outpath, mask)
            logger.info('segment images done!')

        if self.data_type == 'blender':
            self.train_dataset = BlenderDataset(
                root_fp=data_dir,
                split='train',
                img_wh=self.img_wh,
                num_rays=self.train_num_rays,
                color_bkgd_aug=self.background,
            )

            self.test_dataset = BlenderDataset(
                root_fp=data_dir,
                split='test',
                img_wh=self.img_wh,
                num_rays=self.train_num_rays,
            )

        elif self.data_type == 'colmap':
            self.train_dataset = ColmapDataset(
                root_fp=data_dir,
                split='train',
                img_wh=self.img_wh,
                max_size=self.max_size,
                num_rays=self.train_num_rays,
                color_bkgd_aug=self.background,
            )

            self.test_dataset = ColmapDataset(
                root_fp=data_dir,
                split='test',
                img_wh=self.img_wh,
                max_size=self.max_size,
                num_rays=self.train_num_rays,
                n_test_traj_steps=self.n_test_traj_steps,
            )

        step = 0
        tic = time.time()
        while step < self.max_step:
            for i in range(len(self.train_dataset)):
                self.model.train()
                data = self.train_dataset[i]
                self.model.update_step(step)
                rays = data['rays'].cuda()
                pixels = data['pixels'].cuda()

                out = self.model(rays)

                if out['num_samples'] == 0:
                    continue

                loss = 0.

                if self.dynamic_ray_sampling:
                    temp = self.train_num_samples / sum(out['num_samples'])
                    train_num_rays = int(self.train_num_rays * temp)
                    self.train_num_rays = min(
                        int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                        self.max_train_num_rays)

                self.train_dataset.update_num_rays(self.train_num_rays)
                loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid']],
                                            pixels[out['rays_valid']])
                loss += loss_rgb
                psnr = self.criterions(out['comp_rgb'], pixels)
                self.optimizer.zero_grad()
                self.grad_scaler.scale(loss).backward()
                self.optimizer.step()
                self.scheduler.step()

                if step % self.log_every_n_steps == 0:

                    elapsed_time = time.time() - tic
                    logger.info(
                        f'elapsed_time={elapsed_time:.2f}s | step={step} | '
                        f'loss={loss:.4f} | '
                        f'train/num_rays={self.train_num_rays:d} |'
                        f'PSNR={psnr:.4f} ')

                step += 1

        save_video_path = os.path.join(data_dir, 'render.mp4')
        self.render_video(data_dir, save_video_path)
        if self.save_ckpt:
            save_ckpt_dir = os.path.join(data_dir, 'ckpt')
            os.makedirs(save_ckpt_dir, exist_ok=True)
            torch.save(
                {
                    'global_step': self.max_step,
                    'network_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(save_ckpt_dir, '{}.ckpt'.format(step)))
            logger.info('save checkpoints done')

        logger.info('reconstruction finish')
        return save_video_path

    def render_video(self, data_dir, save_video_path):
        self.model.eval()
        with torch.no_grad():
            psnr = 0
            for i in tqdm.tqdm(range(len(self.test_dataset))):

                data = self.test_dataset[i]
                rays = data['rays'].cuda()
                pixels = data['pixels'].cuda()
                image_wh = data['image_wh']
                out = self.model.inference(rays)

                psnr += self.criterions(out['comp_rgb'], pixels)

                W, H = image_wh
                img = out['comp_rgb'].view(H, W, 3)
                save_img_dir = os.path.join(data_dir, 'render')
                os.makedirs(save_img_dir, exist_ok=True)
                save_img_path = os.path.join(save_img_dir, f'{i:d}.png')
                self.save_image(save_img_path, img)

            self.save_video(save_video_path, save_img_dir)
            logger.info('test psnr: {}'.format(psnr / len(self.test_dataset)))
            logger.info('save render video done.')

            if self.save_mesh:
                mesh = self.model.isosurface()
                save_mesh_path = os.path.join(data_dir, 'out.obj')
                self.save_obj(save_mesh_path, mesh['v_pos'], mesh['t_pos_idx'])

    def save_image(self, filename, img):
        img = img.clip(0, 1).cpu().numpy()
        img = (img * 255.).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(filename, img)

    def save_video(self, filename, img_dir, fps=20):
        img_paths = glob.glob('{}/*.png'.format(img_dir))
        img_paths = sorted(
            img_paths, key=lambda f: int(os.path.basename(f)[:-4]))
        imgs = [cv2.imread(f) for f in img_paths]

        H, W, _ = imgs[0].shape
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (W, H), True)
        for img in imgs:
            writer.write(img)
        writer.release()

    def write_obj(self, filename, v_pos, t_pos_idx, v_tex, t_tex_idx):
        with open(filename, 'w') as f:
            for v in v_pos:
                f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

            if v_tex is not None:
                assert (len(t_pos_idx) == len(t_tex_idx))
                for v in v_tex:
                    f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

            for i in range(len(t_pos_idx)):
                f.write('f ')
                for j in range(3):
                    f.write(
                        ' %s/%s' %
                        (str(t_pos_idx[i][j] + 1),
                         '' if v_tex is None else str(t_tex_idx[i][j] + 1)))
                f.write('\n')

    def save_obj(self, filename, v_pos, t_pos_idx, v_tex=None, t_tex_idx=None):

        v_pos = v_pos.cpu().numpy()
        t_pos_idx = t_pos_idx.cpu().numpy()
        save_dir = os.path.dirname(filename)
        os.makedirs(save_dir, exist_ok=True)
        if v_tex is not None and t_tex_idx is not None:
            v_tex = v_tex.cpu().numpy()
            t_tex_idx = t_tex_idx.cpu().numpy()
            self.write_obj(filename, v_pos, t_pos_idx, v_tex, t_tex_idx)
        else:
            self.write_obj(filename, v_pos, t_pos_idx, v_tex, t_tex_idx)
