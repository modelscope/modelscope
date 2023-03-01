# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import time

import cv2
import numpy as np
import torch
import tqdm

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .dataloader.nerf_dataset import BlenderDataset, ColmapDataset
from .network.nerf import NeRFModel
from .network.utils import PSNR

logger = get_logger()

__all__ = ['NeRFReconAcc']


@MODELS.register_module(
    Tasks.nerf_recon_acc, module_name=Models.nerf_recon_acc)
class NeRFReconAcc(TorchModel):

    def __init__(self, model_dir, network_cfg, **kwargs):
        """initialize the acceleration version of nerf reconstruction model for object.
         NeRFReconAcc accelerate single object reconstruction time from ~10hours to ~10min.

        Args:
            model_dir (str): the model path.
            data_type (str): only support 'blender' or 'colmap'
            use_mask (bool): whether use mask of objects, default True
            num_samples_per_ray (int): sampling numbers for each ray, default 1024
            test_ray_chunk (int): chunk size for rendering, default 1024
            max_size (int): max size of (width, height) when training, default 800
            n_test_traj_steps (int): number of testing images, default 120
            log_every_n_steps (int): print log info every n steps, default 1000
            save_mesh (bool): whether to save the reconstructed mesh of object, default False
            network_cfg (dict): args of network config
        """
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')
        logger.info('model params:{}'.format(kwargs))
        self.data_type = kwargs['data_type']
        self.use_mask = kwargs['use_mask']
        self.num_samples_per_ray = kwargs['num_samples_per_ray']
        self.test_ray_chunk = kwargs['test_ray_chunk']
        self.save_mesh = kwargs['save_mesh']
        self.ckpt_path = kwargs['ckpt_path']

        if self.ckpt_path == '':
            self.ckpt_path = os.path.join(model_dir, 'model.ckpt')
            if not os.path.exists(self.ckpt_path):
                raise Exception('ckpt path not found')

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

        self.model = NeRFModel(
            network_cfg,
            num_samples_per_ray=self.num_samples_per_ray,
            test_ray_chunk=self.test_ray_chunk).cuda()

        checkpoints = torch.load(self.ckpt_path)
        self.model.load_state_dict(checkpoints['network_state_dict'])
        self.model = self.model.cuda()
        self.model.eval()

        self.criterions = PSNR()

    def nerf_reconstruction(self, data_dir, render_dir):
        if self.data_type == 'blender':
            self.test_dataset = BlenderDataset(
                root_fp=data_dir,
                split='test',
                img_wh=self.img_wh,
            )

        elif self.data_type == 'colmap':
            self.test_dataset = ColmapDataset(
                root_fp=data_dir,
                split='test',
                img_wh=self.img_wh,
                max_size=self.max_size,
                n_test_traj_steps=self.n_test_traj_steps,
            )

        tic_start = time.time()
        os.makedirs(render_dir, exist_ok=True)
        logger.info('save render path: {}.'.format(render_dir))
        self.render_video(render_dir)
        tic_end = time.time()
        duration = tic_end - tic_start
        logger.info('reconstruction done, cost time: {}'.format(duration))

    def render_video(self, render_dir):
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
                save_img_dir = os.path.join(render_dir, 'render')
                os.makedirs(save_img_dir, exist_ok=True)
                save_img_path = os.path.join(save_img_dir, f'{i:d}.png')
                self.save_image(save_img_path, img)

            save_video_path = os.path.join(render_dir, 'render.mp4')
            self.save_video(save_video_path, save_img_dir)
            logger.info('test psnr: {}'.format(psnr / len(self.test_dataset)))
            logger.info('save render video done.')

            if self.save_mesh:
                mesh = self.model.isosurface()
                save_mesh_path = os.path.join(render_dir, 'render.obj')
                self.save_obj(save_mesh_path, mesh['v_pos'], mesh['t_pos_idx'])
                logger.info('save render mesh done.')

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
