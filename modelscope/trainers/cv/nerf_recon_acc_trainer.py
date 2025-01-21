# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import os.path as osp
import random
import time
from datetime import datetime
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from modelscope.metainfo import Trainers
from modelscope.models.cv.nerf_recon_acc import NeRFReconPreprocessor
from modelscope.models.cv.nerf_recon_acc.dataloader.nerf_dataset import (
    BlenderDataset, ColmapDataset)
from modelscope.models.cv.nerf_recon_acc.network.nerf import NeRFModel
from modelscope.models.cv.nerf_recon_acc.network.segmenter import \
    ObjectSegmenter
from modelscope.models.cv.nerf_recon_acc.network.utils import PSNR
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.nerf_recon_acc)
class NeRFReconAccTrainer(BaseTrainer):
    """initialize the acceleration version of nerf reconstruction model for object.

    Args:
        model (str): the model path.
        cfg_file (str): cfg json file
        data_type (str): only support 'blender' or 'colmap'
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
        work_dir (str): dir to save ckpt and other results
        render_images (bool): whether to render test image after training
        save_mesh (bool): whether to save the reconstructed mesh of object, default False
        save_ckpt (bool): whether to save the checkpoints in data_dir, default False
        network_cfg (dict): args of network config
        match_type (str): colmap feature matching type, only for colmap data
        frame_count (str): extract number of frames, only for video input
        use_distortion (bool): whether run colmap undistortion
    """

    def __init__(self,
                 model: str,
                 cfg_file: str = None,
                 data_type=None,
                 use_mask=None,
                 max_step=None,
                 train_num_rays=None,
                 max_train_num_rays=None,
                 log_every_n_steps=None,
                 work_dir=None,
                 render_images=None,
                 save_ckpt=None,
                 frame_count=None,
                 use_distortion=None,
                 *args,
                 **kwargs):

        model = self.get_or_download_model_dir(model)
        self.model_dir = model
        if cfg_file is None:
            cfg_file = osp.join(model, ModelFile.CONFIGURATION)
        super().__init__(cfg_file)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')

        self.params = {}
        self._override_params_from_file()
        for key, value in kwargs.items():
            self.params[key] = value
        if data_type is not None:
            self.params['data_type'] = data_type
        if use_mask is not None:
            self.params['use_mask'] = use_mask
        if max_step is not None:
            self.params['max_step'] = max_step
        if train_num_rays is not None:
            self.params['train_num_rays'] = train_num_rays
        if max_train_num_rays is not None:
            self.params['max_train_num_rays'] = max_train_num_rays
        if log_every_n_steps is not None:
            self.params['log_every_n_steps'] = log_every_n_steps
        if work_dir is not None:
            self.params['work_dir'] = work_dir
        if render_images is not None:
            self.params['render_images'] = render_images
        if save_ckpt is not None:
            self.params['save_ckpt'] = save_ckpt
        if frame_count is not None:
            self.params['frame_count'] = frame_count
        if use_distortion is not None:
            self.params['use_distortion'] = use_distortion

        self.data_type = self.params['data_type']
        if self.data_type != 'blender' and self.data_type != 'colmap':
            raise Exception('data type {} is not support currently'.format(
                self.data_type))

        self.use_mask = self.params['use_mask']
        self.max_step = self.params['max_step']
        self.train_num_rays = self.params['train_num_rays']
        self.num_samples_per_ray = self.params['num_samples_per_ray']
        self.train_num_samples = self.train_num_rays * self.num_samples_per_ray
        self.max_train_num_rays = self.params['max_train_num_rays']
        self.test_ray_chunk = self.params['test_ray_chunk']
        self.dynamic_ray_sampling = self.params['dynamic_ray_sampling']
        self.max_size = self.params['max_size']
        self.n_test_traj_steps = self.params['n_test_traj_steps']
        self.log_every_n_steps = self.params['log_every_n_steps']
        self.render_images = self.params['render_images']
        self.save_mesh = self.params['save_mesh']
        self.save_ckpt = self.params['save_ckpt']
        self.work_dir = self.params['work_dir']
        self.network_cfg = self.params['network_cfg']
        self.match_type = self.params['match_type']
        self.frame_count = self.params['frame_count']
        self.use_distortion = self.params['use_distortion']
        logger.info('params:{}'.format(self.params))

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.preprocessor = NeRFReconPreprocessor(
            data_type=self.data_type,
            use_mask=self.use_mask,
            match_type=self.match_type,
            frame_count=self.frame_count,
            use_distortion=self.use_distortion)

        if self.use_mask and self.data_type == 'colmap':
            segment_path = os.path.join(self.model_dir, 'matting.pb')
            self.segmenter = ObjectSegmenter(segment_path)

        if self.data_type == 'blender':
            self.img_wh = (800, 800)
            self.network_cfg['radius'] = 1.5
            self.background = 'white'
            self.network_cfg['background'] = 'white'
        elif self.data_type == 'colmap':
            self.img_wh = None
            self.max_size = self.max_size
            self.n_test_traj_steps = self.n_test_traj_steps
            self.network_cfg['radius'] = 0.5
            if self.use_mask:
                self.background = 'white'
                self.network_cfg['background'] = 'white'
                logger.info('run nerf with mask data')
            else:
                self.background = 'random'
                self.network_cfg['background'] = 'random'
                logger.info('run nerf without mask data')

        logger.info(self.network_cfg)

        self.model = NeRFModel(
            self.network_cfg,
            num_samples_per_ray=self.num_samples_per_ray,
            test_ray_chunk=self.test_ray_chunk).cuda()
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

    def _override_params_from_file(self):

        self.params['data_type'] = self.cfg['train']['data_type']
        self.params['use_mask'] = self.cfg['train']['use_mask']
        self.params['max_step'] = self.cfg['train']['max_step']
        self.params['train_num_rays'] = self.cfg['train']['train_num_rays']
        self.params['max_train_num_rays'] = self.cfg['train'][
            'max_train_num_rays']
        self.params['dynamic_ray_sampling'] = self.cfg['train'][
            'dynamic_ray_sampling']
        self.params['log_every_n_steps'] = self.cfg['train'][
            'log_every_n_steps']
        self.params['render_images'] = self.cfg['train']['render_images']
        self.params['save_ckpt'] = self.cfg['train']['save_ckpt']
        self.params['work_dir'] = self.cfg['train']['work_dir']

        self.params['num_samples_per_ray'] = self.cfg['model'][
            'num_samples_per_ray']
        self.params['test_ray_chunk'] = self.cfg['model']['test_ray_chunk']
        self.params['max_size'] = self.cfg['model']['max_size']
        self.params['n_test_traj_steps'] = self.cfg['model'][
            'n_test_traj_steps']
        self.params['save_mesh'] = self.cfg['model']['save_mesh']
        self.params['network_cfg'] = self.cfg['model']['network_cfg']

        self.params['match_type'] = self.cfg['preprocessor']['match_type']
        self.params['frame_count'] = self.cfg['preprocessor']['frame_count']
        self.params['use_distortion'] = self.cfg['preprocessor'][
            'use_distortion']

    def train(self, *args, **kwargs):
        logger.info('Begin nerf reconstruction training')
        processor_input = {}
        if self.data_type == 'blender':
            if 'data_dir' not in kwargs:
                raise Exception(
                    'Please specify data_dir of nerf_synthetic data')
            data_dir = kwargs['data_dir']
            processor_input['data_dir'] = data_dir
            processor_input['video_input_path'] = ''

        if self.data_type == 'colmap':
            if 'video_input_path' in kwargs:
                video_input_path = kwargs['video_input_path']
                processor_input['data_dir'] = self.work_dir
                processor_input['video_input_path'] = video_input_path
            elif 'data_dir' in kwargs:
                data_dir = kwargs['data_dir']
                images_dir = os.path.join(data_dir, 'images')
                if os.path.exists(images_dir):
                    image_list = glob.glob('{}/*.*g'.format(images_dir))
                    if len(image_list) == 0:
                        raise Exception('no images found in images dir')
                    else:
                        processor_input['data_dir'] = data_dir
                        processor_input['video_input_path'] = ''
                else:
                    raise Exception('images dir not found in data_dir')
            else:
                raise Exception(
                    'Please specify video_path or images path for colmap process'
                )

        processor_output = self.preprocessor(processor_input)
        data_dir = processor_output['data_dir']
        logger.info(
            'nerf reconstruction preprocess done, data_dir is {}'.format(
                data_dir))

        if self.data_type == 'colmap' and self.use_mask:
            if os.path.exists(os.path.join(data_dir, 'preprocess')):
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

        if self.save_ckpt:
            save_ckpt_name = os.path.join(self.work_dir, 'model.ckpt')
            torch.save(
                {
                    'global_step': self.max_step,
                    'network_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_ckpt_name)
            logger.info(
                'save checkpoints done, saved as {}'.format(save_ckpt_name))
        if self.render_images:
            save_video_path = os.path.join(self.work_dir, 'render.mp4')
            self.render_video(self.work_dir, save_video_path)

        logger.info('NeRF reconstruction finish')

    def evaluate(self,
                 checkpoint_path: Optional[str] = None,
                 *args,
                 **kwargs) -> Dict[str, float]:
        """evaluate a dataset

        evaluate a dataset via a specific model from the `checkpoint_path` path, if the `checkpoint_path`
        does not exist, read from the config file.

        Args:
            checkpoint_path (Optional[str], optional): the model path. Defaults to None.

        Returns:
            Dict[str, float]: the results about the evaluation
            Example:
            {"accuracy": 0.5091743119266054, "f1": 0.673780487804878}
        """
        raise NotImplementedError('evaluate is not supported currently')

    def render_video(self, save_dir, save_video_path):
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
                save_img_dir = os.path.join(save_dir, 'render')
                os.makedirs(save_img_dir, exist_ok=True)
                save_img_path = os.path.join(save_img_dir, f'{i:d}.png')
                self.save_image(save_img_path, img)

            self.save_video(save_video_path, save_img_dir)
            logger.info('test psnr: {}'.format(psnr / len(self.test_dataset)))
            logger.info(
                'save render video done. saved as {}'.format(save_video_path))

            if self.save_mesh:
                mesh = self.model.isosurface()
                save_mesh_path = os.path.join(save_dir, 'render.obj')
                self.save_obj(save_mesh_path, mesh['v_pos'], mesh['t_pos_idx'])
                logger.info('save render mesh done. saved as {}'.format(
                    save_mesh_path))

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
