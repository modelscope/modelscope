# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import time

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .colmap2mvsnet import processing_single_scene
from .depth_filter import pcd_depth_filter
from .general_eval_dataset import MVSDataset, save_pfm
from .models.geomvsnet import GeoMVSNet
from .models.utils import *
from .models.utils.opts import get_opts
from .utils import (generate_pointcloud, numpy2torch, tensor2numpy, tocuda,
                    write_cam)

logger = get_logger()


@MODELS.register_module(
    Tasks.image_multi_view_depth_estimation,
    module_name=Models.image_geomvsnet_depth_estimation)
class GeoMVSNetDepthEstimation(TorchModel):
    '''
    GeoMVSNet is a state-of-the-art MVS(multi-view stereo) depth estimation method.
    For more details, please refer to https://github.com/doublez0108/geomvsnet
    '''

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        self.n_views = 5
        self.levels = 4
        self.hypo_plane_num_stages = '8,8,4,4'
        self.depth_interal_ratio_stages = '0.5,0.5,0.5,1'
        self.feat_base_channel = 8
        self.reg_base_channel = 8
        self.group_cor_dim_stages = '8,8,4,4'
        self.batch_size = 1

        self.model = GeoMVSNet(
            levels=self.levels,
            hypo_plane_num_stages=[
                int(n) for n in self.hypo_plane_num_stages.split(',')
            ],
            depth_interal_ratio_stages=[
                float(ir) for ir in self.depth_interal_ratio_stages.split(',')
            ],
            feat_base_channel=self.feat_base_channel,
            reg_base_channel=self.reg_base_channel,
            group_cor_dim_stages=[
                int(n) for n in self.group_cor_dim_stages.split(',')
            ],
        )

        # load checkpoint file
        ckpt_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict['model'], strict=False)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model.to(self.device)
        self.model.eval()
        logger.info(f'model init done! Device:{self.device}')

    def preprocess_make_pair(self, inputs):

        data = inputs['input_dir']
        casmvs_inp_dir = inputs['casmvs_inp_dir']

        args = edict()
        args.dense_folder = data
        args.save_folder = casmvs_inp_dir
        args.max_d = 192
        args.interval_scale = 1.06
        args.theta0 = 5
        args.sigma1 = 1
        args.sigma2 = 10
        args.model_ext = '.bin'

        logger.info('preprocess of making pair data start, folder: %s',
                    args.dense_folder)
        processing_single_scene(args)
        logger.info('preprocess of making pair data done')

    def forward(self, inputs):

        test_dir = os.path.dirname(inputs['casmvs_inp_dir'])
        scene = os.path.basename(inputs['casmvs_inp_dir'])
        test_list = [scene]
        save_dir = inputs['casmvs_res_dir']

        logger.info('depth estimation start')

        test_dataset = MVSDataset(
            test_dir, test_list, 'test', self.n_views, max_wh=(1600, 1200))
        TestImgLoader = DataLoader(
            test_dataset,
            self.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False)

        total_time = 0
        with torch.no_grad():
            for batch_idx, sample in enumerate(TestImgLoader):
                sample_cuda = tocuda(sample)

                # @Note GeoMVSNet main
                start_time = time.time()
                outputs = self.model(sample_cuda['imgs'],
                                     sample_cuda['proj_matrices'],
                                     sample_cuda['intrinsics_matrices'],
                                     sample_cuda['depth_values'],
                                     sample['filename'])
                end_time = time.time()
                total_time += end_time - start_time

                outputs = tensor2numpy(outputs)
                del sample_cuda
                filenames = sample['filename']
                cams = sample['proj_matrices']['stage{}'.format(
                    self.levels)].numpy()
                imgs = sample['imgs']
                logger.info('Iter {}/{}, Time:{:.3f} Res:{}'.format(
                    batch_idx, len(TestImgLoader), end_time - start_time,
                    imgs[0].shape))

                for filename, cam, img, depth_est, photometric_confidence in zip(
                        filenames, cams, imgs, outputs['depth'],
                        outputs['photometric_confidence']):
                    img = img[0].numpy()  # ref view
                    cam = cam[0]  # ref cam

                    depth_filename = os.path.join(
                        save_dir, filename.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(
                        save_dir, filename.format('confidence', '.pfm'))
                    cam_filename = os.path.join(
                        save_dir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(
                        save_dir, filename.format('images', '.jpg'))
                    os.makedirs(
                        depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(
                        confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)

                    # save depth maps
                    save_pfm(depth_filename, depth_est)

                    # save confidence maps
                    confidence_list = [
                        outputs['stage{}'.format(i)]
                        ['photometric_confidence'].squeeze(0)
                        for i in range(1, self.levels + 1)
                    ]
                    print('confidence_list', len(confidence_list))
                    photometric_confidence = confidence_list[-1]
                    save_pfm(confidence_filename, photometric_confidence)

                    # save camera info
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0,
                                  255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

        torch.cuda.empty_cache()
        logger.info('depth estimation end')
        return inputs

    def postprocess(self, inputs):
        test_dir = os.path.dirname(inputs['casmvs_inp_dir'])
        scene = os.path.basename(inputs['casmvs_inp_dir'])
        logger.info('depth fusion start')
        pcd = pcd_depth_filter(
            scene, test_dir, inputs['casmvs_res_dir'], thres_view=4)
        logger.info('depth fusion end')
        return pcd
