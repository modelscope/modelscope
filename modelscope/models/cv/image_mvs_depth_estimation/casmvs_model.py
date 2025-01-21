# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .cas_mvsnet import CascadeMVSNet
from .colmap2mvsnet import processing_single_scene
from .depth_filter import pcd_depth_filter
from .general_eval_dataset import MVSDataset, save_pfm
from .utils import (generate_pointcloud, numpy2torch, tensor2numpy, tocuda,
                    write_cam)

logger = get_logger()


@MODELS.register_module(
    Tasks.image_multi_view_depth_estimation,
    module_name=Models.image_casmvs_depth_estimation)
class ImageMultiViewDepthEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        # build model
        self.model = CascadeMVSNet(
            refine=False,
            ndepths=[48, 32, 8],
            depth_interals_ratio=[float(d_i) for d_i in [4, 2, 1]],
            share_cr=False,
            cr_base_chs=[8, 8, 8],
            grad_method='detach')

        # load checkpoint file
        ckpt_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict['model'], strict=True)

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

        logger.info('preprocess of making pair data start')
        processing_single_scene(args)
        logger.info('preprocess of making pair data done')

    def forward(self, inputs):
        test_dir = os.path.dirname(inputs['casmvs_inp_dir'])
        scene = os.path.basename(inputs['casmvs_inp_dir'])
        test_list = [scene]
        save_dir = inputs['casmvs_res_dir']

        logger.info('depth estimation start')

        test_dataset = MVSDataset(
            test_dir,
            test_list,
            'test',
            5,
            192,
            1.06,
            max_h=1200,
            max_w=1200,
            fix_res=False)

        with torch.no_grad():
            for batch_idx, sample in enumerate(test_dataset):
                sample = numpy2torch(sample)

                if self.device == 'cuda':
                    sample_cuda = tocuda(sample)

                proj_matrices_dict = sample_cuda['proj_matrices']
                proj_matrices_dict_new = {}
                for k, v in proj_matrices_dict.items():
                    proj_matrices_dict_new[k] = v.unsqueeze(0)

                outputs = self.model(sample_cuda['imgs'].unsqueeze(0),
                                     proj_matrices_dict_new,
                                     sample_cuda['depth_values'].unsqueeze(0))

                outputs = tensor2numpy(outputs)
                del sample_cuda
                filenames = [sample['filename']]
                cams = sample['proj_matrices']['stage{}'.format(3)].unsqueeze(
                    0).numpy()
                imgs = sample['imgs'].unsqueeze(0).numpy()

                # save depth maps and confidence maps
                for filename, cam, img, depth_est, photometric_confidence in zip(
                        filenames, cams, imgs, outputs['depth'],
                        outputs['photometric_confidence']):

                    img = img[0]  # ref view
                    cam = cam[0]  # ref cam
                    depth_filename = os.path.join(
                        save_dir, filename.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(
                        save_dir, filename.format('confidence', '.pfm'))
                    cam_filename = os.path.join(
                        save_dir, filename.format('cams', '_cam.txt'))
                    img_filename = os.path.join(
                        save_dir, filename.format('images', '.jpg'))
                    ply_filename = os.path.join(
                        save_dir, filename.format('ply_local', '.ply'))
                    os.makedirs(
                        depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(
                        confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                    # save depth maps
                    save_pfm(depth_filename, depth_est)
                    # save confidence maps
                    save_pfm(confidence_filename, photometric_confidence)
                    # save cams, img
                    write_cam(cam_filename, cam)
                    img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0,
                                  255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(img_filename, img_bgr)

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
