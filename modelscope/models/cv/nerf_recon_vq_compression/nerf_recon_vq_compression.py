# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import time
from functools import partial

import cv2
import numpy as np
import torch
import tqdm

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .dataloader import dataset_dict
from .network.tensoRF import TensorVM, raw2alpha
from .network.tensoRF_VQ import TensorVMSplitVQ
from .renderer import OctreeRender_trilinear_fast
from .renderer import evaluation as evaluation_render
from .renderer import render_path

logger = get_logger()

__all__ = ['NeRFReconVQCompression']


@MODELS.register_module(
    Tasks.nerf_recon_vq_compression,
    module_name=Models.nerf_recon_vq_compression)
class NeRFReconVQCompression(TorchModel):

    def __init__(self, model_dir=None, **kwargs):
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')

        self.device = torch.device('cuda')
        self.data_type = kwargs['dataset_name']
        self.data_dir = kwargs['data_dir']
        self.downsample = kwargs['downsample']
        self.ndc_ray = kwargs['ndc_ray']
        self.ckpt_path = os.path.join(model_dir, kwargs['ckpt_path'])

        if self.ckpt_path == '' or self.ckpt_path is None:
            self.ckpt_path = os.path.join(model_dir, 'ficus_demo.pt')
            if not os.path.exists(self.ckpt_path):
                raise Exception('ckpt path not found')

        # load model
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        model_kwargs = ckpt['kwargs']
        model_kwargs['device'] = self.device
        self.model = TensorVMSplitVQ(**model_kwargs)
        self.model.extreme_load(ckpt)

        self.renderer = OctreeRender_trilinear_fast

        # load data
        dataset = dataset_dict[self.data_type]
        self.test_dataset = dataset(
            self.data_dir,
            split='test',
            downsample=self.downsample,
            is_stack=True)

    def evaluation(self, render_dir, N_vis=-1):
        white_bg = self.test_dataset.white_bg
        ndc_ray = self.ndc_ray
        evaluation_test = partial(
            evaluation_render,
            test_dataset=self.test_dataset,
            renderer=self.renderer,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=self.device,
            compute_extra_metrics=True,
            im_save=True)

        logfolder = render_dir
        os.makedirs(f'{logfolder}/evalution_test', exist_ok=True)
        PSNRs = evaluation_test(
            tensorf=self.model,
            N_vis=N_vis,
            savePath=f'{logfolder}/evalution_test')
        logger.info(
            f'VQRF-Evaluation: {self.ckpt_path} mean PSNR: {np.mean(PSNRs)}')

    def render_path(self, render_dir, N_vis=120):
        white_bg = self.test_dataset.white_bg
        ndc_ray = self.ndc_ray

        logfolder = render_dir
        os.makedirs(f'{logfolder}/render_path', exist_ok=True)

        render_poses = self.get_render_pose(N_cameras=N_vis)
        render_path(
            self.test_dataset,
            self.model,
            render_poses,
            self.renderer,
            savePath=f'{logfolder}/render_path',
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            device=self.device)
        logger.info(
            f'VQRF-Render: {self.ckpt_path} render path video result saved in {logfolder}/render_path'
        )

    def get_render_pose(self, N_cameras=120):
        if self.data_type == 'blender':
            return self.test_dataset.get_render_pose(N_cameras=N_cameras)
        elif self.data_type == 'llff':
            return self.test_dataset.get_render_pose(N_cameras=N_cameras)
