# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.s2net_panorama_depth_estimation.networks import (
    EffnetSphDecoderNet, ResnetSphDecoderNet, SwinSphDecoderNet)
from modelscope.models.cv.s2net_panorama_depth_estimation.networks.config import \
    get_config
from modelscope.models.cv.s2net_panorama_depth_estimation.networks.util_helper import (
    compute_hp_info, render_depth_map)
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module(
    Tasks.panorama_depth_estimation, module_name=Models.s2net_depth_estimation)
class PanoramaDepthEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """
         An end-to-end deep network for monocular panorama depth estimation on a unit spherical surface.
         This is the  official implementation of paper S2Net: Accurate Panorama Depth Estimation on Spherical Surface,
        https://arxiv.org/abs/2301.05845.
        Args:
            model_dir: the path of the pretrained model file
        """
        super().__init__(model_dir, **kwargs)
        if 'device' in kwargs:
            self.device = create_device(kwargs['device'])
        else:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

        # load network config
        cfg_path = osp.join(model_dir, 'model.yaml')
        cfg = get_config(cfg_path)
        encoder_model_dict = {
            'swin': SwinSphDecoderNet,
            'resNet': ResnetSphDecoderNet,
            'effnet': EffnetSphDecoderNet
        }

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model {model_path}')
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))

        self.w = cfg.DATA.IMG_HEIGHT
        self.h = cfg.DATA.IMG_WIDTH
        self.max_depth_meters = 10.0
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # build model
        model_type = encoder_model_dict[cfg.BACKBONE.TYPE]
        self.model = model_type(cfg, pretrained=False)

        # load state dict
        self.model.to(self.device)
        self.model.load_state_dict(model_dict['model'], strict=True)
        self.model.eval()

        nside = 128
        self.hp_info = compute_hp_info(
            nside, (cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))

        logger.info(f'model init done! Device:{self.device}')

    def forward(self, rgb):
        """
        Args:
            rgb:  equirectangular panorama images
            The torch size of rgb should be [n, 3, 512, 1024]
        Returns:
            S2net model outputs containing the predicted equirectangular depth images in metric
        """
        equi_inputs = rgb.to(self.device)
        return self.model(equi_inputs)

    def postprocess(self, pred_depths_hp):
        depth_maps = render_depth_map(pred_depths_hp,
                                      self.hp_info['image_to_sp_map'])[0]
        results = {OutputKeys.DEPTHS: depth_maps}
        return results
