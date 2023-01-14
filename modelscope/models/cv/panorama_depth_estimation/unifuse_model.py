# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.panorama_depth_estimation.networks import (Equi,
                                                                     UniFuse)
from modelscope.models.cv.panorama_depth_estimation.networks.util import \
    Equirec2Cube
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module(
    Tasks.panorama_depth_estimation,
    module_name=Models.unifuse_depth_estimation)
class PanoramaDepthEstimation(TorchModel):
    """
    UniFuse: Unidirectional Fusion for 360 Panorama Depth Estimation
    https://arxiv.org/abs/2102.03550
    """

    def __init__(self, model_dir: str, **kwargs):
        """
        Args:
            model_dir: the path of the pretrained model file
        """
        super().__init__(model_dir, **kwargs)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model {model_path}')
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        Net_dict = {'UniFuse': UniFuse, 'Equi': Equi}
        Net = Net_dict[model_dict['net']]
        self.w = model_dict['width']
        self.h = model_dict['height']
        self.max_depth_meters = 10.0
        self.e2c = Equirec2Cube(self.h, self.w, self.h // 2)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # build model
        self.model = Net(
            model_dict['layers'],
            model_dict['height'],
            model_dict['width'],
            max_depth=self.max_depth_meters,
            fusion_type=model_dict['fusion'],
            se_in_fusion=model_dict['se_in_fusion'])

        # load state dict
        self.model.to(self.device)
        model_state_dict = self.model.state_dict()
        self.model.load_state_dict(
            {k: v
             for k, v in model_dict.items() if k in model_state_dict})
        self.model.eval()

        logger.info(f'model init done! Device:{self.device}')

    def forward(self, Inputs):
        """
        Args:
            Inputs: model inputs containning equirectangular panorama images and the corresponding cubmap images
            The torch size of Inputs['rgb'] should be [n, 3, 512, 1024]
            The torch size of Inputs['cube_rgb'] should be [n, 3, 256, 1536]
        Returns:
            Unifuse model outputs containing the predicted equirectangular depth images in metric
        """
        equi_inputs = Inputs['rgb'].to(self.device)
        cube_inputs = Inputs['cube_rgb'].to(self.device)
        return self.model(equi_inputs, cube_inputs)

    def postprocess(self, Inputs):
        depth_result = Inputs['pred_depth'][0]
        results = {OutputKeys.DEPTHS: depth_result}
        return results
