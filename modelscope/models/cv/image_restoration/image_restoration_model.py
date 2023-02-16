# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import cv2
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .demoire_models import model_map


@MODELS.register_module(
    Tasks.image_demoireing, module_name=Models.image_restoration)
class ImageRestorationModel(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)
        config = Config.from_file(config_path)
        model_name = config.model.network_type
        model_class = model_map[model_name]
        self.model = model_class(**config.model.network_param)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.pad_32 = config.preprocessor.pad_32

    def inference(self, data):
        """data is tensor -1 * C * H * W ---> return tensor -1 * C * H * W ."""
        if next(self.model.parameters()).is_cuda:
            data = data.to(
                torch.device([next(self.model.parameters()).device][0]))
        with torch.no_grad():
            results = self.model(data)
        if next(self.model.parameters()).is_cuda:
            return results[0].cpu()
        return results[0]

    def forward(self, inputs):
        """inputs is dict"""
        data = self.inference(inputs['img'])
        outputs = inputs
        outputs['img'] = data
        return outputs

    def postprocess(self, inputs):
        """ inputs is dict return is numpy"""
        data = inputs['img'][0, :, :, :]
        if self.pad_32:
            h_pad = inputs['h_pad']
            h_odd_pad = inputs['h_odd_pad']
            w_pad = inputs['w_pad']
            w_odd_pad = inputs['w_odd_pad']
            if h_pad != 0:
                data = data[:, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                data = data[:, :, w_pad:-w_odd_pad]
        data_norm_np = np.array(np.clip(data.numpy(), 0, 1)
                                * 255).astype('uint8').transpose(1, 2, 0)
        if data_norm_np.shape[0] != inputs['img_h']:
            data_norm_np = cv2.resize(data_norm_np,
                                      (inputs['img_w'], inputs['img_h']))
        return data_norm_np
