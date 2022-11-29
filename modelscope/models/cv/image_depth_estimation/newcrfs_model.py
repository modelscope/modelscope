# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_depth_estimation.networks.newcrf_depth import \
    NewCRFDepth
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.image_depth_estimation, module_name=Models.newcrfs_depth_estimation)
class DepthEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        # build model
        self.model = NewCRFDepth(
            version='large07', inv_depth=False, max_depth=10)

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = torch.load(model_path)

        state_dict = {}
        for k in checkpoint['model'].keys():
            if k.startswith('module.'):
                state_dict[k[7:]] = checkpoint['model'][k]
            else:
                state_dict[k] = checkpoint['model'][k]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, Inputs):
        return self.model(Inputs['imgs'])

    def postprocess(self, Inputs):
        depth_result = Inputs

        results = {OutputKeys.DEPTHS: depth_result}
        return results

    def inference(self, data):
        results = self.forward(data)

        return results
