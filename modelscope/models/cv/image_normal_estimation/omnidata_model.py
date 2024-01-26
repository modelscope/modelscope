# Copyright (c) Alibaba, Inc. and its affiliates.
# Model: Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans
# Paper link: https://arxiv.org/pdf/2110.04994.pdf
import os.path as osp

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.image_normal_estimation.modules.midas.dpt_depth import \
    DPTDepthModel
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.image_normal_estimation,
    module_name=Models.omnidata_normal_estimation)
class OmnidataNormalEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        # build model
        self.model = DPTDepthModel(
            backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
        # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, inputs):
        return self.model(inputs['imgs']).clamp(min=0, max=1)

    def postprocess(self, inputs):
        normal_result = inputs.flip(1)
        results = {OutputKeys.NORMALS: normal_result}
        return results

    def inference(self, data):
        results = self.forward(data)

        return results
