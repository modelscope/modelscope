# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
import torch
from yacs.config import CfgNode as CN

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.indoor_layout_estimation.networks.panovit import \
    PanoVIT
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.indoor_layout_estimation,
    module_name=Models.panovit_layout_estimation)
class LayoutEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        config = CN()
        config.model = CN()
        config.model.kwargs = CN(new_allowed=True)
        config.defrost()
        config_path = osp.join(model_dir, ModelFile.YAML_FILE)
        config.merge_from_file(config_path)
        config.freeze()
        # build model
        self.model = PanoVIT(**config.model.kwargs)

        # load model
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        if torch.cuda.is_available():
            state_dict = torch.load(model_path)
        else:
            state_dict = torch.load(model_path, map_location='cpu')

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, Inputs):
        return self.model.infer(Inputs['images'])

    def postprocess(self, Inputs):
        image, y_bon, y_cor = Inputs['image'], Inputs['y_bon_'], Inputs[
            'y_cor_']
        layout_image = self.model.postprocess(image[0, 0:3], y_bon, y_cor)
        return layout_image

    def inference(self, data):
        results = self.forward(data)
        return results
