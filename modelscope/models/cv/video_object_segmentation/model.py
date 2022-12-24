# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_object_segmentation.eval_network import RDE_VOS
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.video_object_segmentation,
    module_name=Models.video_object_segmentation)
class VideoObjectSegmentation(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        params = torch.load(model_path, map_location='cpu')
        self.model = RDE_VOS()
        self.model.load_state_dict(params, strict=True)
        self.model.eval()

    def forward(self, inputs: Dict[str, Any]):
        return self.model(inputs)
