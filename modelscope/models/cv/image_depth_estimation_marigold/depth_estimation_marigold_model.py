# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path

import torch
import numpy as np
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .networks.marigold_model import MarigoldPipeline

logger = get_logger()
__all__ = ['DepthEstimationMarigoldModel']


@MODELS.register_module(
    Tasks.image_depth_estimation, module_name=Models.marigold_depth_estimation)
class DepthEstimationMarigoldModel(TorchModel):
    """ Depth estimation model marigold, implemented from paper https://arxiv.org/abs/2312.02145.
        The network utilizes pre-trained text-to-image model to finetune depth estimation network.
        The marigold model is composed with a finetuned protocol to obtain better estimated depth
    """

    def __init__(self, model_dir=None, **kwargs):
        """initialize the marigold model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            focal: focal length, pictures that do not work are input according to
                the camera setting value at the time of shooting
            dataset: used to set focal value according dataset type, only support 'kitti'
        """
        super().__init__(model_dir, **kwargs)

        if not torch.cuda.is_available():
            raise Exception('GPU is required')
        self.model_id = 'Damo_XR_Lab/cv_marigold_monocular-depth-estimation'
        self.gpu_id = 0
        self.device = torch.device('cuda:%d' % self.gpu_id)
        self.checkpoint_path = os.path.join(model_dir, 'Marigold_v1_merged_2')
        self.dtype = torch.float16

        self.pipe = MarigoldPipeline.from_pretrained(self.checkpoint_path, torch_type=self.dtype)
        self.pipe.to(self.device)

    def forward(self, inputs):
        # print(inputs.size)
        return self.pipe(inputs)

    def postprocess(self, inputs):
        # print('model postprocess')
        # print('model postprocess inputs', inputs)
        depth_pred: np.ndarray = inputs.depth_np
        depth_colored: Image.Image = inputs.depth_colored
        results = {OutputKeys.DEPTHS: depth_pred,
                   OutputKeys.DEPTHS_COLOR: depth_colored}
        return results

    def inference(self, data):
        results = self.forward(data)
        return results
