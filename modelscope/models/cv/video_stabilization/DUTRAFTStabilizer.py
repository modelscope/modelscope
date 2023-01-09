# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import math
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_stabilization.DUT.config import cfg
from modelscope.models.cv.video_stabilization.DUT.DUT_raft import DUT
from modelscope.preprocessors.cv import VideoReader, stabilization_preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

__all__ = ['DUTRAFTStabilizer']


@MODELS.register_module(
    Tasks.video_stabilization, module_name=Models.video_stabilization)
class DUTRAFTStabilizer(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video stabilization model from the `model_dir` path.
        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        SmootherPath = os.path.join(self.model_dir,
                                    self.config.modelsetting.SmootherPath)
        RFDetPath = os.path.join(self.model_dir,
                                 self.config.modelsetting.RFDetPath)
        RAFTPath = os.path.join(self.model_dir,
                                self.config.modelsetting.RAFTPath)
        MotionProPath = os.path.join(self.model_dir,
                                     self.config.modelsetting.MotionProPath)
        homo = self.config.modelsetting.homo
        args = self.config.modelsetting.args
        self.base_crop_width = self.config.modelsetting.base_crop_width

        self.net = DUT(
            SmootherPath=SmootherPath,
            RFDetPath=RFDetPath,
            RAFTPath=RAFTPath,
            MotionProPath=MotionProPath,
            homo=homo,
            args=args)

        self.net.cuda()
        self.net.eval()

    def _inference_forward(self, input: str) -> Dict[str, Any]:
        data = stabilization_preprocessor(input, cfg)
        with torch.no_grad():
            origin_motion, smooth_path = self.net.inference(
                data['x'].cuda(), data['x_rgb'].cuda(), repeat=50)

        origin_motion = origin_motion.cpu().numpy()
        smooth_path = smooth_path.cpu().numpy()
        origin_motion = np.transpose(origin_motion[0], (2, 3, 1, 0))
        smooth_path = np.transpose(smooth_path[0], (2, 3, 1, 0))

        return {
            'origin_motion': origin_motion,
            'smooth_path': smooth_path,
            'ori_images': data['ori_images'],
            'fps': data['fps'],
            'width': data['width'],
            'height': data['height'],
            'base_crop_width': self.base_crop_width
        }

    def forward(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """return the result by the model
        Args:
            inputs (str): the input video path
        Returns:
            Dict[str, str]: results
        """
        return self._inference_forward(inputs['input'][0])
