# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.preprocessors import LoadImage
from modelscope.models.cv.facial_68ldk_detection import infer  
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

@MODELS.register_module(
    Tasks.facial_68ldk_detection, module_name=Models.star_68ldk_detection)
class FaceLandmarkDetection(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        
    def forward(self, Inputs):
        return Inputs
    
    def postprocess(self, Inputs):
        return Inputs
        
    def inference(self, data):
        return data